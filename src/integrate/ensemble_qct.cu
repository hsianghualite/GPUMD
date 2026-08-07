/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
The quasi-classical trajectory (QCT) integrator.

It propagates either a supplied phase point or a harmonic normal-mode sample
with the same GPU velocity-Verlet scheme as NVE.
------------------------------------------------------------------------------*/

#include "ensemble_qct.cuh"
#include "model/read_xyz.cuh"
#include "phonon/molecular_hessian.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace
{
void qct_input_error(const std::string& message)
{
  PRINT_INPUT_ERROR(message.c_str());
}

std::vector<std::string> get_qct_tokens(const std::string& line)
{
  std::vector<std::string> tokens = get_tokens(line);
  std::vector<std::string> tokens_without_comments;
  for (const auto& token : tokens) {
    if (!token.empty() && token[0] == '#') {
      break;
    }
    tokens_without_comments.emplace_back(token);
  }
  return tokens_without_comments;
}

bool read_next_qct_tokens(
  std::ifstream& input,
  std::vector<std::string>& tokens,
  int& line_number)
{
  std::string line;
  while (std::getline(input, line)) {
    ++line_number;
    tokens = get_qct_tokens(line);
    if (!tokens.empty()) {
      return true;
    }
  }
  return false;
}

bool parse_yes_no(const std::string& token, const std::string& keyword)
{
  if (token == "yes") {
    return true;
  }
  if (token == "no") {
    return false;
  }
  qct_input_error(keyword + " should be yes or no.");
  return false;
}

void require_token_count(
  const std::vector<std::string>& tokens,
  const int expected,
  const int line_number,
  const std::string& context)
{
  if (tokens.size() != expected) {
    qct_input_error(
      context + " at line " + std::to_string(line_number) + " has an invalid number of fields.");
  }
}

void require_exact_token(
  const std::string& value,
  const std::string& expected,
  const std::string& keyword)
{
  if (value != expected) {
    qct_input_error(keyword + " should be " + expected + ".");
  }
}

const double THZ_TO_NATURAL_ANGULAR_FREQUENCY =
  2.0 * PI * 1.0e-3 * TIME_UNIT_CONVERSION;

// GPUMD's phonon Hessian (hessian.cu, molecular_hessian.cu) stores eigenvalues as
// omega^2 * 1e6 / TIME_UNIT_CONVERSION^2, which gives angular frequency squared in
// (rad/ps)^2.  To convert to ordinary frequency in THz (cycles/ps), divide by 2*pi.
// This factor converts sqrt(omega2_raw) [rad/ps] -> frequency_THz [cycles/ps = THz].
const double RAD_PER_PS_TO_THZ = 1.0 / (2.0 * PI);

std::uint64_t qct_sampling_seed(const std::uint64_t base_seed, const int attempt)
{
  if (attempt == 0) {
    return base_seed;
  }
  std::uint64_t value = base_seed + 0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(attempt);
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

void orient_eigenvector(std::vector<double>& eigenvector)
{
  size_t largest_component = 0;
  double largest_absolute_value = 0.0;
  for (size_t i = 0; i < eigenvector.size(); ++i) {
    const double absolute_value = std::fabs(eigenvector[i]);
    if (absolute_value > largest_absolute_value) {
      largest_absolute_value = absolute_value;
      largest_component = i;
    }
  }
  if (largest_absolute_value > 0.0 && eigenvector[largest_component] < 0.0) {
    for (double& component : eigenvector) {
      component = -component;
    }
  }
}

double evaluate_potential_energy(
  Force& force,
  Box& box,
  Atom& atom,
  std::vector<Group>& group,
  const std::vector<double>& position)
{
  atom.position_per_atom.copy_from_host(position.data());
  force.compute(
    box,
    atom.position_per_atom,
    atom.type,
    group,
    atom.potential_per_atom,
    atom.force_per_atom,
    atom.virial_per_atom);
  std::vector<double> potential(atom.number_of_atoms, 0.0);
  atom.potential_per_atom.copy_to_host(potential.data());
  double energy = 0.0;
  for (const double value : potential) {
    energy += value;
  }
  return energy;
}
} // namespace

Ensemble_QCT::Ensemble_QCT(const char** param, int num_param)
{
  type = -13;

  if (num_param == 2) {
    init_mode_ = Init_Mode::phase_point;
  } else if (strcmp(param[2], "phase_point") == 0) {
    parse_phase_point(param, num_param);
  } else if (
    strcmp(param[2], "harmonic") == 0 || strcmp(param[2], "canonical") == 0 ||
    strcmp(param[2], "microcanonical") == 0 || strcmp(param[2], "mode_energy") == 0 ||
    strcmp(param[2], "semiclassical") == 0 || strcmp(param[2], "wigner") == 0) {
    if (strcmp(param[2], "microcanonical") == 0) {
      sampling_mode_ = Sampling_Mode::microcanonical;
    } else if (strcmp(param[2], "mode_energy") == 0) {
      sampling_mode_ = Sampling_Mode::mode_energy;
    } else if (strcmp(param[2], "semiclassical") == 0) {
      sampling_mode_ = Sampling_Mode::semiclassical;
    } else if (strcmp(param[2], "wigner") == 0) {
      sampling_mode_ = Sampling_Mode::wigner;
    } else {
      sampling_mode_ = Sampling_Mode::canonical;
    }
    parse_harmonic(param, num_param);
  } else {
    PRINT_INPUT_ERROR(
      "ensemble qct mode should be phase_point, harmonic, canonical, microcanonical, mode_energy, semiclassical, or wigner.");
  }

  printf("Use QCT ensemble for this run.\n");
  if (init_mode_ == Init_Mode::phase_point) {
    printf("    propagate the supplied quasi-classical phase point with NVE dynamics.\n");
  } else if (mode_source_ == Mode_Source::automatic_hessian) {
    printf("    calculate molecular Hessian from the current model.xyz structure.\n");
    printf("    Hessian displacement is %g A.\n", hessian_displacement_);
  } else if (mode_source_ == Mode_Source::qct_modes) {
    printf("    initialize a harmonic quasi-classical phase point from %s.\n", modes_file_.c_str());
  } else {
    printf("    initialize a harmonic quasi-classical phase point from %s.\n", eigenvector_file_.c_str());
    printf("    exclude the lowest %d normal modes.\n", exclude_lowest_);
    printf("    minimum active mode frequency is %g THz.\n", min_frequency_);
  }
  if (init_mode_ == Init_Mode::harmonic) {
    if (sampling_mode_ == Sampling_Mode::canonical) {
      printf("    sampling mode is canonical.\n");
    } else if (sampling_mode_ == Sampling_Mode::microcanonical) {
      printf("    sampling mode is microcanonical.\n");
    } else if (sampling_mode_ == Sampling_Mode::mode_energy) {
      printf("    sampling mode is mode-energy.\n");
    } else if (sampling_mode_ == Sampling_Mode::semiclassical) {
      printf("    sampling mode is semiclassical EBK.\n");
    } else {
      printf("    sampling mode is Wigner (LSC-IVR).\n");
      printf("    anharmonic reweighting is %s.\n", anharmonic_reweight_ ? "enabled" : "disabled");
    }
    printf("    sample temperature is %g K.\n", sample_temperature_);
    printf("    random seed is %d.\n", seed_);
    printf("    zero-point energy is %s.\n", zpe_ ? "enabled" : "disabled");
    printf("    phase sampling is %s.\n", phase_mode_ == Phase_Mode::random ? "random" : "zero");
    if (stationary_point_ == Stationary_Point::automatic) {
      printf("    stationary-point classification is automatic.\n");
    } else if (stationary_point_ == Stationary_Point::minimum) {
      printf("    stationary-point classification is minimum.\n");
    } else {
      printf("    stationary-point classification is first-order saddle.\n");
    }
    if (reaction_direction_ == Reaction_Direction::positive) {
      printf("    reaction direction is positive.\n");
    } else if (reaction_direction_ == Reaction_Direction::negative) {
      printf("    reaction direction is negative.\n");
    } else {
      printf("    reaction direction is random.\n");
    }
  }
}

Ensemble_QCT::~Ensemble_QCT(void)
{
  // nothing now
}

void Ensemble_QCT::parse_phase_point(const char** param, int num_param)
{
  init_mode_ = Init_Mode::phase_point;
  if ((num_param - 3) % 2 != 0) {
    PRINT_INPUT_ERROR("ensemble qct phase_point should use optional key-value pairs.");
  }

  for (int i = 3; i < num_param; i += 2) {
    if (strcmp(param[i], "replicas") == 0) {
      replicas_ = get_int_from_token(param[i + 1], __FILE__, __LINE__);
      if (replicas_ != 1) {
        PRINT_INPUT_ERROR("ensemble qct phase_point currently supports replicas 1 only.");
      }
    } else {
      PRINT_INPUT_ERROR("Unknown keyword for ensemble qct phase_point.");
    }
  }
}

void Ensemble_QCT::parse_harmonic(const char** param, int num_param)
{
  init_mode_ = Init_Mode::harmonic;
  if (num_param <= 3 || (num_param - 3) % 2 != 0) {
    PRINT_INPUT_ERROR("ensemble qct harmonic should use key-value pairs.");
  }

  bool has_modes = false;
  bool has_eigenvector = false;
  bool has_exclude_lowest = false;
  bool has_min_frequency = false;
  bool has_hessian_displacement = false;
  bool has_temperature = false;
  bool has_energy = false;
  bool has_mode = false;
  bool has_vibrational_quantum = false;
  bool has_rotational_quantum = false;
  for (int i = 3; i < num_param; i += 2) {
    if (strcmp(param[i], "modes") == 0) {
      modes_file_ = param[i + 1];
      has_modes = true;
    } else if (strcmp(param[i], "eigenvector") == 0) {
      eigenvector_file_ = param[i + 1];
      has_eigenvector = true;
    } else if (strcmp(param[i], "exclude_lowest") == 0) {
      exclude_lowest_ = get_int_from_token(param[i + 1], __FILE__, __LINE__);
      if (exclude_lowest_ < 0) {
        PRINT_INPUT_ERROR("exclude_lowest for ensemble qct harmonic should be non-negative.");
      }
      has_exclude_lowest = true;
    } else if (strcmp(param[i], "min_frequency") == 0) {
      min_frequency_ = get_double_from_token(param[i + 1], __FILE__, __LINE__);
      if (min_frequency_ < 0.0) {
        PRINT_INPUT_ERROR("min_frequency for ensemble qct harmonic should be non-negative.");
      }
      has_min_frequency = true;
    } else if (strcmp(param[i], "hessian_displacement") == 0) {
      hessian_displacement_ = get_double_from_token(param[i + 1], __FILE__, __LINE__);
      if (hessian_displacement_ <= 0.0) {
        PRINT_INPUT_ERROR("hessian_displacement for ensemble qct harmonic should be positive.");
      }
      has_hessian_displacement = true;
    } else if (strcmp(param[i], "temperature") == 0) {
      sample_temperature_ = get_double_from_token(param[i + 1], __FILE__, __LINE__);
      if (sample_temperature_ < 0.0) {
        PRINT_INPUT_ERROR("temperature for ensemble qct harmonic should be non-negative.");
      }
      has_temperature = true;
    } else if (strcmp(param[i], "energy") == 0) {
      total_energy_eV_ = get_double_from_token(param[i + 1], __FILE__, __LINE__);
      if (total_energy_eV_ < 0.0) {
        PRINT_INPUT_ERROR("energy for ensemble qct should be non-negative.");
      }
      selected_mode_energy_eV_ = total_energy_eV_;
      has_energy = true;
    } else if (strcmp(param[i], "mode") == 0) {
      selected_mode_index_ = get_int_from_token(param[i + 1], __FILE__, __LINE__);
      if (selected_mode_index_ < 0) {
        PRINT_INPUT_ERROR("mode for ensemble qct should be non-negative.");
      }
      has_mode = true;
    } else if (strcmp(param[i], "v") == 0) {
      vibrational_quantum_ = get_int_from_token(param[i + 1], __FILE__, __LINE__);
      if (vibrational_quantum_ < 0) {
        PRINT_INPUT_ERROR("v for ensemble qct semiclassical should be non-negative.");
      }
      has_vibrational_quantum = true;
    } else if (strcmp(param[i], "J") == 0) {
      rotational_quantum_ = get_int_from_token(param[i + 1], __FILE__, __LINE__);
      if (rotational_quantum_ < 0) {
        PRINT_INPUT_ERROR("J for ensemble qct semiclassical should be non-negative.");
      }
      has_rotational_quantum = true;
    } else if (strcmp(param[i], "stationary_point") == 0) {
      if (strcmp(param[i + 1], "auto") == 0) {
        stationary_point_ = Stationary_Point::automatic;
      } else if (strcmp(param[i + 1], "minimum") == 0) {
        stationary_point_ = Stationary_Point::minimum;
      } else if (strcmp(param[i + 1], "saddle") == 0) {
        stationary_point_ = Stationary_Point::saddle;
      } else {
        PRINT_INPUT_ERROR(
          "stationary_point for ensemble qct harmonic should be auto, minimum, or saddle.");
      }
    } else if (strcmp(param[i], "reaction_direction") == 0) {
      if (strcmp(param[i + 1], "positive") == 0) {
        reaction_direction_ = Reaction_Direction::positive;
      } else if (strcmp(param[i + 1], "negative") == 0) {
        reaction_direction_ = Reaction_Direction::negative;
      } else if (strcmp(param[i + 1], "random") == 0) {
        reaction_direction_ = Reaction_Direction::random;
      } else {
        PRINT_INPUT_ERROR(
          "reaction_direction for ensemble qct harmonic should be positive, negative, or random.");
      }
    } else if (strcmp(param[i], "reaction_energy") == 0) {
      reaction_energy_eV_ = get_double_from_token(param[i + 1], __FILE__, __LINE__);
      if (reaction_energy_eV_ < 0.0) {
        PRINT_INPUT_ERROR("reaction_energy for ensemble qct harmonic should be non-negative.");
      }
    } else if (strcmp(param[i], "stationary_force_tolerance") == 0) {
      stationary_force_tolerance_ =
        get_double_from_token(param[i + 1], __FILE__, __LINE__);
      if (stationary_force_tolerance_ <= 0.0) {
        PRINT_INPUT_ERROR(
          "stationary_force_tolerance for ensemble qct harmonic should be positive.");
      }
    } else if (strcmp(param[i], "seed") == 0) {
      seed_ = get_int_from_token(param[i + 1], __FILE__, __LINE__);
      if (seed_ < 0) {
        PRINT_INPUT_ERROR("seed for ensemble qct harmonic should be non-negative.");
      }
    } else if (strcmp(param[i], "replicas") == 0) {
      replicas_ = get_int_from_token(param[i + 1], __FILE__, __LINE__);
      if (replicas_ <= 0) {
        PRINT_INPUT_ERROR("replicas for ensemble qct harmonic should be positive.");
      }
    } else if (strcmp(param[i], "zpe") == 0) {
      zpe_ = parse_yes_no(param[i + 1], "zpe");
    } else if (strcmp(param[i], "anharmonic_reweighting") == 0) {
      anharmonic_reweight_ = parse_yes_no(param[i + 1], "anharmonic_reweighting");
    } else if (strcmp(param[i], "phase") == 0) {
      if (strcmp(param[i + 1], "random") == 0) {
        phase_mode_ = Phase_Mode::random;
      } else if (strcmp(param[i + 1], "zero") == 0) {
        phase_mode_ = Phase_Mode::zero;
      } else {
        PRINT_INPUT_ERROR("phase for ensemble qct harmonic should be random or zero.");
      }
    } else {
      PRINT_INPUT_ERROR("Unknown keyword for ensemble qct harmonic.");
    }
  }

  if (has_modes && has_eigenvector) {
    PRINT_INPUT_ERROR("ensemble qct harmonic modes cannot be combined with eigenvector.");
  }
  if ((has_modes || has_eigenvector) && has_hessian_displacement) {
    PRINT_INPUT_ERROR(
      "hessian_displacement for ensemble qct harmonic is only valid without an external mode source.");
  }
  if (has_modes && (has_exclude_lowest || has_min_frequency)) {
    PRINT_INPUT_ERROR("exclude_lowest and min_frequency are only valid with eigenvector FILE.");
  }
  if (sampling_mode_ == Sampling_Mode::canonical && !has_temperature) {
    PRINT_INPUT_ERROR("ensemble qct canonical requires temperature T.");
  }
  if (sampling_mode_ == Sampling_Mode::microcanonical && !has_energy) {
    PRINT_INPUT_ERROR("ensemble qct microcanonical requires energy E.");
  }
  if (sampling_mode_ == Sampling_Mode::mode_energy && (!has_mode || !has_energy)) {
    PRINT_INPUT_ERROR("ensemble qct mode_energy requires mode INDEX and energy E.");
  }
  if (sampling_mode_ == Sampling_Mode::semiclassical &&
      (!has_vibrational_quantum || !has_rotational_quantum)) {
    PRINT_INPUT_ERROR("ensemble qct semiclassical requires v and J.");
  }
  if (sampling_mode_ == Sampling_Mode::wigner) {
    if (!has_temperature) {
      PRINT_INPUT_ERROR("ensemble qct wigner requires temperature T (use 0 for ground-state Wigner).");
    }
    if (stationary_point_ == Stationary_Point::saddle) {
      PRINT_INPUT_ERROR("ensemble qct wigner does not support saddle points; use minimum or auto.");
    }
  }
  mode_source_ = has_modes ? Mode_Source::qct_modes :
                            (has_eigenvector ? Mode_Source::gpumd_eigenvector :
                                               Mode_Source::automatic_hessian);
}

Ensemble_QCT::QCT_Modes Ensemble_QCT::read_qct_modes(const Atom& atom) const
{
  std::ifstream input(modes_file_);
  if (!input.is_open()) {
    qct_input_error("Cannot open QCT modes file " + modes_file_ + ".");
  }

  QCT_Modes qct_modes;
  bool has_num_atoms = false;
  bool has_num_modes = false;
  bool has_frequency_unit = false;
  bool has_coordinate_unit = false;
  bool has_mass_unit = false;
  bool has_eigenvector_type = false;
  bool has_normalization = false;
  bool has_atoms = false;
  bool has_modes = false;

  std::vector<std::string> tokens;
  int line_number = 0;
  while (read_next_qct_tokens(input, tokens, line_number)) {
    if (tokens[0] == "QCT_MODES") {
      continue;
    } else if (tokens[0] == "num_atoms") {
      require_token_count(tokens, 2, line_number, "num_atoms");
      qct_modes.num_atoms = get_int_from_token(tokens[1], __FILE__, __LINE__);
      if (qct_modes.num_atoms <= 0) {
        qct_input_error("num_atoms in QCT modes file should be positive.");
      }
      has_num_atoms = true;
    } else if (tokens[0] == "num_modes") {
      require_token_count(tokens, 2, line_number, "num_modes");
      qct_modes.num_modes = get_int_from_token(tokens[1], __FILE__, __LINE__);
      if (qct_modes.num_modes <= 0) {
        qct_input_error("num_modes in QCT modes file should be positive.");
      }
      has_num_modes = true;
    } else if (tokens[0] == "frequency_unit") {
      require_token_count(tokens, 2, line_number, "frequency_unit");
      require_exact_token(tokens[1], "THz", "frequency_unit");
      has_frequency_unit = true;
    } else if (tokens[0] == "coordinate_unit") {
      require_token_count(tokens, 2, line_number, "coordinate_unit");
      require_exact_token(tokens[1], "Angstrom", "coordinate_unit");
      has_coordinate_unit = true;
    } else if (tokens[0] == "mass_unit") {
      require_token_count(tokens, 2, line_number, "mass_unit");
      require_exact_token(tokens[1], "amu", "mass_unit");
      has_mass_unit = true;
    } else if (tokens[0] == "eigenvector_type") {
      require_token_count(tokens, 2, line_number, "eigenvector_type");
      require_exact_token(tokens[1], "mass_weighted", "eigenvector_type");
      has_eigenvector_type = true;
    } else if (tokens[0] == "normalization") {
      require_token_count(tokens, 2, line_number, "normalization");
      require_exact_token(tokens[1], "sum_e2_1", "normalization");
      has_normalization = true;
    } else if (tokens[0] == "reference_position") {
      require_token_count(tokens, 2, line_number, "reference_position");
      require_exact_token(tokens[1], "yes", "reference_position");
    } else if (tokens[0] == "atoms") {
      if (!has_num_atoms) {
        qct_input_error("num_atoms should be specified before atoms block.");
      }
      if (has_atoms) {
        qct_input_error("QCT modes file has more than one atoms block.");
      }
      qct_modes.symbol.assign(qct_modes.num_atoms, "");
      qct_modes.mass.assign(qct_modes.num_atoms, 0.0);
      qct_modes.reference_position.assign(qct_modes.num_atoms * 3, 0.0);
      std::vector<bool> atom_seen(qct_modes.num_atoms, false);
      int num_atoms_read = 0;
      bool found_end_atoms = false;
      while (read_next_qct_tokens(input, tokens, line_number)) {
        if (tokens[0] == "end_atoms") {
          found_end_atoms = true;
          break;
        }
        require_token_count(tokens, 6, line_number, "atom entry");
        const int atom_index = get_int_from_token(tokens[0], __FILE__, __LINE__);
        if (atom_index < 0 || atom_index >= qct_modes.num_atoms) {
          qct_input_error("Atom index in QCT modes file is out of range.");
        }
        if (atom_seen[atom_index]) {
          qct_input_error("Duplicate atom index in QCT modes file.");
        }
        atom_seen[atom_index] = true;
        qct_modes.symbol[atom_index] = tokens[1];
        qct_modes.mass[atom_index] = get_double_from_token(tokens[2], __FILE__, __LINE__);
        if (qct_modes.mass[atom_index] <= 0.0) {
          qct_input_error("Atom mass in QCT modes file should be positive.");
        }
        for (int d = 0; d < 3; ++d) {
          qct_modes.reference_position[atom_index + qct_modes.num_atoms * d] =
            get_double_from_token(tokens[3 + d], __FILE__, __LINE__);
        }
        ++num_atoms_read;
      }
      if (!found_end_atoms) {
        qct_input_error("QCT modes atoms block is missing end_atoms.");
      }
      if (num_atoms_read != qct_modes.num_atoms) {
        qct_input_error("QCT modes atoms block does not contain num_atoms entries.");
      }
      has_atoms = true;
    } else if (tokens[0] == "modes") {
      if (!has_num_atoms || !has_num_modes) {
        qct_input_error("num_atoms and num_modes should be specified before modes block.");
      }
      if (has_modes) {
        qct_input_error("QCT modes file has more than one modes block.");
      }
      std::vector<bool> mode_seen(qct_modes.num_modes, false);
      bool found_end_modes = false;
      while (read_next_qct_tokens(input, tokens, line_number)) {
        if (tokens[0] == "end_modes") {
          found_end_modes = true;
          break;
        }
        require_token_count(tokens, 6, line_number, "mode header");
        if (tokens[0] != "mode" || tokens[2] != "frequency" || tokens[4] != "active") {
          qct_input_error(
            "Mode header should be: mode INDEX frequency FREQ active yes|no.");
        }
        Normal_Mode mode;
        mode.index = get_int_from_token(tokens[1], __FILE__, __LINE__);
        if (mode.index < 0 || mode.index >= qct_modes.num_modes) {
          qct_input_error("Mode index in QCT modes file is out of range.");
        }
        if (mode_seen[mode.index]) {
          qct_input_error("Duplicate mode index in QCT modes file.");
        }
        mode_seen[mode.index] = true;
        mode.frequency_THz = get_double_from_token(tokens[3], __FILE__, __LINE__);
        mode.active = parse_yes_no(tokens[5], "mode active");
        mode.eigenvector.assign(qct_modes.num_atoms * 3, 0.0);

        std::vector<bool> atom_seen(qct_modes.num_atoms, false);
        int num_eigenvectors_read = 0;
        bool found_end_mode = false;
        while (read_next_qct_tokens(input, tokens, line_number)) {
          if (tokens[0] == "end_mode") {
            found_end_mode = true;
            break;
          }
          require_token_count(tokens, 4, line_number, "mode eigenvector entry");
          const int atom_index = get_int_from_token(tokens[0], __FILE__, __LINE__);
          if (atom_index < 0 || atom_index >= qct_modes.num_atoms) {
            qct_input_error("Atom index in mode eigenvector entry is out of range.");
          }
          if (atom_seen[atom_index]) {
            qct_input_error("Duplicate atom index in mode eigenvector entries.");
          }
          atom_seen[atom_index] = true;
          for (int d = 0; d < 3; ++d) {
            mode.eigenvector[atom_index + qct_modes.num_atoms * d] =
              get_double_from_token(tokens[1 + d], __FILE__, __LINE__);
          }
          ++num_eigenvectors_read;
        }
        if (!found_end_mode) {
          qct_input_error("QCT mode block is missing end_mode.");
        }
        if (num_eigenvectors_read != qct_modes.num_atoms) {
          qct_input_error("QCT mode block does not contain num_atoms eigenvector entries.");
        }
        qct_modes.modes.emplace_back(std::move(mode));
      }
      if (!found_end_modes) {
        qct_input_error("QCT modes block is missing end_modes.");
      }
      has_modes = true;
    } else {
      qct_input_error("Unknown keyword in QCT modes file: " + tokens[0] + ".");
    }
  }

  if (
    !has_num_atoms || !has_num_modes || !has_frequency_unit || !has_coordinate_unit ||
    !has_mass_unit || !has_eigenvector_type || !has_normalization || !has_atoms || !has_modes) {
    qct_input_error("QCT modes file is missing required header or data blocks.");
  }
  // model.xyz is the sole source of the Cartesian reference structure.
  qct_modes.reference_position = atom.cpu_position_per_atom;
  validate_qct_modes(qct_modes, atom);

  return qct_modes;
}

Ensemble_QCT::QCT_Modes Ensemble_QCT::read_gpumd_modes(const Atom& atom) const
{
  QCT_Modes qct_modes;
  qct_modes.num_atoms = atom.number_of_atoms;
  qct_modes.num_modes = atom.number_of_atoms * 3;
  qct_modes.number_of_rigid_modes = exclude_lowest_;
  qct_modes.symbol = atom.cpu_atom_symbol;
  qct_modes.mass = atom.cpu_mass;
  qct_modes.reference_position = atom.cpu_position_per_atom;
  if (exclude_lowest_ >= qct_modes.num_modes) {
    qct_input_error("exclude_lowest should be smaller than 3 * num_atoms.");
  }
  const size_t num_values =
    static_cast<size_t>(qct_modes.num_modes) * (qct_modes.num_modes + 1);
  const size_t expected_bytes = num_values * sizeof(float);
  std::ifstream eigenvector(eigenvector_file_, std::ios::binary | std::ios::ate);
  if (!eigenvector.is_open()) {
    qct_input_error("Cannot open GPUMD eigenvector file " + eigenvector_file_ + ".");
  }
  const std::streamoff file_size = eigenvector.tellg();
  if (file_size < 0 || static_cast<uint64_t>(file_size) != expected_bytes) {
    qct_input_error(
      "GPUMD eigenvector file size does not match the current model.xyz atom count.");
  }
  eigenvector.seekg(0);
  std::vector<float> values(num_values);
  eigenvector.read(
    reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expected_bytes));
  if (!eigenvector) {
    qct_input_error("Failed to read GPUMD eigenvector file " + eigenvector_file_ + ".");
  }

  qct_modes.modes.reserve(qct_modes.num_modes);
  for (int mode_index = 0; mode_index < qct_modes.num_modes; ++mode_index) {
    Normal_Mode mode;
    mode.index = mode_index;
    const double omega2 = values[mode_index];
    if (!std::isfinite(omega2)) {
      qct_input_error(
        "Mode " + std::to_string(mode_index) +
        " has a non-finite frequency in the GPUMD eigenvector file.");
    }
    // omega2 from eigenvector.out is in (rad/ps)^2; convert to regular THz
    mode.frequency_THz =
      omega2 >= 0.0 ? std::sqrt(omega2) * RAD_PER_PS_TO_THZ
                    : -std::sqrt(-omega2) * RAD_PER_PS_TO_THZ;
    const size_t vector_begin =
      static_cast<size_t>(qct_modes.num_modes) * (mode_index + 1);
    mode.eigenvector.assign(
      values.begin() + vector_begin, values.begin() + vector_begin + qct_modes.num_modes);
    qct_modes.modes.emplace_back(std::move(mode));
  }
  std::vector<int> mode_indices(qct_modes.num_modes);
  std::iota(mode_indices.begin(), mode_indices.end(), 0);
  std::stable_sort(mode_indices.begin(), mode_indices.end(), [&](const int first, const int second) {
    return std::fabs(qct_modes.modes[first].frequency_THz) <
           std::fabs(qct_modes.modes[second].frequency_THz);
  });
  for (int n = 0; n < exclude_lowest_; ++n) {
    qct_modes.modes[mode_indices[n]].rigid = true;
  }
  for (auto& mode : qct_modes.modes) {
    mode.active = !mode.rigid && mode.frequency_THz > min_frequency_;
  }
  validate_qct_modes(qct_modes, atom);
  return qct_modes;
}

Ensemble_QCT::QCT_Modes Ensemble_QCT::build_automatic_modes(
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  Force& force) const
{
  Molecular_Hessian_Result hessian =
    Molecular_Hessian::compute(hessian_displacement_, force, box, atom, group);
  if (hessian.max_force > stationary_force_tolerance_) {
    qct_input_error(
      "QCT automatic Hessian requires a stationary model.xyz structure: maximum force is " +
      std::to_string(hessian.max_force) + " eV/A.");
  }
  Molecular_Hessian::write_qct_audit(hessian, atom);

  QCT_Modes qct_modes;
  qct_modes.num_atoms = atom.number_of_atoms;
  qct_modes.num_modes = atom.number_of_atoms * 3;
  qct_modes.number_of_rigid_modes = hessian.number_of_rigid_modes;
  qct_modes.symbol = atom.cpu_atom_symbol;
  qct_modes.mass = atom.cpu_mass;
  qct_modes.reference_position = atom.cpu_position_per_atom;
  qct_modes.modes.reserve(qct_modes.num_modes);

  for (int mode_index = 0; mode_index < qct_modes.num_modes; ++mode_index) {
    Normal_Mode mode;
    mode.index = mode_index;
    mode.rigid = mode_index < hessian.number_of_rigid_modes;
    const double omega2 = hessian.omega2_THz2[mode_index];
    // omega2 from molecular_hessian is in (rad/ps)^2; convert to regular THz
    mode.frequency_THz =
      omega2 >= 0.0 ? std::sqrt(omega2) * RAD_PER_PS_TO_THZ
                    : -std::sqrt(-omega2) * RAD_PER_PS_TO_THZ;
    mode.active = !mode.rigid && mode.frequency_THz > min_frequency_;
    const int dimension = qct_modes.num_modes;
    mode.eigenvector.assign(
      hessian.eigenvectors.begin() + static_cast<size_t>(dimension) * mode_index,
      hessian.eigenvectors.begin() + static_cast<size_t>(dimension) * (mode_index + 1));
    orient_eigenvector(mode.eigenvector);
    qct_modes.modes.emplace_back(std::move(mode));
  }
  validate_qct_modes(qct_modes, atom);
  return qct_modes;
}

void Ensemble_QCT::classify_stationary_point(QCT_Modes& qct_modes) const
{
  int number_of_imaginary_modes = 0;
  int reaction_mode_index = -1;
  for (const auto& mode : qct_modes.modes) {
    if (mode.rigid) {
      continue;
    }
    if (mode.frequency_THz < -min_frequency_) {
      ++number_of_imaginary_modes;
      reaction_mode_index = mode.index;
    }
  }

  if (stationary_point_ == Stationary_Point::minimum && number_of_imaginary_modes != 0) {
    qct_input_error(
      "stationary_point minimum requires no significant imaginary vibrational modes; found " +
      std::to_string(number_of_imaginary_modes) + ".");
  }
  if (stationary_point_ == Stationary_Point::saddle && number_of_imaginary_modes != 1) {
    qct_input_error(
      "stationary_point saddle requires exactly one significant imaginary vibrational mode; found " +
      std::to_string(number_of_imaginary_modes) + ".");
  }
  if (stationary_point_ == Stationary_Point::automatic && number_of_imaginary_modes > 1) {
    qct_input_error(
      "Automatic QCT classification found more than one significant imaginary vibrational mode (" +
      std::to_string(number_of_imaginary_modes) + ").");
  }

  qct_modes.reaction_mode_index = number_of_imaginary_modes == 1 ? reaction_mode_index : -1;
  int number_of_active_modes = 0;
  for (auto& mode : qct_modes.modes) {
    orient_eigenvector(mode.eigenvector);
    if (mode.index == qct_modes.reaction_mode_index) {
      mode.active = false;
    } else if (mode_source_ == Mode_Source::automatic_hessian) {
      mode.active = !mode.rigid && mode.frequency_THz > min_frequency_;
    } else if (mode.frequency_THz <= min_frequency_) {
      mode.active = false;
    }
    number_of_active_modes += mode.active ? 1 : 0;
  }

  if (qct_modes.reaction_mode_index < 0 && number_of_active_modes == 0) {
    qct_input_error("QCT found no active stable vibrational modes.");
  }
  if (qct_modes.reaction_mode_index >= 0) {
    const double orthogonality_tolerance = 1.0e-6;
    for (const auto& mode : qct_modes.modes) {
      if (!mode.active) {
        continue;
      }
      double overlap = 0.0;
      const auto& stable = mode.eigenvector;
      const auto& reaction = qct_modes.modes[qct_modes.reaction_mode_index].eigenvector;
      for (size_t n = 0; n < stable.size(); ++n) {
        overlap += stable[n] * reaction[n];
      }
      if (std::fabs(overlap) > orthogonality_tolerance) {
        qct_input_error(
          "Active stable QCT mode " + std::to_string(mode.index) +
          " is not orthogonal to the reaction mode (overlap " +
          std::to_string(overlap) + ").");
      }
    }
    printf(
      "    classified current structure as a first-order saddle; reaction mode is %d at %g THz.\n",
      qct_modes.reaction_mode_index,
      qct_modes.modes[qct_modes.reaction_mode_index].frequency_THz);
  } else {
    printf("    classified current structure as a minimum.\n");
  }
}

void Ensemble_QCT::validate_qct_modes(const QCT_Modes& qct_modes, const Atom& atom) const
{
  if (qct_modes.num_atoms != atom.number_of_atoms) {
    qct_input_error("Number of atoms in the QCT mode input does not match model.xyz.");
  }
  if (qct_modes.num_modes > qct_modes.num_atoms * 3) {
    qct_input_error("Number of QCT modes cannot exceed 3 * num_atoms.");
  }
  if (qct_modes.modes.size() != static_cast<size_t>(qct_modes.num_modes)) {
    qct_input_error("QCT mode input does not contain num_modes mode entries.");
  }
  for (int mode_index = 0; mode_index < qct_modes.num_modes; ++mode_index) {
    if (qct_modes.modes[mode_index].index != mode_index) {
      qct_input_error("QCT mode entries should be ordered by mode index from 0 to num_modes - 1.");
    }
  }

  for (int n = 0; n < qct_modes.num_atoms; ++n) {
    if (
      atom.cpu_atom_symbol.size() == static_cast<size_t>(qct_modes.num_atoms) &&
      qct_modes.symbol[n] != atom.cpu_atom_symbol[n]) {
      qct_input_error("Atom symbols in the QCT mode input do not match model.xyz.");
    }
    const double mass_scale = std::max(1.0, std::fabs(atom.cpu_mass[n]));
    if (std::fabs(qct_modes.mass[n] - atom.cpu_mass[n]) > 1.0e-6 * mass_scale) {
      qct_input_error("Atom masses in the QCT mode input do not match model.xyz.");
    }
  }

  for (const auto& mode : qct_modes.modes) {
    double norm = 0.0;
    for (const auto& component : mode.eigenvector) {
      if (!std::isfinite(component)) {
        qct_input_error("QCT mode eigenvector contains a non-finite component.");
      }
      norm += component * component;
    }
    if (mode.active) {
      if (mode.frequency_THz <= 0.0) {
        qct_input_error("Active QCT mode should have a positive frequency.");
      }
      if (std::fabs(norm - 1.0) > 1.0e-3) {
        qct_input_error("Active QCT mode eigenvector should satisfy normalization sum_e2_1.");
      }
    } else if (norm > 1.0e-12 && std::fabs(norm - 1.0) > 1.0e-3) {
      qct_input_error("QCT mode eigenvector should satisfy normalization sum_e2_1.");
    }
  }

  const double orthogonality_tolerance = 1.0e-6;
  for (int first = 0; first < qct_modes.num_modes; ++first) {
    if (!qct_modes.modes[first].active) {
      continue;
    }
    for (int second = first + 1; second < qct_modes.num_modes; ++second) {
      if (!qct_modes.modes[second].active) {
        continue;
      }
      double overlap = 0.0;
      const auto& a = qct_modes.modes[first].eigenvector;
      const auto& b = qct_modes.modes[second].eigenvector;
      for (size_t n = 0; n < a.size(); ++n) {
        overlap += a[n] * b[n];
      }
      if (std::fabs(overlap) > orthogonality_tolerance) {
        qct_input_error(
          "Active QCT modes " + std::to_string(first) + " and " +
          std::to_string(second) + " are not mass-weighted orthogonal (overlap " +
          std::to_string(overlap) + ").");
      }
    }
  }
}

void Ensemble_QCT::initialize_before_run(
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Force& force)
{
  if (initialized_) {
    return;
  }

  atoms_per_replica_ = atom.number_of_atoms;
  if (replicas_ == 1) {
    replica_seeds_.assign(1, static_cast<std::uint64_t>(seed_));
  }

  if (init_mode_ == Init_Mode::harmonic) {
    initialize_harmonic_replicas(atom, box, group, thermo, force);
  } else if (!atom.has_velocity_in_xyz) {
    qct_input_error(
      "QCT phase_point requires explicit vel properties in model.xyz; default random velocities are not a QCT phase point.");
  }

  initialized_ = true;
}

Ensemble_QCT::Sampled_Point Ensemble_QCT::sample_harmonic_point(
  const QCT_Modes& qct_modes, const std::uint64_t seed) const
{
  Sampled_Point point;
  point.seed = seed;
  point.position = qct_modes.reference_position;
  point.velocity.assign(static_cast<size_t>(qct_modes.num_atoms) * 3, 0.0);
  point.rotational_velocity.assign(static_cast<size_t>(qct_modes.num_atoms) * 3, 0.0);
  point.modes.resize(qct_modes.num_modes);

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uniform_01(0.0, 1.0);
  std::vector<double> prescribed_energy(qct_modes.num_modes, -1.0);
  std::vector<int> active_mode_indices;
  double stable_zero_point_energy = 0.0;
  for (const auto& mode : qct_modes.modes) {
    if (!mode.active) {
      continue;
    }
    if (mode.frequency_THz <= min_frequency_) {
      qct_input_error("Active QCT mode has a non-positive frequency.");
    }
    const double omega = mode.frequency_THz * THZ_TO_NATURAL_ANGULAR_FREQUENCY;
    stable_zero_point_energy += zpe_ ? 0.5 * HBAR * omega : 0.0;
    active_mode_indices.emplace_back(mode.index);
  }

  if (sampling_mode_ == Sampling_Mode::microcanonical) {
    if (total_energy_eV_ < 0.0) {
      qct_input_error("ensemble qct microcanonical requires energy E.");
    }
    if (qct_modes.reaction_mode_index >= 0 && reaction_energy_eV_ < 0.0) {
      qct_input_error("Microcanonical saddle QCT requires an explicit positive reaction_energy.");
    }
    const double requested_reaction_energy =
      qct_modes.reaction_mode_index >= 0 ? reaction_energy_eV_ : 0.0;
    const double stable_energy = total_energy_eV_ - requested_reaction_energy;
    if (stable_energy + 1.0e-12 < stable_zero_point_energy) {
      qct_input_error(
        "Microcanonical QCT energy is smaller than the requested stable-mode zero-point energy.");
    }
    const double excess_energy = std::max(0.0, stable_energy - stable_zero_point_energy);
    if (active_mode_indices.empty() && excess_energy > 1.0e-12) {
      qct_input_error("Microcanonical QCT has excess energy but no active stable modes.");
    }
    std::vector<double> weights(active_mode_indices.size(), 0.0);
    double weight_sum = 0.0;
    for (double& weight : weights) {
      weight = -std::log(std::max(uniform_01(rng), std::numeric_limits<double>::min()));
      weight_sum += weight;
    }
    for (size_t i = 0; i < active_mode_indices.size(); ++i) {
      const int mode_index = active_mode_indices[i];
      const double omega = qct_modes.modes[mode_index].frequency_THz *
                           THZ_TO_NATURAL_ANGULAR_FREQUENCY;
      const double zero_point_energy = zpe_ ? 0.5 * HBAR * omega : 0.0;
      prescribed_energy[mode_index] =
        zero_point_energy + (weight_sum > 0.0 ? excess_energy * weights[i] / weight_sum : 0.0);
    }
  } else if (sampling_mode_ == Sampling_Mode::mode_energy) {
    if (selected_mode_index_ < 0 || selected_mode_index_ >= qct_modes.num_modes ||
        !qct_modes.modes[selected_mode_index_].active) {
      qct_input_error("mode_energy should select an active stable QCT mode.");
    }
    if (selected_mode_energy_eV_ < 0.0) {
      qct_input_error("mode_energy requires a non-negative mode energy.");
    }
    const double selected_omega = qct_modes.modes[selected_mode_index_].frequency_THz *
                                  THZ_TO_NATURAL_ANGULAR_FREQUENCY;
    const double selected_zero_point = zpe_ ? 0.5 * HBAR * selected_omega : 0.0;
    if (selected_mode_energy_eV_ + 1.0e-12 < selected_zero_point) {
      qct_input_error("mode_energy cannot be below the selected mode zero-point energy.");
    }
    for (const int mode_index : active_mode_indices) {
      const double omega = qct_modes.modes[mode_index].frequency_THz *
                           THZ_TO_NATURAL_ANGULAR_FREQUENCY;
      prescribed_energy[mode_index] = zpe_ ? 0.5 * HBAR * omega : 0.0;
    }
    prescribed_energy[selected_mode_index_] = selected_mode_energy_eV_;
    if (qct_modes.reaction_mode_index >= 0 && reaction_energy_eV_ < 0.0) {
      qct_input_error("Saddle mode_energy QCT requires an explicit positive reaction_energy.");
    }
  } else if (sampling_mode_ == Sampling_Mode::semiclassical) {
    if (qct_modes.reaction_mode_index >= 0) {
      qct_input_error("semiclassical QCT is only supported for stationary minima.");
    }
    if (qct_modes.num_atoms != 2 || active_mode_indices.size() != 1) {
      qct_input_error("semiclassical QCT currently requires a diatomic molecule with one vibration.");
    }
    const int vibration_mode = active_mode_indices.front();
    const double omega = qct_modes.modes[vibration_mode].frequency_THz *
                         THZ_TO_NATURAL_ANGULAR_FREQUENCY;
    prescribed_energy[vibration_mode] =
      (static_cast<double>(vibrational_quantum_) + 0.5) * HBAR * omega;
  }

  // Wigner (LSC-IVR) sampling: each active mode (Q_k, P_k) is drawn from
  // independent Gaussians with quantum-corrected variance.  Unlike the
  // classical modes above, there is no prescribed energy and no phase
  // ring; the Wigner distribution is a product of Gaussians.
  if (sampling_mode_ == Sampling_Mode::wigner) {
    std::normal_distribution<double> standard_normal(0.0, 1.0);
    for (const int mode_index : active_mode_indices) {
      const auto& mode = qct_modes.modes[mode_index];
      const double omega = mode.frequency_THz * THZ_TO_NATURAL_ANGULAR_FREQUENCY;
      const double zpe_k = 0.5 * HBAR * omega;
      // coth(beta * hbar * omega / 2); at T=0, coth -> 1 (ground-state Wigner)
      double coth_factor = 1.0;
      if (sample_temperature_ > 0.0) {
        const double x = zpe_k / (K_B * sample_temperature_);
        coth_factor = (x > 350.0) ? 1.0 : (std::exp(2.0 * x) + 1.0) / (std::exp(2.0 * x) - 1.0);
      }
      // Variance of Q_k and P_k in the harmonic Wigner distribution.
      // sigma_Q^2 = (hbar / (2 omega)) * coth   [= zpe_k / omega^2 * coth]
      // sigma_P^2 = (hbar * omega / 2) * coth   [= zpe_k * coth]
      const double sigma_Q = std::sqrt(zpe_k / (omega * omega) * coth_factor);
      const double sigma_P = std::sqrt(zpe_k * coth_factor);
      Sampled_Mode sampled_mode;
      sampled_mode.Q = sigma_Q * standard_normal(rng);
      sampled_mode.P = sigma_P * standard_normal(rng);
      sampled_mode.energy = 0.5 * (sampled_mode.P * sampled_mode.P +
                                   omega * omega * sampled_mode.Q * sampled_mode.Q);
      sampled_mode.phase = std::numeric_limits<double>::quiet_NaN(); // Wigner has no phase
      for (int n = 0; n < qct_modes.num_atoms; ++n) {
        const double mass_sqrt_inv = 1.0 / std::sqrt(qct_modes.mass[n]);
        for (int d = 0; d < 3; ++d) {
          const int index = n + qct_modes.num_atoms * d;
          point.position[index] += mode.eigenvector[index] * sampled_mode.Q * mass_sqrt_inv;
          point.velocity[index] += mode.eigenvector[index] * sampled_mode.P * mass_sqrt_inv;
        }
      }
      point.total_sampled_energy += sampled_mode.energy;
      point.modes[mode_index] = sampled_mode;
    }
  } else {

  for (const auto& mode : qct_modes.modes) {
    Sampled_Mode sampled_mode;
    if (mode.index == qct_modes.reaction_mode_index || !mode.active) {
      point.modes.emplace_back(sampled_mode);
      continue;
    }

    const double omega = mode.frequency_THz * THZ_TO_NATURAL_ANGULAR_FREQUENCY;
    if (sampling_mode_ == Sampling_Mode::canonical) {
      sampled_mode.energy = zpe_ ? 0.5 * HBAR * omega : 0.0;
      if (sample_temperature_ > 0.0) {
        const double u = std::max(uniform_01(rng), std::numeric_limits<double>::min());
        sampled_mode.energy += -K_B * sample_temperature_ * std::log(u);
      }
    } else {
      sampled_mode.energy = prescribed_energy[mode.index];
    }

    if (sampled_mode.energy > 0.0) {
      if (phase_mode_ == Phase_Mode::random) {
        sampled_mode.phase = 2.0 * PI * uniform_01(rng);
      } else {
        sampled_mode.phase = 0.25 * PI;
      }
      sampled_mode.Q =
        std::sqrt(2.0 * sampled_mode.energy) * std::cos(sampled_mode.phase) / omega;
      sampled_mode.P = -std::sqrt(2.0 * sampled_mode.energy) * std::sin(sampled_mode.phase);
      for (int n = 0; n < qct_modes.num_atoms; ++n) {
        const double mass_sqrt_inv = 1.0 / std::sqrt(qct_modes.mass[n]);
        for (int d = 0; d < 3; ++d) {
          const int index = n + qct_modes.num_atoms * d;
          point.position[index] += mode.eigenvector[index] * sampled_mode.Q * mass_sqrt_inv;
          point.velocity[index] += mode.eigenvector[index] * sampled_mode.P * mass_sqrt_inv;
        }
      }
      point.total_sampled_energy += sampled_mode.energy;
    }
    point.modes.emplace_back(sampled_mode);
  }
  } // end non-wigner branch

  // Wigner (LSC-IVR) sampling does not support reactive trajectories along
  // imaginary modes; the reaction mode is already marked inactive and simply
  // skipped.  All other sampling modes inject a reaction momentum here.
  if (qct_modes.reaction_mode_index >= 0 &&
      sampling_mode_ != Sampling_Mode::wigner) {
    if (sampling_mode_ != Sampling_Mode::canonical && reaction_energy_eV_ < 0.0) {
      qct_input_error("Non-canonical saddle QCT requires an explicit reaction_energy.");
    }
    if (sampling_mode_ == Sampling_Mode::canonical && reaction_energy_eV_ < 0.0 &&
        sample_temperature_ <= 0.0) {
      qct_input_error(
        "First-order-saddle canonical QCT requires positive temperature or reaction_energy.");
    }
    const double reaction_energy =
      reaction_energy_eV_ >= 0.0
        ? reaction_energy_eV_
        : -K_B * sample_temperature_ *
            std::log(std::max(uniform_01(rng), std::numeric_limits<double>::min()));
    if (reaction_energy <= 0.0) {
      qct_input_error("QCT reaction_energy should be positive for a saddle launch.");
    }
    double direction = 1.0;
    if (reaction_direction_ == Reaction_Direction::negative) {
      direction = -1.0;
    } else if (reaction_direction_ == Reaction_Direction::random && uniform_01(rng) >= 0.5) {
      direction = -1.0;
    }
    const double reaction_momentum = direction * std::sqrt(2.0 * reaction_energy);
    point.reaction_energy = reaction_energy;
    point.modes[qct_modes.reaction_mode_index].energy = reaction_energy;
    point.modes[qct_modes.reaction_mode_index].P = reaction_momentum;
    const auto& reaction_mode = qct_modes.modes[qct_modes.reaction_mode_index];
    for (int n = 0; n < qct_modes.num_atoms; ++n) {
      const double mass_sqrt_inv = 1.0 / std::sqrt(qct_modes.mass[n]);
      for (int d = 0; d < 3; ++d) {
        const int index = n + qct_modes.num_atoms * d;
        point.velocity[index] +=
          reaction_mode.eigenvector[index] * reaction_momentum * mass_sqrt_inv;
      }
    }
    point.total_sampled_energy += reaction_energy;
  }

  if (sampling_mode_ == Sampling_Mode::semiclassical) {
    const double x = point.position[1] - point.position[0];
    const double y = point.position[qct_modes.num_atoms + 1] -
                     point.position[qct_modes.num_atoms];
    const double z = point.position[2 * qct_modes.num_atoms + 1] -
                     point.position[2 * qct_modes.num_atoms];
    const double bond_length = std::sqrt(x * x + y * y + z * z);
    if (bond_length <= 1.0e-12) {
      qct_input_error("semiclassical QCT requires a non-zero diatomic bond length.");
    }
    double basis_x = 0.0;
    double basis_y = 0.0;
    double basis_z = 0.0;
    const double ax = std::fabs(x / bond_length);
    const double ay = std::fabs(y / bond_length);
    const double az = std::fabs(z / bond_length);
    if (ax <= ay && ax <= az) {
      basis_x = 1.0;
    } else if (ay <= az) {
      basis_y = 1.0;
    } else {
      basis_z = 1.0;
    }
    double perpendicular_x = y * basis_z - z * basis_y;
    double perpendicular_y = z * basis_x - x * basis_z;
    double perpendicular_z = x * basis_y - y * basis_x;
    const double perpendicular_length = std::sqrt(
      perpendicular_x * perpendicular_x + perpendicular_y * perpendicular_y +
      perpendicular_z * perpendicular_z);
    perpendicular_x /= perpendicular_length;
    perpendicular_y /= perpendicular_length;
    perpendicular_z /= perpendicular_length;
    const double total_mass = qct_modes.mass[0] + qct_modes.mass[1];
    const double center_x =
      (qct_modes.mass[0] * point.position[0] + qct_modes.mass[1] * point.position[1]) /
      total_mass;
    const double center_y =
      (qct_modes.mass[0] * point.position[qct_modes.num_atoms] +
       qct_modes.mass[1] * point.position[qct_modes.num_atoms + 1]) /
      total_mass;
    const double center_z =
      (qct_modes.mass[0] * point.position[2 * qct_modes.num_atoms] +
       qct_modes.mass[1] * point.position[2 * qct_modes.num_atoms + 1]) /
      total_mass;
    const double reduced_mass = qct_modes.mass[0] * qct_modes.mass[1] / total_mass;
    const double moment_perpendicular = reduced_mass * bond_length * bond_length;
    const double angular_momentum =
      std::sqrt(static_cast<double>(rotational_quantum_) * (rotational_quantum_ + 1.0)) * HBAR;
    const double angular_velocity = angular_momentum / moment_perpendicular;
    const double omega_x = angular_velocity * perpendicular_x;
    const double omega_y = angular_velocity * perpendicular_y;
    const double omega_z = angular_velocity * perpendicular_z;
    for (int n = 0; n < 2; ++n) {
      const double rx = point.position[n] - center_x;
      const double ry = point.position[qct_modes.num_atoms + n] - center_y;
      const double rz = point.position[2 * qct_modes.num_atoms + n] - center_z;
      point.rotational_velocity[n] = omega_y * rz - omega_z * ry;
      point.rotational_velocity[qct_modes.num_atoms + n] = omega_z * rx - omega_x * rz;
      point.rotational_velocity[2 * qct_modes.num_atoms + n] = omega_x * ry - omega_y * rx;
      point.velocity[n] += point.rotational_velocity[n];
      point.velocity[qct_modes.num_atoms + n] +=
        point.rotational_velocity[qct_modes.num_atoms + n];
      point.velocity[2 * qct_modes.num_atoms + n] +=
        point.rotational_velocity[2 * qct_modes.num_atoms + n];
    }
    point.rotational_energy = 0.5 * angular_momentum * angular_momentum / moment_perpendicular;
    point.total_sampled_energy += point.rotational_energy;
  }
  return point;
}

bool Ensemble_QCT::apply_potential_correction(
  const QCT_Modes& qct_modes,
  Sampled_Point& point,
  const double reference_potential,
  const double sampled_potential) const
{
  if (sampling_mode_ == Sampling_Mode::semiclassical && point.total_sampled_energy <= 0.0) {
    return true;
  }

  point.potential_correction = sampled_potential - reference_potential;
  std::vector<double> reaction_velocity(point.velocity.size(), 0.0);
  if (qct_modes.reaction_mode_index >= 0) {
    const auto& reaction_mode = qct_modes.modes[qct_modes.reaction_mode_index];
    const double reaction_momentum = point.modes[qct_modes.reaction_mode_index].P;
    for (int n = 0; n < qct_modes.num_atoms; ++n) {
      const double mass_sqrt_inv = 1.0 / std::sqrt(qct_modes.mass[n]);
      for (int d = 0; d < 3; ++d) {
        const int index = n + qct_modes.num_atoms * d;
        reaction_velocity[index] =
          reaction_mode.eigenvector[index] * reaction_momentum * mass_sqrt_inv;
      }
    }
  }

  std::vector<double> vibrational_velocity = point.velocity;
  double current_vibrational_kinetic = 0.0;
  for (int index = 0; index < qct_modes.num_atoms * 3; ++index) {
    vibrational_velocity[index] -= reaction_velocity[index] + point.rotational_velocity[index];
    const int atom_index = index % qct_modes.num_atoms;
    current_vibrational_kinetic += 0.5 * qct_modes.mass[atom_index] *
                                   vibrational_velocity[index] * vibrational_velocity[index];
  }
  const double target_vibrational_kinetic = point.total_sampled_energy - point.reaction_energy -
                                            point.rotational_energy - point.potential_correction;
  if (target_vibrational_kinetic < -1.0e-10) {
    return false;
  }
  if (current_vibrational_kinetic <= 1.0e-20) {
    if (target_vibrational_kinetic > 1.0e-10) {
      return false;
    }
    point.stable_velocity_scale = 0.0;
    for (size_t index = 0; index < point.velocity.size(); ++index) {
      point.velocity[index] = reaction_velocity[index] + point.rotational_velocity[index];
    }
    return true;
  }

  point.stable_velocity_scale =
    std::sqrt(std::max(0.0, target_vibrational_kinetic) / current_vibrational_kinetic);
  for (double& component : vibrational_velocity) {
    component *= point.stable_velocity_scale;
  }
  for (size_t index = 0; index < point.velocity.size(); ++index) {
    point.velocity[index] =
      vibrational_velocity[index] + reaction_velocity[index] + point.rotational_velocity[index];
  }
  return true;
}

void Ensemble_QCT::expand_atom_for_batch(
  Atom& atom, std::vector<Group>& group, GPU_Vector<double>& thermo) const
{
  const int original_atoms = atom.number_of_atoms;
  const long long total_atoms_ll = static_cast<long long>(original_atoms) * replicas_;
  if (original_atoms <= 0 || total_atoms_ll > std::numeric_limits<int>::max()) {
    qct_input_error("QCT batch atom count is outside the supported integer range.");
  }
  const int total_atoms = static_cast<int>(total_atoms_ll);
  const auto old_type = atom.cpu_type;
  const auto old_mass = atom.cpu_mass;
  const auto old_charge = atom.cpu_charge;
  const auto old_symbol = atom.cpu_atom_symbol;
  const auto old_position = atom.cpu_position_per_atom;
  const auto old_velocity = atom.cpu_velocity_per_atom;
  const auto old_type_size = atom.cpu_type_size;
  std::vector<std::vector<int>> old_labels(group.size());
  for (size_t m = 0; m < group.size(); ++m) {
    old_labels[m] = group[m].cpu_label;
  }

  atom.number_of_atoms = total_atoms;
  atom.cpu_type.resize(total_atoms);
  atom.cpu_mass.resize(total_atoms);
  atom.cpu_charge.resize(total_atoms);
  atom.cpu_atom_symbol.resize(total_atoms);
  atom.cpu_position_per_atom.resize(static_cast<size_t>(total_atoms) * 3);
  atom.cpu_velocity_per_atom.resize(static_cast<size_t>(total_atoms) * 3);
  for (int replica = 0; replica < replicas_; ++replica) {
    const auto& point = sampled_points_[replica];
    for (int n = 0; n < original_atoms; ++n) {
      const int destination = replica * original_atoms + n;
      atom.cpu_type[destination] = old_type[n];
      atom.cpu_mass[destination] = old_mass[n];
      atom.cpu_charge[destination] = old_charge[n];
      atom.cpu_atom_symbol[destination] = old_symbol[n];
      for (int d = 0; d < 3; ++d) {
        atom.cpu_position_per_atom[destination + total_atoms * d] =
          point.position[n + original_atoms * d];
        atom.cpu_velocity_per_atom[destination + total_atoms * d] =
          point.velocity[n + original_atoms * d];
      }
    }
  }
  atom.cpu_type_size = old_type_size;
  for (int& size : atom.cpu_type_size) {
    size *= replicas_;
  }
  for (size_t m = 0; m < group.size(); ++m) {
    group[m].cpu_label.resize(total_atoms);
    for (int replica = 0; replica < replicas_; ++replica) {
      for (int n = 0; n < original_atoms; ++n) {
        group[m].cpu_label[replica * original_atoms + n] = old_labels[m][n];
      }
    }
    group[m].find_size(total_atoms, static_cast<int>(m));
    group[m].find_contents(total_atoms);
  }
  allocate_memory_gpu(group, atom, thermo);
}

std::vector<double> Ensemble_QCT::evaluate_batch_potential_energy(
  Atom& atom, Box& box, std::vector<Group>& group, Force& force) const
{
  force.compute(
    box,
    atom.position_per_atom,
    atom.type,
    group,
    atom.potential_per_atom,
    atom.force_per_atom,
    atom.virial_per_atom);
  std::vector<double> potential(static_cast<size_t>(atom.number_of_atoms), 0.0);
  atom.potential_per_atom.copy_to_host(potential.data());
  std::vector<double> result(replicas_, 0.0);
  for (int index = 0; index < atom.number_of_atoms; ++index) {
    result[index / atoms_per_replica_] += potential[index];
  }
  return result;
}

void Ensemble_QCT::write_initial_outputs(const QCT_Modes& qct_modes, const Box& box) const
{
  std::ofstream summary_output("qct_initial_summary.csv");
  if (!summary_output.is_open()) {
    qct_input_error("Cannot open qct_initial_summary.csv for writing.");
  }
  summary_output << std::setprecision(17);
  summary_output
    << "replica,seed,total_sampled_energy_eV,rotational_energy_eV,reaction_energy_eV,"
       "potential_correction_eV,stable_velocity_scale,wigner_weight,log_wigner_weight\n";
  for (int replica = 0; replica < replicas_; ++replica) {
    const auto& point = sampled_points_[replica];
    summary_output << replica << ',' << point.seed << ',' << point.total_sampled_energy << ','
                   << point.rotational_energy << ',' << point.reaction_energy << ','
                   << point.potential_correction << ',' << point.stable_velocity_scale << ','
                   << point.wigner_weight << ',' << point.log_wigner_weight << '\n';
  }
  if (!summary_output) {
    qct_input_error("Failed while writing qct_initial_summary.csv.");
  }

  std::ofstream mode_output("qct_initial.out");
  if (!mode_output.is_open()) {
    qct_input_error("Cannot open qct_initial.out for writing.");
  }
  mode_output << std::setprecision(17);
  mode_output << "# QCT_INITIAL v2\n";
  mode_output << "# replicas " << replicas_ << "\n";
  mode_output << "# temperature_K " << sample_temperature_ << "\n";
  mode_output << "# zpe " << (sampling_mode_ == Sampling_Mode::wigner
                                    ? std::string("intrinsic")
                                    : (zpe_ ? "yes" : "no")) << "\n";
  mode_output << "# phase " << (phase_mode_ == Phase_Mode::random ? "random" : "zero") << "\n";
  mode_output << "# reaction_mode " << qct_modes.reaction_mode_index << "\n";
  mode_output << "# sampling ";
  if (sampling_mode_ == Sampling_Mode::canonical) {
    mode_output << "canonical\n";
  } else if (sampling_mode_ == Sampling_Mode::microcanonical) {
    mode_output << "microcanonical\n";
  } else if (sampling_mode_ == Sampling_Mode::mode_energy) {
    mode_output << "mode_energy\n";
  } else if (sampling_mode_ == Sampling_Mode::semiclassical) {
    mode_output << "semiclassical\n";
  } else {
    mode_output << "wigner\n";
  }
  mode_output << "replica,seed,mode,frequency_THz,active,energy_eV,phase_rad,Q_sqrt_amu_A,P_sqrt_eV\n";
  for (int replica = 0; replica < replicas_; ++replica) {
    const auto& point = sampled_points_[replica];
    for (size_t mode_index = 0; mode_index < qct_modes.modes.size(); ++mode_index) {
      const auto& mode = qct_modes.modes[mode_index];
      const auto& sampled_mode = point.modes[mode_index];
      mode_output << replica << ',' << point.seed << ',' << mode.index << ','
                  << mode.frequency_THz << ',' << (mode.active ? "yes" : "no") << ','
                  << sampled_mode.energy << ',' << sampled_mode.phase << ','
                  << sampled_mode.Q << ',' << sampled_mode.P << '\n';
    }
  }
  if (!mode_output) {
    qct_input_error("Failed while writing qct_initial.out.");
  }

  std::ofstream phase_point_output("qct_initial.xyz");
  if (!phase_point_output.is_open()) {
    qct_input_error("Cannot open qct_initial.xyz for writing.");
  }
  phase_point_output << std::setprecision(17);
  for (int replica = 0; replica < replicas_; ++replica) {
    const auto& point = sampled_points_[replica];
    phase_point_output << qct_modes.num_atoms << '\n';
    phase_point_output << "pbc=\"" << (box.pbc_x ? 'T' : 'F') << ' '
                       << (box.pbc_y ? 'T' : 'F') << ' ' << (box.pbc_z ? 'T' : 'F')
                       << "\" Lattice=\"" << box.cpu_h[0] << ' ' << box.cpu_h[3] << ' '
                       << box.cpu_h[6] << ' ' << box.cpu_h[1] << ' ' << box.cpu_h[4] << ' '
                       << box.cpu_h[7] << ' ' << box.cpu_h[2] << ' ' << box.cpu_h[5] << ' '
                       << box.cpu_h[8]
                       << "\" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3 Replica="
                       << replica << " Seed=" << point.seed << " temperature="
                       << sample_temperature_ << "\n";
    for (int n = 0; n < qct_modes.num_atoms; ++n) {
      phase_point_output << qct_modes.symbol[n];
      for (int d = 0; d < 3; ++d) {
        phase_point_output << ' ' << point.position[n + qct_modes.num_atoms * d];
      }
      phase_point_output << ' ' << qct_modes.mass[n];
      for (int d = 0; d < 3; ++d) {
        phase_point_output << ' '
                          << point.velocity[n + qct_modes.num_atoms * d] / TIME_UNIT_CONVERSION;
      }
      phase_point_output << '\n';
    }
  }
  if (!phase_point_output) {
    qct_input_error("Failed while writing qct_initial.xyz.");
  }
}

void Ensemble_QCT::initialize_harmonic_replicas(
  Atom& atom, Box& box, std::vector<Group>& group, GPU_Vector<double>& thermo, Force& force)
{
  if (replicas_ > 1 && (box.pbc_x || box.pbc_y || box.pbc_z)) {
    qct_input_error("QCT batch currently requires pbc=F F F in model.xyz.");
  }
  atoms_per_replica_ = atom.number_of_atoms;
  QCT_Modes qct_modes;
  if (mode_source_ == Mode_Source::automatic_hessian) {
    qct_modes = build_automatic_modes(atom, box, group, force);
  } else if (mode_source_ == Mode_Source::qct_modes) {
    qct_modes = read_qct_modes(atom);
  } else {
    qct_modes = read_gpumd_modes(atom);
  }
  classify_stationary_point(qct_modes);

  replica_seeds_.clear();
  sampled_points_.clear();
  replica_seeds_.reserve(replicas_);
  sampled_points_.reserve(replicas_);
  for (int replica = 0; replica < replicas_; ++replica) {
    const std::uint64_t base_seed =
      static_cast<std::uint64_t>(seed_) + static_cast<std::uint64_t>(replica);
    const std::uint64_t replica_seed = qct_sampling_seed(base_seed, 0);
    replica_seeds_.emplace_back(replica_seed);
    sampled_points_.emplace_back(sample_harmonic_point(qct_modes, replica_seed));
  }

  const bool is_wigner = sampling_mode_ == Sampling_Mode::wigner;

  bool needs_potential_correction = false;
  for (const auto& point : sampled_points_) {
    needs_potential_correction = needs_potential_correction || point.total_sampled_energy > 0.0;
  }
  double reference_potential = 0.0;
  if (needs_potential_correction) {
    reference_potential = evaluate_potential_energy(
      force, box, atom, group, qct_modes.reference_position);
  }

  if (replicas_ > 1) {
    expand_atom_for_batch(atom, group, thermo);
    force.configure_qct_batch(atoms_per_replica_, replicas_);
  } else {
    atom.cpu_position_per_atom = sampled_points_.front().position;
    atom.position_per_atom.copy_from_host(atom.cpu_position_per_atom.data());
  }

  if (is_wigner) {
    // Wigner (LSC-IVR) path: every sample is statistically valid, so we
    // skip the apply_potential_correction resampling loop.  Instead we
    // compute anharmonic reweighting weights from the true PES.
    std::vector<double> sampled_potentials;
    if (anharmonic_reweight_) {
      sampled_potentials = evaluate_batch_potential_energy(atom, box, group, force);
    }
    for (int replica = 0; replica < replicas_; ++replica) {
      auto& point = sampled_points_[replica];
      if (anharmonic_reweight_) {
        const double v_real = sampled_potentials[replica] - reference_potential;
        double v_harmonic = 0.0;
        // Compute harmonic potential relative to reference:
        // sum_k 0.5 * omega_k^2 * Q_k^2
        for (size_t mi = 0; mi < qct_modes.modes.size(); ++mi) {
          const auto& mode = qct_modes.modes[mi];
          if (!mode.active || mi == static_cast<size_t>(qct_modes.reaction_mode_index)) continue;
          const double omega = mode.frequency_THz * THZ_TO_NATURAL_ANGULAR_FREQUENCY;
          const double Q = point.modes[mi].Q;
          v_harmonic += 0.5 * omega * omega * Q * Q;
        }
        const double dv = v_real - v_harmonic;
        const double beta = (sample_temperature_ > 0.0) ? 1.0 / (K_B * sample_temperature_) : 0.0;
        point.log_wigner_weight = -beta * dv;
        point.wigner_weight = std::exp(point.log_wigner_weight);
        point.potential_correction = dv;
      } else {
        point.wigner_weight = 1.0;
        point.log_wigner_weight = 0.0;
        point.potential_correction = 0.0;
      }
    }
  } else {

  std::vector<int> sampling_attempts(replicas_, 0);
  std::vector<bool> pending_correction(replicas_, false);
  for (int replica = 0; replica < replicas_; ++replica) {
    pending_correction[replica] = sampled_points_[replica].total_sampled_energy > 0.0;
  }
  int number_of_resampled_points = 0;
  if (needs_potential_correction) {
    const int maximum_sampling_attempts = 100;
    while (true) {
      const std::vector<double> sampled_potentials =
        evaluate_batch_potential_energy(atom, box, group, force);
      bool all_points_accepted = true;
      for (int replica = 0; replica < replicas_; ++replica) {
        if (!pending_correction[replica]) {
          continue;
        }
        if (apply_potential_correction(
              qct_modes,
              sampled_points_[replica],
              reference_potential,
              sampled_potentials[replica])) {
          pending_correction[replica] = false;
          continue;
        }

        all_points_accepted = false;
        ++sampling_attempts[replica];
        ++number_of_resampled_points;
        if (sampling_attempts[replica] >= maximum_sampling_attempts) {
          qct_input_error(
            "QCT could not generate an energetically valid phase point after 100 attempts for replica " +
            std::to_string(replica) + ".");
        }
        const std::uint64_t base_seed =
          static_cast<std::uint64_t>(seed_) + static_cast<std::uint64_t>(replica);
        const std::uint64_t retry_seed =
          qct_sampling_seed(base_seed, sampling_attempts[replica]);
        sampled_points_[replica] = sample_harmonic_point(qct_modes, retry_seed);
        replica_seeds_[replica] = retry_seed;
      }
      if (all_points_accepted) {
        break;
      }
      for (int replica = 0; replica < replicas_; ++replica) {
        if (!pending_correction[replica]) {
          continue;
        }
        const auto& point = sampled_points_[replica];
        for (int n = 0; n < atoms_per_replica_; ++n) {
          const int destination = replica * atoms_per_replica_ + n;
          for (int d = 0; d < 3; ++d) {
            atom.cpu_position_per_atom[destination + atom.number_of_atoms * d] =
              point.position[n + atoms_per_replica_ * d];
          }
        }
      }
      atom.position_per_atom.copy_from_host(atom.cpu_position_per_atom.data());
    }
  }
  if (number_of_resampled_points > 0) {
    printf(
      "    resampled %d energetically invalid harmonic phase point(s).\n",
      number_of_resampled_points);
  }
  } // end non-wigner correction branch

  for (int replica = 0; replica < replicas_; ++replica) {
    const auto& point = sampled_points_[replica];
    for (int n = 0; n < atoms_per_replica_; ++n) {
      const int destination = replica * atoms_per_replica_ + n;
      for (int d = 0; d < 3; ++d) {
        atom.cpu_position_per_atom[destination + atom.number_of_atoms * d] =
          point.position[n + atoms_per_replica_ * d];
        atom.cpu_velocity_per_atom[destination + atom.number_of_atoms * d] =
          point.velocity[n + atoms_per_replica_ * d];
      }
    }
  }
  atom.position_per_atom.copy_from_host(atom.cpu_position_per_atom.data());
  atom.velocity_per_atom.copy_from_host(atom.cpu_velocity_per_atom.data());
  write_initial_outputs(qct_modes, box);

  if (replicas_ > 1) {
    printf("    generated native QCT batch with %d replicas (%d atoms per replica).\n",
           replicas_,
           atoms_per_replica_);
    printf("    shared QCT mode energy is sampled independently for every replica.\n");
  } else {
    const char* sampling_name =
      sampling_mode_ == Sampling_Mode::canonical
        ? "canonical"
        : sampling_mode_ == Sampling_Mode::microcanonical
            ? "microcanonical"
            : sampling_mode_ == Sampling_Mode::mode_energy
                ? "mode_energy"
                : sampling_mode_ == Sampling_Mode::semiclassical ? "semiclassical EBK" : "wigner";
    int number_of_active_modes = 0;
    for (const auto& mode : qct_modes.modes) {
      number_of_active_modes += mode.active ? 1 : 0;
    }
    const auto& point = sampled_points_.front();
    printf("    generated harmonic QCT initial condition (sampling mode is %s).\n", sampling_name);
    printf("    number of active QCT modes is %d (stable modes).\n", number_of_active_modes);
    if (qct_modes.reaction_mode_index >= 0) {
      printf(
        "    sampled saddle reaction energy is %g eV on mode %d.\n",
        point.reaction_energy,
        qct_modes.reaction_mode_index);
    }
    if (sampling_mode_ == Sampling_Mode::semiclassical) {
      printf(
        "    semiclassical quantum numbers are v=%d and J=%d; rotational energy is %g eV.\n",
        vibrational_quantum_,
        rotational_quantum_,
        point.rotational_energy);
    }
    printf("    sampled QCT mode energy is %g eV.\n", point.total_sampled_energy);
  }
}

void Ensemble_QCT::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
#ifdef USE_NEPCG
  velocity_verlet_cg(
    true,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);
#else
  velocity_verlet(
    true,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);
#endif
}

void Ensemble_QCT::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
#ifdef USE_NEPCG
  velocity_verlet_cg(
    false,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);
#else
  velocity_verlet(
    false,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);
#endif

  if (!is_batch()) {
    find_thermo(
      false,
      box.get_volume(),
      group,
      atom.mass,
      atom.potential_per_atom,
      atom.velocity_per_atom,
      atom.virial_per_atom,
      thermo);
  }
}
