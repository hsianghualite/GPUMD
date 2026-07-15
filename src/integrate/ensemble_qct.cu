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
The minimal quasi-classical trajectory (QCT) integrator.

This first version propagates a user-provided QCT phase point with the same
GPU velocity-Verlet scheme as NVE. Normal-mode initial-condition sampling will
be added in front of this propagation path.
------------------------------------------------------------------------------*/

#include "ensemble_qct.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
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
} // namespace

Ensemble_QCT::Ensemble_QCT(const char** param, int num_param)
{
  type = -13;

  if (num_param == 2) {
    init_mode_ = Init_Mode::phase_point;
  } else if (strcmp(param[2], "phase_point") == 0) {
    parse_phase_point(param, num_param);
  } else if (strcmp(param[2], "harmonic") == 0) {
    parse_harmonic(param, num_param);
  } else {
    PRINT_INPUT_ERROR("ensemble qct mode should be phase_point or harmonic.");
  }

  printf("Use QCT ensemble for this run.\n");
  if (init_mode_ == Init_Mode::phase_point) {
    printf("    propagate the supplied quasi-classical phase point with NVE dynamics.\n");
  } else {
    printf("    initialize a harmonic quasi-classical phase point from %s.\n", modes_file_.c_str());
    printf("    sample temperature is %g K.\n", sample_temperature_);
    printf("    random seed is %d.\n", seed_);
    printf("    zero-point energy is %s.\n", zpe_ ? "enabled" : "disabled");
    printf("    phase sampling is %s.\n", phase_mode_ == Phase_Mode::random ? "random" : "zero");
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
  bool has_temperature = false;
  for (int i = 3; i < num_param; i += 2) {
    if (strcmp(param[i], "modes") == 0) {
      modes_file_ = param[i + 1];
      has_modes = true;
    } else if (strcmp(param[i], "temperature") == 0) {
      sample_temperature_ = get_double_from_token(param[i + 1], __FILE__, __LINE__);
      if (sample_temperature_ < 0.0) {
        PRINT_INPUT_ERROR("temperature for ensemble qct harmonic should be non-negative.");
      }
      has_temperature = true;
    } else if (strcmp(param[i], "seed") == 0) {
      seed_ = get_int_from_token(param[i + 1], __FILE__, __LINE__);
      if (seed_ < 0) {
        PRINT_INPUT_ERROR("seed for ensemble qct harmonic should be non-negative.");
      }
    } else if (strcmp(param[i], "replicas") == 0) {
      replicas_ = get_int_from_token(param[i + 1], __FILE__, __LINE__);
      if (replicas_ != 1) {
        PRINT_INPUT_ERROR("ensemble qct harmonic currently supports replicas 1 only.");
      }
    } else if (strcmp(param[i], "zpe") == 0) {
      zpe_ = parse_yes_no(param[i + 1], "zpe");
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

  if (!has_modes) {
    PRINT_INPUT_ERROR("ensemble qct harmonic requires modes FILE.");
  }
  if (!has_temperature) {
    PRINT_INPUT_ERROR("ensemble qct harmonic requires temperature T.");
  }
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
  if (qct_modes.num_atoms != atom.number_of_atoms) {
    qct_input_error("num_atoms in QCT modes file does not match model.xyz.");
  }
  if (qct_modes.num_modes > qct_modes.num_atoms * 3) {
    qct_input_error("num_modes in QCT modes file cannot exceed 3 * num_atoms.");
  }
  if (qct_modes.modes.size() != static_cast<size_t>(qct_modes.num_modes)) {
    qct_input_error("QCT modes block does not contain num_modes mode entries.");
  }

  for (int n = 0; n < qct_modes.num_atoms; ++n) {
    if (
      atom.cpu_atom_symbol.size() == static_cast<size_t>(qct_modes.num_atoms) &&
      qct_modes.symbol[n] != atom.cpu_atom_symbol[n]) {
      qct_input_error("Atom symbols in QCT modes file do not match model.xyz.");
    }
    const double mass_scale = std::max(1.0, std::fabs(atom.cpu_mass[n]));
    if (std::fabs(qct_modes.mass[n] - atom.cpu_mass[n]) > 1.0e-6 * mass_scale) {
      qct_input_error("Atom masses in QCT modes file do not match model.xyz.");
    }
  }

  for (const auto& mode : qct_modes.modes) {
    double norm = 0.0;
    for (const auto& component : mode.eigenvector) {
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

  return qct_modes;
}

void Ensemble_QCT::initialize_before_run(
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo)
{
  if (initialized_) {
    return;
  }

  if (init_mode_ == Init_Mode::harmonic) {
    initialize_harmonic(atom);
  }

  initialized_ = true;
}

void Ensemble_QCT::initialize_harmonic(Atom& atom)
{
  QCT_Modes qct_modes = read_qct_modes(atom);
  std::vector<double> position = qct_modes.reference_position;
  std::vector<double> velocity(qct_modes.num_atoms * 3, 0.0);

  std::mt19937_64 rng(static_cast<unsigned long long>(seed_));
  std::uniform_real_distribution<double> uniform_01(0.0, 1.0);

  int num_active_modes = 0;
  double total_sampled_energy = 0.0;
  for (const auto& mode : qct_modes.modes) {
    if (!mode.active) {
      continue;
    }

    const double omega = mode.frequency_THz * THZ_TO_NATURAL_ANGULAR_FREQUENCY;
    double energy = zpe_ ? 0.5 * HBAR * omega : 0.0;
    if (sample_temperature_ > 0.0) {
      const double u = std::max(uniform_01(rng), std::numeric_limits<double>::min());
      energy += -K_B * sample_temperature_ * std::log(u);
    }

    if (energy <= 0.0) {
      ++num_active_modes;
      continue;
    }

    const double phase =
      phase_mode_ == Phase_Mode::random ? 2.0 * PI * uniform_01(rng) : 0.0;
    const double Q = std::sqrt(2.0 * energy) * std::cos(phase) / omega;
    const double P = -std::sqrt(2.0 * energy) * std::sin(phase);

    for (int n = 0; n < qct_modes.num_atoms; ++n) {
      const double mass_sqrt_inv = 1.0 / std::sqrt(qct_modes.mass[n]);
      for (int d = 0; d < 3; ++d) {
        const int index = n + qct_modes.num_atoms * d;
        const double eigenvector_component = mode.eigenvector[index];
        position[index] += eigenvector_component * Q * mass_sqrt_inv;
        velocity[index] += eigenvector_component * P * mass_sqrt_inv;
      }
    }

    total_sampled_energy += energy;
    ++num_active_modes;
  }

  atom.cpu_position_per_atom = position;
  atom.cpu_velocity_per_atom = velocity;
  atom.position_per_atom.copy_from_host(position.data());
  atom.velocity_per_atom.copy_from_host(velocity.data());

  printf("    generated harmonic QCT initial condition.\n");
  printf("    number of active QCT modes is %d.\n", num_active_modes);
  printf("    sampled harmonic mode energy is %g eV.\n", total_sampled_energy);
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
