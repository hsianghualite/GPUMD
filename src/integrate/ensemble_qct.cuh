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

#pragma once
#include "ensemble.cuh"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct Molecular_Hessian_Device_Data;

class Ensemble_QCT : public Ensemble
{
public:
  Ensemble_QCT(const char** param, int num_param);
  virtual ~Ensemble_QCT(void);

  virtual void initialize_before_run(
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    GPU_Vector<double>& thermo,
    Force& force);

  virtual void compute1(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

  virtual void compute2(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

  bool is_batch() const { return replicas_ > 1; }
  int number_of_replicas() const { return replicas_; }
  int atoms_per_replica() const { return atoms_per_replica_; }
  const std::vector<std::uint64_t>& replica_seeds() const { return replica_seeds_; }
  std::vector<double> replica_log_wigner_weights() const;

private:
  enum class Init_Mode { phase_point, harmonic };
  enum class Phase_Mode { random, zero };
  enum class Mode_Source { automatic_hessian, qct_modes, gpumd_eigenvector };
  enum class Sampling_Mode { canonical, microcanonical, mode_energy, semiclassical, wigner };
  enum class Stationary_Point { automatic, minimum, saddle };
  enum class Reaction_Direction { positive, negative, random };

  struct Normal_Mode {
    int index = -1;
    double frequency_THz = 0.0;
    bool active = false;
    bool rigid = false;
    std::vector<double> eigenvector;
  };

  struct QCT_Modes {
    int num_atoms = 0;
    int num_modes = 0;
    int number_of_rigid_modes = 0;
    int reaction_mode_index = -1;
    std::vector<std::string> symbol;
    std::vector<double> mass;
    std::vector<double> reference_position;
    std::vector<Normal_Mode> modes;
    std::shared_ptr<Molecular_Hessian_Device_Data> device_data;
  };

  struct Sampled_Mode {
    double energy = 0.0;
    double phase = 0.0;
    double Q = 0.0;
    double P = 0.0;
  };

  struct Sampled_Point {
    std::uint64_t seed = 0;
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> rotational_velocity;
    std::vector<Sampled_Mode> modes;
    double total_sampled_energy = 0.0;
    double rotational_energy = 0.0;
    double reaction_energy = 0.0;
    double potential_correction = 0.0;
    double stable_velocity_scale = 1.0;
    double wigner_weight = 1.0;
    double log_wigner_weight = 0.0;
  };

  Init_Mode init_mode_ = Init_Mode::phase_point;
  Phase_Mode phase_mode_ = Phase_Mode::random;
  Mode_Source mode_source_ = Mode_Source::automatic_hessian;
  Sampling_Mode sampling_mode_ = Sampling_Mode::canonical;
  Stationary_Point stationary_point_ = Stationary_Point::automatic;
  Reaction_Direction reaction_direction_ = Reaction_Direction::random;
  std::string modes_file_;
  std::string eigenvector_file_;
  int exclude_lowest_ = 6;
  bool exclude_lowest_specified_ = false;
  double min_frequency_ = 1.0e-3;
  double hessian_displacement_ = 1.0e-3;
  bool hessian_progress_ = true;
  int hessian_progress_interval_ = 0;
  double stationary_force_tolerance_ = 1.0e-3;
  double sample_temperature_ = 0.0;
  double total_energy_eV_ = -1.0;
  double selected_mode_energy_eV_ = -1.0;
  int selected_mode_index_ = -1;
  int vibrational_quantum_ = -1;
  int rotational_quantum_ = -1;
  double reaction_energy_eV_ = -1.0;
  int seed_ = 1;
  int replicas_ = 1;
  int atoms_per_replica_ = 0;
  bool zpe_ = true;
  bool anharmonic_reweight_ = true;
  bool initialized_ = false;
  std::vector<std::uint64_t> replica_seeds_;
  std::vector<Sampled_Point> sampled_points_;
  QCT_Modes qct_modes_;
  bool qct_modes_stored_ = false;

  void parse_harmonic(const char** param, int num_param);
  void parse_phase_point(const char** param, int num_param);
  QCT_Modes read_qct_modes(const Atom& atom) const;
  QCT_Modes read_gpumd_modes(const Atom& atom, const Box& box) const;
  QCT_Modes build_automatic_modes(
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    Force& force) const;
  void classify_stationary_point(QCT_Modes& qct_modes) const;
  void validate_qct_modes(const QCT_Modes& qct_modes, const Atom& atom) const;
  Sampled_Point sample_harmonic_point(
    const QCT_Modes& qct_modes, const std::uint64_t seed) const;
  bool apply_potential_correction(
    const QCT_Modes& qct_modes,
    Sampled_Point& point,
    const double reference_potential,
    const double sampled_potential) const;
  void expand_atom_for_batch(
    Atom& atom, std::vector<Group>& group, GPU_Vector<double>& thermo) const;
  std::vector<double> evaluate_batch_potential_energy(
    Atom& atom, Box& box, std::vector<Group>& group, Force& force) const;
  void write_initial_outputs(const QCT_Modes& qct_modes, const Box& box) const;
  void initialize_harmonic_replicas(
    Atom& atom, Box& box, std::vector<Group>& group, GPU_Vector<double>& thermo, Force& force);

public:
  // Returns the sampling temperature, used by the Integrate layer to set
  // temperature1/temperature2 so that measurement keywords (compute_hac, etc.)
  // receive the correct T for the Green-Kubo prefactor.
  double get_sample_temperature() const { return sample_temperature_; }

  // Access the QCT normal modes (for ZPE leakage monitoring in dump_qct).
  // Returns nullptr if QCT was initialized in phase_point mode (no modes).
  const QCT_Modes* get_qct_modes() const { return qct_modes_stored_ ? &qct_modes_ : nullptr; }
};
