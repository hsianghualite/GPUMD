/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#include "dump_qct.cuh"
#include "integrate/ensemble_qct.cuh"
#include "integrate/integrate.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>

static const double THZ_TO_NATURAL_ANGULAR_FREQUENCY =
  2.0 * PI * 1.0e-3 * TIME_UNIT_CONVERSION;

Dump_QCT::Dump_QCT(const char** param, int num_param)
{
  property_name = "dump_qct";
  if (num_param != 2 && num_param != 4 && num_param != 6 && num_param != 8) {
    PRINT_INPUT_ERROR(
      "dump_qct should have an interval and optional trajectory/thermo/zpe key-value pairs.");
  }
  if (!is_valid_int(param[1], &dump_interval_) || dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("QCT dump interval should be a positive integer.");
  }
  bool has_trajectory = false;
  bool has_thermo = false;
  bool has_zpe = false;
  for (int i = 2; i < num_param; i += 2) {
    if (strcmp(param[i], "trajectory") == 0) {
      if (has_trajectory) {
        PRINT_INPUT_ERROR("dump_qct trajectory may only be specified once.");
      }
      trajectory_filename_ = param[i + 1];
      has_trajectory = true;
    } else if (strcmp(param[i], "thermo") == 0) {
      if (has_thermo) {
        PRINT_INPUT_ERROR("dump_qct thermo may only be specified once.");
      }
      thermo_filename_ = param[i + 1];
      has_thermo = true;
    } else if (strcmp(param[i], "zpe") == 0) {
      if (has_zpe) {
        PRINT_INPUT_ERROR("dump_qct zpe may only be specified once.");
      }
      zpe_filename_ = param[i + 1];
      zpe_requested_ = true;
      has_zpe = true;
    } else {
      PRINT_INPUT_ERROR("Unknown key for dump_qct.");
    }
  }
  printf("Dump QCT replicas every %d steps.\n", dump_interval_);
}

void Dump_QCT::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  (void)number_of_steps;
  (void)time_step;
  (void)group;
  (void)box;
  (void)force;
  auto* qct = dynamic_cast<Ensemble_QCT*>(integrate.ensemble.get());
  if (qct == nullptr) {
    PRINT_INPUT_ERROR("dump_qct requires ensemble qct.");
  }
  replicas_ = qct->number_of_replicas();
  atoms_per_replica_ = qct->atoms_per_replica();
  if (replicas_ <= 0 || atoms_per_replica_ <= 0 ||
      atom.number_of_atoms != replicas_ * atoms_per_replica_) {
    PRINT_INPUT_ERROR("dump_qct could not match the QCT batch atom layout.");
  }
  cpu_potential_.resize(atom.number_of_atoms);
  trajectory_ = my_fopen(trajectory_filename_.c_str(), "w");
  thermo_ = my_fopen(thermo_filename_.c_str(), "w");
  fprintf(thermo_, "replica,step,time_fs,kinetic_temperature_K,kinetic_energy_eV,potential_energy_eV,total_energy_eV\n");

  // ZPE leakage monitoring setup
  const auto* qct_modes_ptr = qct->get_qct_modes();
  if (qct_modes_ptr != nullptr && zpe_requested_) {
    zpe_monitoring_ = true;
    zpe_file_ = my_fopen(zpe_filename_.c_str(), "w");
    fprintf(zpe_file_, "replica,step,time_fs,mode,frequency_THz,mode_energy_eV,initial_mode_energy_eV,zpe_drift_eV\n");
    // Store mode frequencies and eigenvectors
    num_active_modes_ = 0;
    mode_indices_.clear();
    mode_frequencies_.clear();
    mode_eigenvectors_.clear();
    reference_position_ = qct_modes_ptr->reference_position;
    for (const auto& mode : qct_modes_ptr->modes) {
      if (mode.active && mode.index >= 0) {
        mode_indices_.push_back(mode.index);
        mode_frequencies_.push_back(mode.frequency_THz);
        mode_eigenvectors_.push_back(mode.eigenvector);
        ++num_active_modes_;
      }
    }
    // Compute initial mode energies for each replica
    initial_mode_energies_.resize(replicas_ * num_active_modes_, 0.0);
    atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());
    const int N = atoms_per_replica_;
    for (int r = 0; r < replicas_; ++r) {
      for (int m = 0; m < num_active_modes_; ++m) {
        const auto& ev = mode_eigenvectors_[m];
        double q_dot = 0.0;
        double p_dot = 0.0;
        // Project displacement and velocity onto the mass-weighted mode.
        for (int n = 0; n < N; ++n) {
          const int idx = r * N + n;
          const double dx = atom.cpu_position_per_atom[idx] - reference_position_[n];
          const double dy = atom.cpu_position_per_atom[idx + atom.number_of_atoms] -
                            reference_position_[N + n];
          const double dz = atom.cpu_position_per_atom[idx + 2 * atom.number_of_atoms] -
                            reference_position_[2 * N + n];
          const double vx = atom.cpu_velocity_per_atom[idx];
          const double vy = atom.cpu_velocity_per_atom[idx + atom.number_of_atoms];
          const double vz = atom.cpu_velocity_per_atom[idx + 2 * atom.number_of_atoms];
          const double mass = atom.cpu_mass[idx];
          q_dot += ev[n] * std::sqrt(mass) * dx +
                   ev[N + n] * std::sqrt(mass) * dy +
                   ev[2 * N + n] * std::sqrt(mass) * dz;
          p_dot += ev[n] * std::sqrt(mass) * vx +
                   ev[N + n] * std::sqrt(mass) * vy +
                   ev[2 * N + n] * std::sqrt(mass) * vz;
        }
        const double omega = mode_frequencies_[m] * THZ_TO_NATURAL_ANGULAR_FREQUENCY;
        initial_mode_energies_[r * num_active_modes_ + m] =
          0.5 * (p_dot * p_dot + omega * omega * q_dot * q_dot);
      }
    }
    printf("    ZPE leakage monitoring: %d active modes\n", num_active_modes_);
  }
}

void Dump_QCT::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  (void)number_of_steps;
  (void)fixed_group;
  (void)move_group;
  (void)temperature;
  (void)integrate;
  (void)group;
  (void)thermo;
  (void)force;
  if ((step + 1) % dump_interval_ != 0) {
    return;
  }

  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());
  atom.potential_per_atom.copy_to_host(cpu_potential_.data());
  const double time_fs = global_time * TIME_UNIT_CONVERSION;
  const double velocity_scale = 1.0 / TIME_UNIT_CONVERSION;

  const auto* qct = dynamic_cast<const Ensemble_QCT*>(integrate.ensemble.get());
  const auto& seeds = qct->replica_seeds();
  for (int replica = 0; replica < replicas_; ++replica) {
    const unsigned long long seed =
      replica < static_cast<int>(seeds.size()) ?
        static_cast<unsigned long long>(seeds[replica]) : 0ULL;
    fprintf(trajectory_, "%d\n", atoms_per_replica_);
    fprintf(
      trajectory_,
      "Time=%.12g pbc=\"%c %c %c\" Lattice=\"%.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g\" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3 Replica=%d Step=%d Seed=%llu\n",
      time_fs,
      box.pbc_x ? 'T' : 'F',
      box.pbc_y ? 'T' : 'F',
      box.pbc_z ? 'T' : 'F',
      box.cpu_h[0],
      box.cpu_h[3],
      box.cpu_h[6],
      box.cpu_h[1],
      box.cpu_h[4],
      box.cpu_h[7],
      box.cpu_h[2],
      box.cpu_h[5],
      box.cpu_h[8],
      replica,
      step + 1,
      seed);

    double kinetic_energy = 0.0;
    double potential_energy = 0.0;
    for (int n = 0; n < atoms_per_replica_; ++n) {
      const int index = replica * atoms_per_replica_ + n;
      const double vx_natural = atom.cpu_velocity_per_atom[index];
      const double vy_natural = atom.cpu_velocity_per_atom[index + atom.number_of_atoms];
      const double vz_natural = atom.cpu_velocity_per_atom[index + 2 * atom.number_of_atoms];
      const double vx = vx_natural * velocity_scale;
      const double vy = vy_natural * velocity_scale;
      const double vz = vz_natural * velocity_scale;
      kinetic_energy += 0.5 * atom.cpu_mass[index] *
                        (vx_natural * vx_natural + vy_natural * vy_natural +
                         vz_natural * vz_natural);
      potential_energy += cpu_potential_[index];
      fprintf(
        trajectory_,
        "%s %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
        atom.cpu_atom_symbol[index].c_str(),
        atom.cpu_position_per_atom[index],
        atom.cpu_position_per_atom[index + atom.number_of_atoms],
        atom.cpu_position_per_atom[index + 2 * atom.number_of_atoms],
        atom.cpu_mass[index],
        vx,
        vy,
        vz);
    }
    const double kinetic_temperature_K =
      2.0 * kinetic_energy / (3.0 * atoms_per_replica_ * K_B);
    fprintf(
      thermo_,
      "%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g\n",
      replica,
      step + 1,
      time_fs,
      kinetic_temperature_K,
      kinetic_energy,
      potential_energy,
      kinetic_energy + potential_energy);
  }
  fflush(trajectory_);
  fflush(thermo_);

  // ZPE leakage monitoring: project velocities onto modes and compute mode energies
  if (zpe_monitoring_ && zpe_file_ != nullptr) {
    atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());
    const int N = atoms_per_replica_;
    for (int r = 0; r < replicas_; ++r) {
      for (int m = 0; m < num_active_modes_; ++m) {
        const auto& ev = mode_eigenvectors_[m];
        double q_dot = 0.0;
        double p_dot = 0.0;
        for (int n = 0; n < N; ++n) {
          const int idx = r * N + n;
          const double dx = atom.cpu_position_per_atom[idx] - reference_position_[n];
          const double dy = atom.cpu_position_per_atom[idx + atom.number_of_atoms] -
                            reference_position_[N + n];
          const double dz = atom.cpu_position_per_atom[idx + 2 * atom.number_of_atoms] -
                            reference_position_[2 * N + n];
          const double vx = atom.cpu_velocity_per_atom[idx];
          const double vy = atom.cpu_velocity_per_atom[idx + atom.number_of_atoms];
          const double vz = atom.cpu_velocity_per_atom[idx + 2 * atom.number_of_atoms];
          const double mass = atom.cpu_mass[idx];
          q_dot += ev[n] * std::sqrt(mass) * dx +
                   ev[N + n] * std::sqrt(mass) * dy +
                   ev[2 * N + n] * std::sqrt(mass) * dz;
          p_dot += ev[n] * std::sqrt(mass) * vx +
                   ev[N + n] * std::sqrt(mass) * vy +
                   ev[2 * N + n] * std::sqrt(mass) * vz;
        }
        const double omega = mode_frequencies_[m] * THZ_TO_NATURAL_ANGULAR_FREQUENCY;
        const double mode_energy = 0.5 * (p_dot * p_dot + omega * omega * q_dot * q_dot);
        const double initial_e = initial_mode_energies_[r * num_active_modes_ + m];
        fprintf(zpe_file_, "%d,%d,%.12g,%d,%.6f,%.12e,%.12e,%.12e\n",
                r, step + 1, time_fs, mode_indices_[m], mode_frequencies_[m],
                mode_energy, initial_e, mode_energy - initial_e);
      }
    }
    fflush(zpe_file_);
  }
}

void Dump_QCT::process_initial(
  const int number_of_steps,
  const int fixed_group,
  const int move_group,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  // Reuse the regular writer with a sentinel step so the initial sampled
  // phase point is emitted as Step=0, Time=0 regardless of dump interval.
  process(
    number_of_steps,
    -1,
    fixed_group,
    move_group,
    0.0,
    0.0,
    integrate,
    box,
    group,
    thermo,
    atom,
    force);
}

void Dump_QCT::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  (void)atom;
  (void)box;
  (void)integrate;
  (void)number_of_steps;
  (void)time_step;
  (void)temperature;
  if (trajectory_ != nullptr) {
    fclose(trajectory_);
    trajectory_ = nullptr;
  }
  if (thermo_ != nullptr) {
    fclose(thermo_);
    thermo_ = nullptr;
  }
  if (zpe_file_ != nullptr) {
    fclose(zpe_file_);
    zpe_file_ = nullptr;
  }
}
