/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#include "dump_qct.cuh"
#include "integrate/ensemble_qct.cuh"
#include "integrate/integrate.cuh"
#include "model/box.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

Dump_QCT::Dump_QCT(const char** param, int num_param)
{
  property_name = "dump_qct";
  if (num_param != 2 && num_param != 4 && num_param != 6) {
    PRINT_INPUT_ERROR(
      "dump_qct should have an interval and optional trajectory/thermo key-value pairs.");
  }
  if (!is_valid_int(param[1], &dump_interval_) || dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("QCT dump interval should be a positive integer.");
  }
  bool has_trajectory = false;
  bool has_thermo = false;
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
  fprintf(thermo_, "replica,step,time_fs,temperature_K,kinetic_energy_eV,potential_energy_eV,total_energy_eV\n");
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
    const double temperature_replica =
      2.0 * kinetic_energy / (3.0 * atoms_per_replica_ * K_B);
    fprintf(
      thermo_,
      "%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g\n",
      replica,
      step + 1,
      time_fs,
      temperature_replica,
      kinetic_energy,
      potential_energy,
      kinetic_energy + potential_energy);
  }
  fflush(trajectory_);
  fflush(thermo_);
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
}
