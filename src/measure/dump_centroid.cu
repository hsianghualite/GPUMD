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

/*-----------------------------------------------------------------------------------------------100
Dump centroid (bead-averaged) trajectory in PIMD/RPMD/TRPMD runs.

The centroid is Q_bar = (1/P) * sum_k q_k, where P is the number of beads
and q_k is the per-bead position.  Velocity centroid is analogous.

Output format: extended XYZ, one frame per dump step, compatible with
lsc_ivr.py for RPMD Kubo correlation functions.
--------------------------------------------------------------------------------------------------*/

#include "dump_centroid.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <cstdio>

Dump_Centroid::Dump_Centroid(const char** param, int num_param)
{
  property_name = "dump_centroid";

  if (num_param != 2 && num_param != 4 && num_param != 6) {
    PRINT_INPUT_ERROR(
      "dump_centroid should have an interval and optional trajectory/thermo key-value pairs.");
  }
  if (!is_valid_int(param[1], &dump_interval_) || dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("dump_centroid interval should be a positive integer.");
  }

  bool has_trajectory = false;
  bool has_thermo = false;
  for (int i = 2; i < num_param; i += 2) {
    if (strcmp(param[i], "trajectory") == 0) {
      if (has_trajectory) {
        PRINT_INPUT_ERROR("dump_centroid trajectory may only be specified once.");
      }
      trajectory_filename_ = param[i + 1];
      has_trajectory = true;
    } else if (strcmp(param[i], "thermo") == 0) {
      if (has_thermo) {
        PRINT_INPUT_ERROR("dump_centroid thermo may only be specified once.");
      }
      thermo_filename_ = param[i + 1];
      has_thermo = true;
    } else {
      PRINT_INPUT_ERROR("Unknown key for dump_centroid.");
    }
  }
  printf("Dump centroid trajectory every %d steps.\n", dump_interval_);
}

void Dump_Centroid::preprocess(
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

  if (atom.number_of_beads <= 0) {
    PRINT_INPUT_ERROR("dump_centroid requires a PIMD/RPMD/TRPMD ensemble (number_of_beads > 0).");
  }
  number_of_beads_ = atom.number_of_beads;

  const int N = atom.number_of_atoms;
  cpu_position_.resize(N * 3);
  cpu_velocity_.resize(N * 3);
  cpu_potential_.resize(N);
  cpu_mass_.resize(N);

  trajectory_ = my_fopen(trajectory_filename_.c_str(), "w");
  thermo_ = my_fopen(thermo_filename_.c_str(), "w");
  fprintf(thermo_, "step,time_fs,kinetic_energy_eV,potential_energy_eV,total_energy_eV\n");
}

static void average_beads_to_host(
  const int N,
  const int P,
  const std::vector<GPU_Vector<double>>& bead_data,
  std::vector<double>& cpu_out)
{
  // bead_data[k] has layout [x0..xN-1, y0..yN-1, z0..zN-1] on device
  std::vector<double> tmp(N * 3);
  for (int d = 0; d < 3; ++d)
    for (int n = 0; n < N; ++n)
      cpu_out[d * N + n] = 0.0;

  for (int k = 0; k < P; ++k) {
    bead_data[k].copy_to_host(tmp.data());
    for (int i = 0; i < N * 3; ++i)
      cpu_out[i] += tmp[i];
  }

  const double inv_P = 1.0 / P;
  for (int i = 0; i < N * 3; ++i)
    cpu_out[i] *= inv_P;
}

void Dump_Centroid::process(
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

  if ((step + 1) % dump_interval_ != 0)
    return;

  const int N = atom.number_of_atoms;

  // Compute centroid position and velocity from bead arrays
  average_beads_to_host(N, number_of_beads_, atom.position_beads, cpu_position_);
  average_beads_to_host(N, number_of_beads_, atom.velocity_beads, cpu_velocity_);

  // Copy mass from device (same for all beads)
  atom.mass.copy_to_host(cpu_mass_.data());

  // Compute potential energy: average of bead potentials
  if ((int)atom.potential_per_atom.size() >= N) {
    // atom.potential_per_atom already holds the centroid-averaged potential
    // after compute2 -> gpu_average
    atom.potential_per_atom.copy_to_host(cpu_potential_.data());
  } else {
    for (int n = 0; n < N; ++n)
      cpu_potential_[n] = 0.0;
  }

  const double time_fs = global_time * TIME_UNIT_CONVERSION;
  const double velocity_scale = 1.0 / TIME_UNIT_CONVERSION;

  // Write trajectory frame in extended XYZ format
  fprintf(trajectory_, "%d\n", N);
  fprintf(
    trajectory_,
    "Time=%.12g pbc=\"%c %c %c\" Lattice=\"%.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g\" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3 Beads=%d Step=%d\n",
    time_fs,
    box.pbc_x ? 'T' : 'F',
    box.pbc_y ? 'T' : 'F',
    box.pbc_z ? 'T' : 'F',
    box.cpu_h[0], box.cpu_h[3], box.cpu_h[6],
    box.cpu_h[1], box.cpu_h[4], box.cpu_h[7],
    box.cpu_h[2], box.cpu_h[5], box.cpu_h[8],
    number_of_beads_,
    step + 1);

  double kinetic_energy = 0.0;
  double potential_energy = 0.0;

  for (int n = 0; n < N; ++n) {
    const double vx_nat = cpu_velocity_[n];
    const double vy_nat = cpu_velocity_[n + N];
    const double vz_nat = cpu_velocity_[n + 2 * N];
    const double vx = vx_nat * velocity_scale;
    const double vy = vy_nat * velocity_scale;
    const double vz = vz_nat * velocity_scale;

    kinetic_energy += 0.5 * cpu_mass_[n] *
                      (vx_nat * vx_nat + vy_nat * vy_nat + vz_nat * vz_nat);
    potential_energy += cpu_potential_[n];

    fprintf(
      trajectory_,
      "%s %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
      atom.cpu_atom_symbol[n].c_str(),
      cpu_position_[n],
      cpu_position_[n + N],
      cpu_position_[n + 2 * N],
      cpu_mass_[n],
      vx, vy, vz);
  }

  fprintf(
    thermo_,
    "%d,%.12g,%.12g,%.12g,%.12g\n",
    step + 1,
    time_fs,
    kinetic_energy,
    potential_energy,
    kinetic_energy + potential_energy);

  fflush(trajectory_);
  fflush(thermo_);
}

void Dump_Centroid::postprocess(
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
