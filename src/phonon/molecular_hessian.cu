/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "molecular_hessian.cuh"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/common.cuh"
#include "utilities/cusolver_wrapper.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <utility>
#include <vector>

struct Molecular_Hessian_Device_Data {
  explicit Molecular_Hessian_Device_Data(const GPU_Vector<double>& eigenvectors, int dimension);

  void sample(
    const std::vector<double>& q_coefficients,
    const std::vector<double>& p_coefficients,
    int replicas,
    int number_of_atoms,
    const std::vector<double>& mass,
    const std::vector<double>& reference_position,
    std::vector<double>& positions,
    std::vector<double>& velocities) const;

  void set_num_rigid_modes(int n) { num_rigid_modes = n; }
  int get_num_rigid_modes() const { return num_rigid_modes; }

  GPU_Vector<double> eigenvectors;
  int dimension = 0;
  int num_rigid_modes = 0;
};

namespace
{
double dot(const std::vector<double>& a, const std::vector<double>& b)
{
  double value = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    value += a[i] * b[i];
  }
  return value;
}

bool append_orthonormal(std::vector<double> vector, std::vector<std::vector<double>>& basis)
{
  for (const auto& existing : basis) {
    const double projection = dot(vector, existing);
    for (size_t i = 0; i < vector.size(); ++i) {
      vector[i] -= projection * existing[i];
    }
  }
  const double norm2 = dot(vector, vector);
  if (norm2 <= 1.0e-20) {
    return false;
  }
  const double inverse_norm = 1.0 / std::sqrt(norm2);
  for (double& component : vector) {
    component *= inverse_norm;
  }
  basis.emplace_back(std::move(vector));
  return true;
}

std::vector<std::vector<double>> build_rigid_basis(const Atom& atom, const Box& box)
{
  const int number_of_atoms = atom.number_of_atoms;
  const int dimension = number_of_atoms * 3;
  double total_mass = 0.0;
  double center[3] = {0.0, 0.0, 0.0};
  for (int n = 0; n < number_of_atoms; ++n) {
    const double mass = atom.cpu_mass[n];
    total_mass += mass;
    for (int d = 0; d < 3; ++d) {
      center[d] += mass * atom.cpu_position_per_atom[n + number_of_atoms * d];
    }
  }
  for (double& component : center) {
    component /= total_mass;
  }

  std::vector<std::vector<double>> basis;
  for (int direction = 0; direction < 3; ++direction) {
    std::vector<double> translation(dimension, 0.0);
    for (int n = 0; n < number_of_atoms; ++n) {
      translation[n + number_of_atoms * direction] = std::sqrt(atom.cpu_mass[n]);
    }
    append_orthonormal(std::move(translation), basis);
  }

  if (box.pbc_x || box.pbc_y || box.pbc_z) {
    return basis;
  }

  for (int axis = 0; axis < 3; ++axis) {
    std::vector<double> rotation(dimension, 0.0);
    for (int n = 0; n < number_of_atoms; ++n) {
      const double x = atom.cpu_position_per_atom[n] - center[0];
      const double y = atom.cpu_position_per_atom[n + number_of_atoms] - center[1];
      const double z = atom.cpu_position_per_atom[n + number_of_atoms * 2] - center[2];
      const double mass_sqrt = std::sqrt(atom.cpu_mass[n]);
      if (axis == 0) {
        rotation[n + number_of_atoms] = -mass_sqrt * z;
        rotation[n + number_of_atoms * 2] = mass_sqrt * y;
      } else if (axis == 1) {
        rotation[n] = mass_sqrt * z;
        rotation[n + number_of_atoms * 2] = -mass_sqrt * x;
      } else {
        rotation[n] = -mass_sqrt * y;
        rotation[n + number_of_atoms] = mass_sqrt * x;
      }
    }
    append_orthonormal(std::move(rotation), basis);
  }
  return basis;
}

std::vector<double> build_vibrational_complement(
  int dimension, const std::vector<std::vector<double>>& rigid_basis)
{
  std::vector<std::vector<double>> all_basis = rigid_basis;
  const int vibrational_dimension = dimension - static_cast<int>(rigid_basis.size());
  std::vector<double> complement(static_cast<size_t>(dimension) * vibrational_dimension, 0.0);
  int column = 0;
  for (int unit = 0; unit < dimension && column < vibrational_dimension; ++unit) {
    std::vector<double> vector(dimension, 0.0);
    vector[unit] = 1.0;
    const int before = static_cast<int>(all_basis.size());
    if (append_orthonormal(std::move(vector), all_basis) &&
        static_cast<int>(all_basis.size()) > before) {
      const auto& normalized = all_basis.back();
      for (int row = 0; row < dimension; ++row) {
        complement[row + dimension * column] = normalized[row];
      }
      ++column;
    }
  }
  if (column != vibrational_dimension) {
    PRINT_INPUT_ERROR("Unable to construct a complete molecular vibrational basis.\n");
  }
  return complement;
}

void evaluate_force(
  Force& force,
  Box& box,
  Atom& atom,
  std::vector<Group>& group,
  const std::vector<double>& position,
  std::vector<double>& force_cpu)
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
  atom.force_per_atom.copy_to_host(force_cpu.data());
}

Molecular_Hessian_Result finish_hessian(
  Molecular_Hessian_Result result,
  const Box& box,
  const Atom& atom)
{
  const int number_of_atoms = atom.number_of_atoms;
  const int dimension = number_of_atoms * 3;
  for (int column = 0; column < dimension; ++column) {
    for (int row = column + 1; row < dimension; ++row) {
      const double symmetric_value =
        0.5 * (result.hessian[row + dimension * column] + result.hessian[column + dimension * row]);
      result.hessian[row + dimension * column] = symmetric_value;
      result.hessian[column + dimension * row] = symmetric_value;
    }
  }

  std::vector<double> dynamical_matrix = result.hessian;
  for (int column = 0; column < dimension; ++column) {
    const int atom_column = column % number_of_atoms;
    for (int row = 0; row < dimension; ++row) {
      const int atom_row = row % number_of_atoms;
      dynamical_matrix[row + dimension * column] /=
        std::sqrt(atom.cpu_mass[atom_row] * atom.cpu_mass[atom_column]);
    }
  }

  const auto rigid_basis = build_rigid_basis(atom, box);
  result.number_of_rigid_modes = static_cast<int>(rigid_basis.size());
  const int vibrational_dimension = dimension - result.number_of_rigid_modes;
  if (vibrational_dimension <= 0) {
    PRINT_INPUT_ERROR("QCT Hessian has no vibrational degrees of freedom.\n");
  }
  const std::vector<double> complement = build_vibrational_complement(dimension, rigid_basis);
  std::vector<double> projected(static_cast<size_t>(vibrational_dimension) * vibrational_dimension, 0.0);
  std::vector<double> temporary(static_cast<size_t>(dimension) * vibrational_dimension, 0.0);
  for (int column = 0; column < vibrational_dimension; ++column) {
    for (int row = 0; row < dimension; ++row) {
      double value = 0.0;
      for (int inner = 0; inner < dimension; ++inner) {
        value += dynamical_matrix[row + dimension * inner] * complement[inner + dimension * column];
      }
      temporary[row + dimension * column] = value;
    }
  }
  for (int column = 0; column < vibrational_dimension; ++column) {
    for (int row = 0; row < vibrational_dimension; ++row) {
      double value = 0.0;
      for (int inner = 0; inner < dimension; ++inner) {
        value += complement[inner + dimension * row] * temporary[inner + dimension * column];
      }
      projected[row + vibrational_dimension * column] = value;
    }
  }

  std::vector<double> omega2_natural(vibrational_dimension, 0.0);
  std::vector<double> projected_eigenvectors(
    static_cast<size_t>(vibrational_dimension) * vibrational_dimension, 0.0);
  eigenvectors_symmetric_Jacobi(
    vibrational_dimension,
    projected.data(),
    omega2_natural.data(),
    projected_eigenvectors.data());

  result.omega2_THz2.assign(dimension, 0.0);
  result.eigenvectors.assign(static_cast<size_t>(dimension) * dimension, 0.0);
  for (int mode = 0; mode < result.number_of_rigid_modes; ++mode) {
    for (int row = 0; row < dimension; ++row) {
      result.eigenvectors[row + dimension * mode] = rigid_basis[mode][row];
    }
  }
  const double natural_to_THz2 = 1.0e6 / (TIME_UNIT_CONVERSION * TIME_UNIT_CONVERSION);
  for (int mode = 0; mode < vibrational_dimension; ++mode) {
    const int full_mode = result.number_of_rigid_modes + mode;
    result.omega2_THz2[full_mode] = omega2_natural[mode] * natural_to_THz2;
    for (int row = 0; row < dimension; ++row) {
      double value = 0.0;
      for (int inner = 0; inner < vibrational_dimension; ++inner) {
        value += complement[row + dimension * inner] *
                 projected_eigenvectors[inner + vibrational_dimension * mode];
      }
      result.eigenvectors[row + dimension * full_mode] = value;
    }
  }
  for (int mode = 0; mode < dimension; ++mode) {
    int largest_component = 0;
    for (int row = 1; row < dimension; ++row) {
      if (std::fabs(result.eigenvectors[row + dimension * mode]) >
          std::fabs(result.eigenvectors[largest_component + dimension * mode])) {
        largest_component = row;
      }
    }
    if (result.eigenvectors[largest_component + dimension * mode] < 0.0) {
      for (int row = 0; row < dimension; ++row) {
        result.eigenvectors[row + dimension * mode] *= -1.0;
      }
    }
  }
  return result;
}

Molecular_Hessian_Result finish_full_periodic_hessian(
  Molecular_Hessian_Result result,
  const Box& box,
  const Atom& atom,
  const std::vector<double>& eigenvalues,
  const std::vector<double>& eigenvectors)
{
  const int dimension = atom.number_of_atoms * 3;
  const auto rigid_basis = build_rigid_basis(atom, box);
  result.number_of_rigid_modes = static_cast<int>(rigid_basis.size());
  if (result.number_of_rigid_modes != 3 || static_cast<int>(eigenvalues.size()) != dimension ||
      eigenvectors.size() != static_cast<size_t>(dimension) * dimension) {
    PRINT_INPUT_ERROR("Full periodic QCT Hessian returned an invalid eigensystem.\n");
  }
  std::vector<int> order(dimension);
  for (int mode = 0; mode < dimension; ++mode) {
    order[mode] = mode;
  }
  std::stable_sort(order.begin(), order.end(), [&](const int left, const int right) {
    return std::fabs(eigenvalues[left]) < std::fabs(eigenvalues[right]);
  });
  std::vector<bool> rigid_source(dimension, false);
  // Validate that the modes selected as rigid are actually near-zero.
  // A real soft mode with small |omega2| should not be misclassified.
  const double rigid_threshold = 1.0e-6;  // natural units (rad/ps)^2
  for (int mode = 0; mode < result.number_of_rigid_modes; ++mode) {
    const double omega2 = eigenvalues[order[mode]];
    if (std::fabs(omega2) > rigid_threshold) {
      std::fprintf(
        stderr,
        "QCT periodic Hessian: mode %d has |omega2|=%.6g which exceeds the "
        "rigid-mode threshold %.6g; the structure may not be at a stationary "
        "point or the Hessian may be ill-conditioned.\n",
        order[mode], std::fabs(omega2), rigid_threshold);
      PRINT_INPUT_ERROR(
        "Rigid-mode identification failed: a selected rigid mode is not near-zero.");
    }
    rigid_source[order[mode]] = true;
  }

  result.omega2_THz2.assign(dimension, 0.0);
  result.eigenvectors.assign(static_cast<size_t>(dimension) * dimension, 0.0);
  for (int mode = 0; mode < result.number_of_rigid_modes; ++mode) {
    for (int row = 0; row < dimension; ++row) {
      result.eigenvectors[row + dimension * mode] = rigid_basis[mode][row];
    }
  }
  const double natural_to_THz2 = 1.0e6 / (TIME_UNIT_CONVERSION * TIME_UNIT_CONVERSION);
  int destination = result.number_of_rigid_modes;
  for (int source = 0; source < dimension; ++source) {
    if (rigid_source[source]) {
      continue;
    }
    result.omega2_THz2[destination] = eigenvalues[source] * natural_to_THz2;
    for (int row = 0; row < dimension; ++row) {
      result.eigenvectors[row + dimension * destination] =
        eigenvectors[row + dimension * source];
    }
    ++destination;
  }
  for (int mode = 0; mode < dimension; ++mode) {
    int largest_component = 0;
    for (int row = 1; row < dimension; ++row) {
      if (std::fabs(result.eigenvectors[row + dimension * mode]) >
          std::fabs(result.eigenvectors[largest_component + dimension * mode])) {
        largest_component = row;
      }
    }
    if (result.eigenvectors[largest_component + dimension * mode] < 0.0) {
      for (int row = 0; row < dimension; ++row) {
        result.eigenvectors[row + dimension * mode] *= -1.0;
      }
    }
  }
  return result;
}

__global__ void qct_hessian_add_coordinate(double* position, const int coordinate, const double delta)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    position[coordinate] += delta;
  }
}

__global__ void qct_hessian_difference_column(
  const double* force_negative,
  const double* force_positive,
  double* hessian,
  const int dimension,
  const int column,
  const double scale)
{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < dimension) {
    hessian[row + dimension * column] =
      (force_negative[row] - force_positive[row]) * scale;
  }
}

__global__ void qct_hessian_symmetrize(double* hessian, const int dimension)
{
  // Use a 2D grid: blockIdx.{y,x} maps directly to (row, column) with
  // row > column, avoiding the O(dimension) while-loop decomposition.
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_base = blockIdx.y * blockDim.y + threadIdx.y;
  if (column >= dimension || row_base >= dimension - column - 1) {
    return;
  }
  const int row = column + 1 + row_base;
  const double value = 0.5 *
    (hessian[row + dimension * column] + hessian[column + dimension * row]);
  hessian[row + dimension * column] = value;
  hessian[column + dimension * row] = value;
}

__global__ void qct_hessian_mass_weight(
  double* matrix,
  const double* mass,
  const int number_of_atoms,
  const int dimension)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= dimension * dimension) {
    return;
  }
  const int row = index % dimension;
  const int column = index / dimension;
  const int atom_row = row % number_of_atoms;
  const int atom_column = column % number_of_atoms;
  matrix[index] /= sqrt(mass[atom_row] * mass[atom_column]);
}

__global__ void qct_mode_accumulate(
  const double* eigenvectors,
  const double* q_coefficients,
  const double* p_coefficients,
  const double* mass,
  const double* reference_position,
  double* positions,
  double* velocities,
  const int dimension,
  const int number_of_atoms,
  const int replicas,
  const int num_rigid_modes)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= dimension * replicas) {
    return;
  }
  const int coordinate = index % dimension;
  const int replica = index / dimension;
  const double inverse_mass_sqrt = 1.0 / sqrt(mass[coordinate % number_of_atoms]);
  double position = reference_position[coordinate];
  double velocity = 0.0;
  // Skip rigid modes (columns 0..num_rigid_modes-1); their Q/P coefficients
  // are zero by construction, but skipping them explicitly avoids relying on
  // that assumption and prevents any rigid-mode contamination.
  for (int mode = num_rigid_modes; mode < dimension; ++mode) {
    const double basis = eigenvectors[coordinate + dimension * mode] * inverse_mass_sqrt;
    position += basis * q_coefficients[mode + dimension * replica];
    velocity += basis * p_coefficients[mode + dimension * replica];
  }
  positions[index] = position;
  velocities[index] = velocity;
}

void qct_hessian_progress(
  const Molecular_Hessian_Options& options,
  const int completed,
  const int dimension,
  const int force_evaluations,
  const std::chrono::steady_clock::time_point start)
{
  if (!options.report_progress) {
    return;
  }
  const int interval = options.progress_interval > 0
                         ? options.progress_interval
                         : std::max(1, (dimension + 11) / 12);
  if (completed != 1 && completed != dimension && completed % interval != 0) {
    return;
  }
  const double elapsed = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - start).count();
  const double eta = completed > 0 ? elapsed * (dimension - completed) / completed : 0.0;
  std::printf(
    "Hessian progress: finite_difference %d/%d columns (%.2f%%), "
    "force_evaluations=%d/%d, elapsed=%.1f s, ETA=%.1f s\n",
    completed,
    dimension,
    100.0 * completed / dimension,
    force_evaluations,
    1 + 2 * dimension,
    elapsed,
    eta);
  std::fflush(stdout);
}
} // namespace

Molecular_Hessian_Device_Data::Molecular_Hessian_Device_Data(
  const GPU_Vector<double>& eigenvectors_in,
  const int dimension_in)
  : eigenvectors(eigenvectors_in.size()), dimension(dimension_in)
{
  eigenvectors.copy_from_device(eigenvectors_in.data());
}

void Molecular_Hessian_Device_Data::sample(
  const std::vector<double>& q_coefficients,
  const std::vector<double>& p_coefficients,
  const int replicas,
  const int number_of_atoms,
  const std::vector<double>& mass,
  const std::vector<double>& reference_position,
  std::vector<double>& positions,
  std::vector<double>& velocities) const
{
  if (dimension != number_of_atoms * 3 || q_coefficients.size() !=
        static_cast<size_t>(dimension) * replicas ||
      p_coefficients.size() != q_coefficients.size() ||
      reference_position.size() != static_cast<size_t>(dimension) ||
      mass.size() != static_cast<size_t>(number_of_atoms)) {
    PRINT_INPUT_ERROR("Invalid dimensions passed to GPU QCT mode sampling.\n");
  }
  GPU_Vector<double> q_device(q_coefficients.size());
  GPU_Vector<double> p_device(p_coefficients.size());
  GPU_Vector<double> mass_device(mass.size());
  GPU_Vector<double> reference_device(reference_position.size());
  const size_t device_size = static_cast<size_t>(dimension) * replicas;
  GPU_Vector<double> positions_device(device_size);
  GPU_Vector<double> velocities_device(device_size);
  q_device.copy_from_host(q_coefficients.data());
  p_device.copy_from_host(p_coefficients.data());
  mass_device.copy_from_host(mass.data());
  reference_device.copy_from_host(reference_position.data());
  const int block_size = 256;
  qct_mode_accumulate<<<
    (device_size + block_size - 1) / block_size,
    block_size>>>(
    eigenvectors.data(),
    q_device.data(),
    p_device.data(),
    mass_device.data(),
    reference_device.data(),
    positions_device.data(),
    velocities_device.data(),
    dimension,
    number_of_atoms,
    replicas,
    num_rigid_modes);
  GPU_CHECK_KERNEL
  positions.resize(positions_device.size());
  velocities.resize(velocities_device.size());
  positions_device.copy_to_host(positions.data());
  velocities_device.copy_to_host(velocities.data());
}

void Molecular_Hessian::sample_device(
  const std::shared_ptr<Molecular_Hessian_Device_Data>& device_data,
  const std::vector<double>& q_coefficients,
  const std::vector<double>& p_coefficients,
  const int replicas,
  const int number_of_atoms,
  const std::vector<double>& mass,
  const std::vector<double>& reference_position,
  std::vector<double>& positions,
  std::vector<double>& velocities)
{
  if (!device_data) {
    PRINT_INPUT_ERROR("GPU QCT mode sampling requested without a device basis.\n");
  }
  device_data->sample(
    q_coefficients,
    p_coefficients,
    replicas,
    number_of_atoms,
    mass,
    reference_position,
    positions,
    velocities);
}

Molecular_Hessian_Result Molecular_Hessian::compute_cpu(
  const double displacement,
  Force& force,
  Box& box,
  Atom& atom,
  std::vector<Group>& group)
{
  if (displacement <= 0.0) {
    PRINT_INPUT_ERROR("QCT Hessian displacement should be positive.\n");
  }
  if (atom.number_of_atoms < 2) {
    PRINT_INPUT_ERROR("QCT Hessian requires at least two atoms.\n");
  }

  const int number_of_atoms = atom.number_of_atoms;
  const int dimension = number_of_atoms * 3;
  const std::vector<double> reference_position = atom.cpu_position_per_atom;
  std::vector<double> force_positive(dimension, 0.0);
  std::vector<double> force_negative(dimension, 0.0);
  std::vector<double> force_reference(dimension, 0.0);

  evaluate_force(force, box, atom, group, reference_position, force_reference);
  Molecular_Hessian_Result result;
  result.number_of_atoms = number_of_atoms;
  for (double component : force_reference) {
    result.max_force = std::max(result.max_force, std::fabs(component));
  }
  result.hessian.assign(static_cast<size_t>(dimension) * dimension, 0.0);

  for (int column = 0; column < dimension; ++column) {
    std::vector<double> displaced_position = reference_position;
    displaced_position[column] += displacement;
    evaluate_force(force, box, atom, group, displaced_position, force_positive);

    displaced_position[column] -= 2.0 * displacement;
    evaluate_force(force, box, atom, group, displaced_position, force_negative);

    for (int row = 0; row < dimension; ++row) {
      result.hessian[row + dimension * column] =
        (force_negative[row] - force_positive[row]) / (2.0 * displacement);
    }
  }

  atom.cpu_position_per_atom = reference_position;
  atom.position_per_atom.copy_from_host(reference_position.data());
  return finish_hessian(std::move(result), box, atom);
}

Molecular_Hessian_Result Molecular_Hessian::compute_device(
  const Molecular_Hessian_Options& options,
  Force& force,
  Box& box,
  Atom& atom,
  std::vector<Group>& group)
{
  const double displacement = options.displacement;
  if (displacement <= 0.0) {
    PRINT_INPUT_ERROR("QCT Hessian displacement should be positive.\n");
  }
  if (atom.number_of_atoms < 2) {
    PRINT_INPUT_ERROR("QCT Hessian requires at least two atoms.\n");
  }

  const int number_of_atoms = atom.number_of_atoms;
  const int dimension = number_of_atoms * 3;
  const size_t dense_elements = static_cast<size_t>(dimension) * dimension;
  size_t free_memory = 0;
  size_t total_memory = 0;
  CHECK(gpuMemGetInfo(&free_memory, &total_memory));
  const size_t requested = dense_elements * 2 * sizeof(double) +
                          static_cast<size_t>(dimension) * 5 * sizeof(double);
  if (free_memory < requested) {
    std::fprintf(
      stderr,
      "QCT GPU Hessian requires at least %zu bytes, but only %zu bytes are free.\n",
      requested,
      free_memory);
    PRINT_INPUT_ERROR("Insufficient GPU memory for the QCT Hessian.\n");
  }

  GPU_Vector<double> reference(dimension);
  GPU_Vector<double> working(dimension);
  GPU_Vector<double> force_positive(dimension);
  GPU_Vector<double> force_reference(dimension);
  GPU_Vector<double> hessian(dense_elements);
  GPU_Vector<double> dynamical_matrix(dense_elements);
  GPU_Vector<double> mass(number_of_atoms);
  reference.copy_from_host(atom.cpu_position_per_atom.data());
  mass.copy_from_host(atom.cpu_mass.data());
  working.copy_from_device(reference.data());

  atom.position_per_atom.copy_from_device(reference.data());
  force.compute(
    box,
    atom.position_per_atom,
    atom.type,
    group,
    atom.potential_per_atom,
    atom.force_per_atom,
    atom.virial_per_atom);
  force_reference.copy_from_device(atom.force_per_atom.data());

  Molecular_Hessian_Result result;
  result.number_of_atoms = number_of_atoms;
  std::vector<double> force_reference_host(dimension, 0.0);
  force_reference.copy_to_host(force_reference_host.data());
  for (double component : force_reference_host) {
    result.max_force = std::max(result.max_force, std::fabs(component));
  }

  const int block_size = 256;
  const int vector_blocks = (dimension + block_size - 1) / block_size;
  const double inverse_difference = 1.0 / (2.0 * displacement);
  const auto start = std::chrono::steady_clock::now();
  for (int column = 0; column < dimension; ++column) {
    working.copy_from_device(reference.data());
    qct_hessian_add_coordinate<<<1, 1>>>(working.data(), column, displacement);
    GPU_CHECK_KERNEL
    atom.position_per_atom.copy_from_device(working.data());
    force.compute(
      box,
      atom.position_per_atom,
      atom.type,
      group,
      atom.potential_per_atom,
      atom.force_per_atom,
      atom.virial_per_atom);
    force_positive.copy_from_device(atom.force_per_atom.data());

    working.copy_from_device(reference.data());
    qct_hessian_add_coordinate<<<1, 1>>>(working.data(), column, -displacement);
    GPU_CHECK_KERNEL
    atom.position_per_atom.copy_from_device(working.data());
    force.compute(
      box,
      atom.position_per_atom,
      atom.type,
      group,
      atom.potential_per_atom,
      atom.force_per_atom,
      atom.virial_per_atom);
    qct_hessian_difference_column<<<vector_blocks, block_size>>>(
      atom.force_per_atom.data(),
      force_positive.data(),
      hessian.data(),
      dimension,
      column,
      inverse_difference);
    GPU_CHECK_KERNEL
    qct_hessian_progress(options, column + 1, dimension, 1 + 2 * (column + 1), start);
  }

  atom.position_per_atom.copy_from_device(reference.data());
  force.compute(
    box,
    atom.position_per_atom,
    atom.type,
    group,
    atom.potential_per_atom,
    atom.force_per_atom,
    atom.virial_per_atom);

  {
    const int sym_block_x = 16;
    const int sym_block_y = 16;
    const int grid_x = (dimension + sym_block_x - 1) / sym_block_x;
    const int max_rows = dimension > 1 ? dimension - 1 : 1;
    const int grid_y = (max_rows + sym_block_y - 1) / sym_block_y;
    dim3 sym_grid(grid_x, grid_y);
    dim3 sym_block(sym_block_x, sym_block_y);
    qct_hessian_symmetrize<<<sym_grid, sym_block>>>(hessian.data(), dimension);
  }
  GPU_CHECK_KERNEL
  std::printf("Hessian progress: symmetrize complete\n");
  std::fflush(stdout);

  dynamical_matrix.copy_from_device(hessian.data());
  qct_hessian_mass_weight<<<(dense_elements + block_size - 1) / block_size, block_size>>>(
    dynamical_matrix.data(), mass.data(), number_of_atoms, dimension);
  GPU_CHECK_KERNEL
  std::printf("Hessian progress: mass_weight complete\n");
  std::fflush(stdout);

  GPU_Vector<double> omega2_device(dimension);
  size_t solver_workspace_bytes = 0;
  const int solver_info = eigenvectors_symmetric_device(
    dimension, dynamical_matrix.data(), omega2_device.data(), &solver_workspace_bytes);
  if (solver_info != 0) {
    std::fprintf(
      stderr,
      "QCT GPU Hessian eigensolver returned info=%d (workspace=%zu bytes).\n",
      solver_info,
      solver_workspace_bytes);
    PRINT_INPUT_ERROR("QCT GPU Hessian eigensolver failed.\n");
  }
  std::printf("Hessian progress: eigensolve complete\n");
  std::fflush(stdout);

  result.hessian.resize(dense_elements);
  std::vector<double> omega2_natural(dimension, 0.0);
  std::vector<double> eigenvectors(dense_elements, 0.0);
  hessian.copy_to_host(result.hessian.data());
  omega2_device.copy_to_host(omega2_natural.data());
  dynamical_matrix.copy_to_host(eigenvectors.data());
  // Do NOT store the raw cuSOLVER eigenvectors in device_data here.
  // finish_full_periodic_hessian reorders the basis (rigid modes first,
  // vibrational modes after).  We must store the *reordered* basis so
  // that qct_mode_accumulate pairs each Q/P coefficient with the correct
  // eigenvector column.  The device_data is created below after the
  // reordering is complete.
  std::printf("Hessian progress: mode_reconstruction complete\n");
  std::fflush(stdout);
  result = finish_full_periodic_hessian(
    std::move(result), box, atom, omega2_natural, eigenvectors);
  // Now upload the reordered eigenvector basis to the device so that the
  // GPU sampling kernel uses the same column order as qct_modes.modes.
  if (result.eigenvectors.size() == static_cast<size_t>(dimension) * dimension) {
    GPU_Vector<double> reordered_eigenvectors_device(result.eigenvectors.size());
    reordered_eigenvectors_device.copy_from_host(result.eigenvectors.data());
    result.device_data = std::make_shared<Molecular_Hessian_Device_Data>(
      reordered_eigenvectors_device, dimension);
    result.device_data->set_num_rigid_modes(result.number_of_rigid_modes);
  }
  return result;
}

Molecular_Hessian_Result Molecular_Hessian::compute(
  const Molecular_Hessian_Options& options,
  Force& force,
  Box& box,
  Atom& atom,
  std::vector<Group>& group)
{
  if (!options.device_resident || !(box.pbc_x || box.pbc_y || box.pbc_z)) {
    return compute_cpu(options.displacement, force, box, atom, group);
  }
#ifdef USE_HIP
  std::printf("Hessian backend: CPU (HIP full-GPU Hessian is not implemented).\n");
  std::fflush(stdout);
  return compute_cpu(options.displacement, force, box, atom, group);
#else
  std::printf("Hessian backend: CUDA device finite-difference path.\n");
  std::fflush(stdout);
  return compute_device(options, force, box, atom, group);
#endif
}

Molecular_Hessian_Result Molecular_Hessian::compute(
  const double displacement,
  Force& force,
  Box& box,
  Atom& atom,
  std::vector<Group>& group)
{
  Molecular_Hessian_Options options;
  options.displacement = displacement;
  return compute(options, force, box, atom, group);
}

void Molecular_Hessian::write_qct_audit(const Molecular_Hessian_Result& result, const Atom& atom)
{
  const int number_of_atoms = result.number_of_atoms;
  const int dimension = number_of_atoms * 3;
  std::ofstream hessian_output("qct_hessian.out");
  if (!hessian_output.is_open()) {
    PRINT_INPUT_ERROR("Cannot open qct_hessian.out for writing.\n");
  }
  hessian_output << std::setprecision(17);
  hessian_output << "# QCT_HESSIAN v1\n";
  hessian_output << "# dimension " << dimension << "\n";
  hessian_output << "# unit eV/A^2\n";
  hessian_output << "# coordinate_order x_all_y_all_z_all\n";
  for (int row = 0; row < dimension; ++row) {
    for (int column = 0; column < dimension; ++column) {
      if (column > 0) {
        hessian_output << ' ';
      }
      hessian_output << result.hessian[row + dimension * column];
    }
    hessian_output << '\n';
  }
  if (!hessian_output) {
    PRINT_INPUT_ERROR("Failed while writing qct_hessian.out.\n");
  }

  std::ofstream eigenvector_output("qct_eigenvector.out", std::ios::binary);
  if (!eigenvector_output.is_open()) {
    PRINT_INPUT_ERROR("Cannot open qct_eigenvector.out for writing.\n");
  }
  for (double omega2 : result.omega2_THz2) {
    const float value = static_cast<float>(omega2);
    eigenvector_output.write(reinterpret_cast<const char*>(&value), sizeof(float));
  }
  for (double eigenvector : result.eigenvectors) {
    const float value = static_cast<float>(eigenvector);
    eigenvector_output.write(reinterpret_cast<const char*>(&value), sizeof(float));
  }
  if (!eigenvector_output) {
    PRINT_INPUT_ERROR("Failed while writing qct_eigenvector.out.\n");
  }

  std::ofstream stationary_output("qct_stationary.xyz");
  if (!stationary_output.is_open()) {
    PRINT_INPUT_ERROR("Cannot open qct_stationary.xyz for writing.\n");
  }
  stationary_output << std::setprecision(17);
  stationary_output << number_of_atoms << '\n';
  stationary_output << "Properties=species:S:1:pos:R:3:mass:R:1\n";
  for (int atom_index = 0; atom_index < number_of_atoms; ++atom_index) {
    stationary_output << atom.cpu_atom_symbol[atom_index];
    for (int direction = 0; direction < 3; ++direction) {
      stationary_output << ' '
                        << atom.cpu_position_per_atom[atom_index + number_of_atoms * direction];
    }
    stationary_output << ' ' << atom.cpu_mass[atom_index] << '\n';
  }
  if (!stationary_output) {
    PRINT_INPUT_ERROR("Failed while writing qct_stationary.xyz.\n");
  }
}
