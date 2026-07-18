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
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

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

std::vector<std::vector<double>> build_rigid_basis(const Atom& atom)
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
} // namespace

Molecular_Hessian_Result Molecular_Hessian::compute(
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

  const auto rigid_basis = build_rigid_basis(atom);
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
  return result;
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
