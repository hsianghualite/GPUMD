/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

/*----------------------------------------------------------------------------80
Finite-difference normal modes for one isolated molecular structure.

Unlike the phonon Hessian, this class does not use a supercell or k-points.
It evaluates one complete force vector for each positive and negative
Cartesian displacement, which is the appropriate O(N) force-evaluation path
for QCT initial-condition sampling.
------------------------------------------------------------------------------*/

#pragma once
#include <memory>
#include <vector>

class Atom;
class Box;
class Force;
class Group;

struct Molecular_Hessian_Device_Data;

struct Molecular_Hessian_Options {
  double displacement = 1.0e-3;
  bool device_resident = true;
  bool report_progress = true;
  int progress_interval = 100;
};

struct Molecular_Hessian_Result {
  int number_of_atoms = 0;
  int number_of_rigid_modes = 0;
  double max_force = 0.0;
  // Cartesian Hessian and eigenvectors use x_all, y_all, z_all ordering.
  std::vector<double> hessian;
  std::vector<double> omega2_THz2;
  std::vector<double> eigenvectors;
  std::shared_ptr<Molecular_Hessian_Device_Data> device_data;
};

class Molecular_Hessian
{
public:
  static Molecular_Hessian_Result compute(
    const Molecular_Hessian_Options& options,
    Force& force,
    Box& box,
    Atom& atom,
    std::vector<Group>& group);

  static Molecular_Hessian_Result compute(
    double displacement,
    Force& force,
    Box& box,
    Atom& atom,
    std::vector<Group>& group);

  static void write_qct_audit(const Molecular_Hessian_Result& result, const Atom& atom);

  static void sample_device(
    const std::shared_ptr<Molecular_Hessian_Device_Data>& device_data,
    const std::vector<double>& q_coefficients,
    const std::vector<double>& p_coefficients,
    int replicas,
    int number_of_atoms,
    const std::vector<double>& mass,
    const std::vector<double>& reference_position,
    std::vector<double>& positions,
    std::vector<double>& velocities);

private:
  static Molecular_Hessian_Result compute_cpu(
    double displacement,
    Force& force,
    Box& box,
    Atom& atom,
    std::vector<Group>& group);

  static Molecular_Hessian_Result compute_device(
    const Molecular_Hessian_Options& options,
    Force& force,
    Box& box,
    Atom& atom,
    std::vector<Group>& group);
};
