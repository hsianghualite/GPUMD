/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#pragma once

#include "property.cuh"
#include <cstdint>
#include <string>
#include <vector>

class Dump_QCT : public Property
{
public:
  Dump_QCT(const char** param, int num_param);

  virtual void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  virtual void process(
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
    Force& force);

  virtual void postprocess(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature);

private:
  int dump_interval_ = 1;
  int replicas_ = 1;
  int atoms_per_replica_ = 0;
  std::string trajectory_filename_ = "qct_trajectory.xyz";
  std::string thermo_filename_ = "qct_thermo.csv";
  FILE* trajectory_ = nullptr;
  FILE* thermo_ = nullptr;
  std::vector<double> cpu_potential_;
};
