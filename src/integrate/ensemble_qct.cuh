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
#include <string>
#include <vector>

class Ensemble_QCT : public Ensemble
{
public:
  Ensemble_QCT(const char** param, int num_param);
  virtual ~Ensemble_QCT(void);

  virtual void initialize_before_run(
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    GPU_Vector<double>& thermo);

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

private:
  enum class Init_Mode { phase_point, harmonic };
  enum class Phase_Mode { random, zero };

  struct Normal_Mode {
    int index = -1;
    double frequency_THz = 0.0;
    bool active = false;
    std::vector<double> eigenvector;
  };

  struct QCT_Modes {
    int num_atoms = 0;
    int num_modes = 0;
    std::vector<std::string> symbol;
    std::vector<double> mass;
    std::vector<double> reference_position;
    std::vector<Normal_Mode> modes;
  };

  Init_Mode init_mode_ = Init_Mode::phase_point;
  Phase_Mode phase_mode_ = Phase_Mode::random;
  std::string modes_file_;
  double sample_temperature_ = 0.0;
  int seed_ = 1;
  int replicas_ = 1;
  bool zpe_ = true;
  bool initialized_ = false;

  void parse_harmonic(const char** param, int num_param);
  void parse_phase_point(const char** param, int num_param);
  QCT_Modes read_qct_modes(const Atom& atom) const;
  void initialize_harmonic(Atom& atom);
};
