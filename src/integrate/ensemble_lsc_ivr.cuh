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
Linearized Semiclassical Initial Value Representation (LSC-IVR) ensemble.

This ensemble inherits from the QCT ensemble and reuses its Wigner-sampling
machinery to generate quantum-corrected initial conditions.  The key
difference from QCT is that LSC-IVR is designed to work with periodic
boundary conditions (PBC), making it suitable for condensed-phase systems
such as crystalline solids and liquids.

Usage in run.in:
    ensemble lsc_ivr T [key=value ...]

When replicas=1 (the default), the system is propagated as a single NVE
trajectory with quantum-corrected Wigner initial conditions.  This allows
straightforward use with compute_hac for quantum-corrected thermal
conductivity via the Green-Kubo relation.
------------------------------------------------------------------------------*/

#pragma once
#include "ensemble_qct.cuh"

class Ensemble_LSC_IVR : public Ensemble_QCT
{
public:
  // Transforms "ensemble lsc_ivr T ..." into "ensemble qct wigner temperature T ..."
  // and delegates to the Ensemble_QCT constructor.
  Ensemble_LSC_IVR(const char** param, int num_param);
  virtual ~Ensemble_LSC_IVR(void) = default;

  // Override to allow periodic boundary conditions and print LSC-IVR info.
  virtual void initialize_before_run(
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    GPU_Vector<double>& thermo,
    Force& force) override;

private:
  bool periodic_system_ = false;
};
