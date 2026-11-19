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

This ensemble inherits from Ensemble_QCT and reuses its Wigner sampling
machinery.  The key difference is that LSC-IVR allows periodic boundary
conditions (PBC), making it suitable for condensed-phase systems.

The constructor transforms the user's "lsc_ivr" ensemble specification
into the equivalent "qct wigner temperature T ..." specification, so that
all parsing and Wigner-sampling code in Ensemble_QCT is reused without
duplication.

  User input:     ensemble lsc_ivr T [key=value ...]
  Transformed to:  ensemble qct wigner temperature T [key=value ...]
------------------------------------------------------------------------------*/

#include "ensemble_lsc_ivr.cuh"
#include "utilities/error.cuh"
#include <string>
#include <vector>

Ensemble_LSC_IVR::Ensemble_LSC_IVR(
  const std::vector<std::string>& tokens,
  const std::string& potential_filename,
  const bool dftd3_enabled)
  : Ensemble_QCT(
      [&tokens]() {
        // Transform: ensemble lsc_ivr T [key=value ...]
        // Into:       ensemble qct wigner temperature T [key=value ...]
        std::vector<std::string> transformed;
        transformed.reserve(tokens.size() + 2);
        transformed.push_back(tokens[0]);
        transformed.push_back("qct");
        transformed.push_back("wigner");
        transformed.push_back("temperature");
        for (size_t i = 2; i < tokens.size(); ++i) {
          transformed.push_back(tokens[i]);
        }
        return transformed;
      }(),
      potential_filename,
      dftd3_enabled)
{
  // The base class constructor has already parsed everything as
  // "qct wigner temperature T ..." and set up Wigner sampling.
  type = EnsembleType::LSC_IVR;

  printf("Use LSC-IVR ensemble for this run.\n");
  printf("    LSC-IVR uses Wigner quantum-corrected initial conditions.\n");
  printf("    Periodic boundary conditions are supported for condensed-phase systems.\n");
}

void Ensemble_LSC_IVR::initialize_before_run(
  Atom& atom, Box& box, std::vector<Group>& group, GPU_Vector<double>& thermo, Force& force)
{
  // For LSC-IVR, we allow periodic boundary conditions when replicas=1.
  // The base QCT class rejects PBC only when replicas>1 (batch mode),
  // so for the typical LSC-IVR case (replicas=1, PBC), no special
  // handling is needed — the base class PBC check passes naturally.
  periodic_system_ = box.pbc_x || box.pbc_y || box.pbc_z;

  Ensemble_QCT::initialize_before_run(atom, box, group, thermo, force);

  if (periodic_system_) {
    printf("    LSC-IVR initialized with periodic boundary conditions.\n");
  }
}
