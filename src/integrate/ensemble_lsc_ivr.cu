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
#include <cstring>
#include <string>
#include <vector>

// File-scope static storage for the transformed parameter strings.
// Safe because GPUMD is single-threaded during setup, and Ensemble_QCT
// copies any string values into its own members during construction.
namespace
{
std::vector<std::string> g_param_strings;
std::vector<std::vector<char>> g_param_buffers;
std::vector<const char*> g_param_argv;
} // namespace

Ensemble_LSC_IVR::Ensemble_LSC_IVR(const char** param, int num_param)
  : Ensemble_QCT([](const char** p, int n) -> const char** {
      // Transform:  ensemble lsc_ivr T [key=value ...]
      // Into:        ensemble qct wigner temperature T [key=value ...]
      //
      // We replace param[1]="lsc_ivr" with "qct", insert "wigner" at
      // position 2, insert "temperature" at position 3, then copy T and
      // all remaining key-value pairs from the original positions 2..n-1.
      g_param_strings.clear();
      g_param_buffers.clear();
      g_param_argv.clear();

      g_param_strings.reserve(n + 2); // +2 for "wigner" and "temperature"
      g_param_strings.push_back(std::string(p[0])); // "ensemble"
      g_param_strings.push_back("qct");              // replaces "lsc_ivr"
      g_param_strings.push_back("wigner");           // sampling mode
      g_param_strings.push_back("temperature");      // keyword before T
      for (int i = 2; i < n; ++i) {
        g_param_strings.push_back(std::string(p[i])); // T and key-value pairs
      }

      g_param_buffers.reserve(g_param_strings.size());
      g_param_argv.reserve(g_param_strings.size());
      for (auto& s : g_param_strings) {
        g_param_buffers.emplace_back(s.begin(), s.end());
        g_param_buffers.back().push_back('\0');
        g_param_argv.push_back(g_param_buffers.back().data());
      }
      return g_param_argv.data();
    }(param, num_param),
    num_param + 2) // +2 for "wigner" and "temperature"
{
  // The base class constructor has already parsed everything as
  // "qct wigner temperature T ..." and set up Wigner sampling.
  // Override the ensemble type to distinguish LSC-IVR from QCT.
  type = -14;

  printf("Use LSC-IVR ensemble for this run.\n");
  printf("    LSC-IVR uses Wigner quantum-corrected initial conditions.\n");
  printf("    Periodic boundary conditions are supported for condensed-phase systems.\n");
}

void Ensemble_LSC_IVR::initialize_before_run(
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Force& force)
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
