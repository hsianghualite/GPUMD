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
NEP_EFA: NEP-based Euclidean Fast Attention layer added on top of NEP.

Design (composition with NEP):

  * E_total = E_NEP + E_EFA, with E_EFA a global, linear-complexity attention
    energy computed in reciprocal (k-space) form using the Ewald mesh machinery
    already present in src/force/ewald.cu.
  * NEP_EFA holds an internal NEP member that computes E_NEP (energy, forces,
    virial, ZBL).  The EFA pass then ADDS E_EFA (and its forces/virial) on top
    of the same per-atom buffers.
  * The EFA layer has its own descriptor coefficients c (separate from NEP's c)
    and its own single-output ANN (w0/b0/w1/b1) producing a per-atom attention
    weight q_n, which is fed into the Ewald reciprocal kernel exactly like a
    partial charge in NEP_Charge.
  * Parameter layout in the flat SNES vector (per device):
        [ NEP params (number_of_variables_ann + number_of_variables_descriptor) ]
        [ EFA params (number_of_variables_efa)                                   ]
    Total per device = number_of_variables (= NEP + EFA).
------------------------------------------------------------------------------*/

#pragma once
#include "nep.cuh"
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"

class Parameters;
class Dataset;

class NEP_EFA : public Potential
{
public:
  // EFA-side parameter metadata.  Mirrors the scalar-cutoff style of
  // NEP_Charge::ParaMB (single cutoff, not per-type) so the EFA kernels can
  // use the same simple addressing as nep_charge.cu.
  struct ParaMB {
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_zbl_factor = 0.65f;
    float rc_radial = 0.0f;     // radial cutoff (EFA uses the same single cutoff)
    float rc_angular = 0.0f;    // angular cutoff
    int basis_size_radial = 0;
    int basis_size_angular = 0;
    int n_max_radial = 0;
    int n_max_angular = 0;
    int L_max = 0;
    int has_q_222;
    int has_q_1111;
    int has_q_112;
    int has_q_123;
    int has_q_233;
    int has_q_134;
    int dim_angular;
    int num_L;
    int num_types = 0;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int version = 4;
    // EFA-specific hyper-parameters.
    int efa_l_max = 3;
    int efa_num_radial = 4;
    int efa_dim = 0;
    float efa_omega_max = 6.0f;
    int efa_num_kpoints_max = 50000;
  };

  // EFA ANN: single-output head producing the per-atom attention weight q_n.
  // The descriptor coefficients c are EFA's own (separate from NEP's c).
  struct ANN {
    int dim = 0;                    // dimension of the descriptor
    int num_neurons1 = 0;           // number of neurons in the hidden layer
    int num_para = 0;               // number of parameters in the EFA head ANN
    const float* w0[NUM_ELEMENTS];  // weight: descriptor -> hidden
    const float* b0[NUM_ELEMENTS];  // bias: hidden layer
    const float* w1[NUM_ELEMENTS];  // weight: hidden -> per-atom attention scalar q_n
    const float* b1;                // bias for the output scalar
    const float* c;                 // EFA descriptor coefficients (own block)
  };

  struct NEP_EFA_Data {
    GPU_Vector<float> descriptors;       // EFA power-spectrum descriptors
    GPU_Vector<float> attention_weight;  // per-atom scalar q_n (Ewald "charge" analog)
    GPU_Vector<float> attention_grad;    // d q_n / d(descriptor) for back-prop
    GPU_Vector<float> sum_fxyz;           // retained for ABI-compatible storage
    GPU_Vector<float> parameters;        // EFA parameters (own slice)
    // Ewald-style reciprocal-space buffers (mirrors NEP_Charge_Data)
    GPU_Vector<float> kx;
    GPU_Vector<float> ky;
    GPU_Vector<float> kz;
    GPU_Vector<float> G;        // kernel weight per k-point (includes e^{-k^2/4alpha^2})
    GPU_Vector<float> S_real;   // structure factor real part
    GPU_Vector<float> S_imag;   // structure factor imag part
    GPU_Vector<float> D_real;   // dE/dq_n (reciprocal-space gradient)
    GPU_Vector<int> num_kpoints;
    // Cached forward A(l,m) accumulators for reuse in force kernel (Step A2)
    GPU_Vector<float> cached_A_real;
    GPU_Vector<float> cached_A_imag;
  };

  struct EFA_Para {
    int num_kpoints_max = 50000;
    float alpha = 0.5f;                 // Ewald screening parameter (1/(2 Angstrom))
    float alpha_factor = 1.0f;          // 1/(4*alpha^2)
    float omega_max = 6.0f;             // maximum frequency (1/Angstrom)
    float omega_min = 0.5f;             // minimum frequency
    int num_omega = 4;                  // number of frequency samples
    float lambda_e = 1.0f;              // weight for EFA energy loss (debug)
  };

  NEP_EFA(
    Parameters& para,
    int N,
    int Nc,
    int version,
    int deviceCount);
  void find_force(
    Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    int deviceCount) override;

private:
  // Composition: the NEP member computes E_NEP (energy, forces, virial, ZBL).
  // Its find_force() is called first; the EFA pass then accumulates E_EFA.
  NEP nep;

  ParaMB paramb;
  ANN annmb[16];
  NEP_EFA_Data nep_data[16];
  EFA_Para efa_para;
  void update_potential(float* parameters, ANN& ann);
};
