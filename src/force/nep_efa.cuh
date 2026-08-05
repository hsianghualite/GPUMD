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
NEP_EFA (MD side): NEP-based Euclidean Fast Attention layer added on top
of NEP, with E_total = E_NEP + E_EFA.

Design (mirrors the training-side src/main_nep/nep_efa.cuh):

  * NEP_EFA holds an internal NEP member that computes E_NEP (energy, forces,
    virial, ZBL) via NEP::compute().  The EFA pass then ADDS E_EFA (and its
    forces/virial) on top of the same per-atom buffers.
  * EFA reuses the NEP neighbor lists (NEP::nep_data.NN_radial etc., which are
    public) but has its own descriptor coefficients c and its own single-output
    ANN (w0/b0/w1/b1) producing a per-atom attention weight q_n, which is fed
    into the Ewald reciprocal kernel exactly like a partial charge in
    NEP_Charge.
  * The EFA k-space formula differs from the Coulomb Ewald formula used by the
    Ewald class: the EFA energy is a sum over k of G(k) * |S(k)|^2 (a global,
    "attention-score" energy), not the per-atom q*G*S*cos/sin formula.  We
    therefore use EFA-specific k-space kernels here, NOT the Ewald class.
------------------------------------------------------------------------------*/

#pragma once
#include "dftd3.cuh"
#include "neighbor.cuh"
#include "nep.cuh"
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"

struct NEP_EFA_Data {
  GPU_Vector<float> descriptors; // EFA power-spectrum descriptors
  // EFA's own descriptor and ANN scratch space (separate from NEP's)
  GPU_Vector<float> f12x; // retained for the legacy EFA storage layout
  GPU_Vector<float> f12y;
  GPU_Vector<float> f12z;
  GPU_Vector<float> attention_grad; // d(q_n)/d(descriptor) per atom (pre-chain-rule)
  GPU_Vector<float> sum_fxyz;  // angular back-prop scratch
  GPU_Vector<float> parameters;// EFA parameters (own slice of nep.txt)
  // Ewald-style reciprocal-space buffers (mirrors NEP_Charge_Data)
  GPU_Vector<float> kx;
  GPU_Vector<float> ky;
  GPU_Vector<float> kz;
  GPU_Vector<float> G;
  GPU_Vector<float> S_real;
  GPU_Vector<float> S_imag;
  GPU_Vector<float> D_real;     // dE_EFA/dq_n (reciprocal-space gradient)
  GPU_Vector<float> attention;  // per-atom attention weight q_n (Ewald "charge" analog)
  // Cached forward A(l,m) accumulators for reuse in force kernel (Step A2)
  // Layout: [N * efa_dim * 9] for real and imag parts separately.
  // For atom n1, channel d, m-index (0..8): cached_A_real[n1 + (d*9+m)*N]
  GPU_Vector<float> cached_A_real;
  GPU_Vector<float> cached_A_imag;
};

class NEP_EFA : public Potential
{
public:
  using Potential::compute;

  NEP_EFA_Data nep_efa_data;

  struct ParaMB {
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_zbl_factor = 0.65f;
    float rc_radial = 0.0f;     // radial cutoff (single, like NEP_Charge)
    float rc_angular = 0.0f;
    float rcinv_radial = 0.0f;
    float rcinv_angular = 0.0f;
    int MN_radial = 200;
    int MN_angular = 100;
    int n_max_radial = 0;
    int n_max_angular = 0;
    int L_max = 0;
    int dim_angular;
    int has_q_222 = 0;
    int has_q_1111 = 0;
    int has_q_112 = 0;
    int has_q_123 = 0;
    int has_q_233 = 0;
    int has_q_134 = 0;
    int num_L;
    int basis_size_radial = 8;
    int basis_size_angular = 8;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int num_types = 0;
    int version = 4;
    // EFA-specific hyper-parameters
    int efa_l_max = 3;
    int efa_num_radial = 4;
    int efa_dim = 0;
    float efa_omega_max = 6.0f;
  };

  struct ANN {
    int dim = 0;
    int num_neurons1 = 0;
    int num_para = 0;          // total EFA params (ANN + descriptor)
    int num_para_ann = 0;      // EFA ANN params only
    const float* w0[NUM_ELEMENTS];
    const float* b0[NUM_ELEMENTS];
    const float* w1[NUM_ELEMENTS];
    const float* b1;
    const float* c;            // EFA descriptor coefficients (own block)
    const float* q_scaler_efa; // EFA's own descriptor scaler
  };

  struct ExpandedBox {
    int num_cells[3];
    float h[18];
  };

  struct EFA_Para {
    int num_kpoints_max = 1;
    int num_kpoints = 1;
    float alpha = 0.5f;          // Ewald screening parameter
    float alpha_factor = 1.0f;  // 1/(4*alpha^2)
    float omega_max = 6.0f;
    float omega_min = 0.5f;
    int num_omega = 4;
    float lambda_e = 1.0f;
  };

  NEP_EFA(const char* file_potential, const int num_atoms);
  virtual ~NEP_EFA(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

private:
  NEP nep;  // Composition: computes E_NEP via nep.compute()
  ParaMB paramb;
  ANN annmb;
  ExpandedBox ebox;
  EFA_Para efa_para;
  bool k_mesh_valid = false;
  double k_mesh_box[9] = {0.0};

  void update_potential(float* parameters, ANN& ann);

  void find_k_and_G(const double* box);

  void compute_efa_pass(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
};
