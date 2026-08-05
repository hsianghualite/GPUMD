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
NEP_EFA training-side implementation.

The NEP-based EFA layer adds a global, linear-complexity attention energy on
top of the standard NEP energy:

    E_total = E_NEP + E_EFA

E_EFA is computed in reciprocal (k-) space using the Ewald mesh machinery,
treating a per-atom learnable "attention weight" q_n as the structure-factor
weight (the direct analog of a partial charge in NEP_Charge).  The forward
pass is:

    1. compute ERoPE equivariant channels and their power spectrum
    2. apply a small per-type ANN: descriptor -> q_n [scalar per atom]
    3. enforce charge neutrality (sum q_n = 0) per structure
    4. Ewald reciprocal-space sum -> E_EFA, forces, virial
    5. back-propagate dE_EFA/dq_n through the ANN and ERoPE pair derivative

The ERoPE descriptor is rotationally invariant only after the power-spectrum
contraction; its underlying A(l,m,r) channels transform equivariantly. The
same ERoPE value and local finite-difference derivative are used by training
and MD.
------------------------------------------------------------------------------*/

#include "dataset.cuh"
#include "mic.cuh"
#include "nep_efa.cuh"
#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/efa_utilities.cuh"
#include <cstring>

// ============================================================================
// Equivariant EFA descriptor.
//
// For every center atom and radial frequency, A_lmr is an ERoPE spherical
// channel.  The ANN receives the power spectrum sum_m |A_lmr|^2, which is a
// scalar under rotations while retaining the angular information of the
// equivariant channels.  The coefficient layout is
// (l * num_radial + radial) * num_types^2 + type1 * num_types + type2.
// ============================================================================
static __device__ __forceinline__ float efa_cutoff(float r, float rc)
{
  float fc = 0.0f;
  find_fc(rc, 1.0f / rc, r, fc);
  return fc;
}

static __device__ __forceinline__ efa::EFAComplex efa_weighted_kernel(
  float x, float y, float z, float omega, int l, int m, float weight)
{
  return efa::efa_cscale(efa::efa_erope_value(x, y, z, omega, l, m), weight);
}

// Analytic derivative of the power spectrum w.r.t. the pair displacement r_vec.
// Same logic as the MD-side efa_power_derivative_analytic in nep_efa.cu (force).
// Computes d(|a_lm|^2)/d(r_vec) = 2 * Re[ conj(a_lm) * d(a_lm)/d(r_vec) ]
// where d(a_lm)/d(r_vec) = coeff * [ f_cut'(r) * r_hat * K_lm + f_cut(r) * dK_lm/d(r_vec) ]
static __device__ __forceinline__ void efa_power_derivative_analytic_train(
  const efa::EFAComplex* a, float rx, float ry, float rz,
  float omega, int l, float coeff, float rc,
  float& dpx, float& dpy, float& dpz)
{
  float r = sqrtf(rx * rx + ry * ry + rz * rz);
  float rcinv = 1.0f / rc;

  // Cutoff and its derivative
  float fc, fcp;
  find_fc_and_fcp(rc, rcinv, r, fc, fcp);

  dpx = dpy = dpz = 0.0f;

  // d(a_lm)/d(r_vec) = coeff * [ fcp * r_hat * K_lm + fc * dK_lm/d(r_vec) ]
  float rinv = (r < 1.0e-12f) ? 0.0f : 1.0f / r;
  float rhat_x = rx * rinv, rhat_y = ry * rinv, rhat_z = rz * rinv;

  efa::EFAComplex K, dKx, dKy, dKz;
  for (int m = -l; m <= l; ++m) {
    efa::efa_erope_derivative(rx, ry, rz, omega, l, m, K, dKx, dKy, dKz);

    // da/d(r_vec) = coeff * [ fcp * r_hat * K + fc * dK ]
    efa::EFAComplex da_x(
      coeff * (fcp * rhat_x * K.re + fc * dKx.re),
      coeff * (fcp * rhat_x * K.im + fc * dKx.im));
    efa::EFAComplex da_y(
      coeff * (fcp * rhat_y * K.re + fc * dKy.re),
      coeff * (fcp * rhat_y * K.im + fc * dKy.im));
    efa::EFAComplex da_z(
      coeff * (fcp * rhat_z * K.re + fc * dKz.re),
      coeff * (fcp * rhat_z * K.im + fc * dKz.im));

    // conj(a) * da
    efa::EFAComplex conj_a = efa::efa_conj(a[m + l]);
    float re_x = conj_a.re * da_x.re - conj_a.im * da_x.im;
    float re_y = conj_a.re * da_y.re - conj_a.im * da_y.im;
    float re_z = conj_a.re * da_z.re - conj_a.im * da_z.im;

    // d(|a_lm|^2)/d(r_vec) = 2 * Re[ conj(a) * da ]
    dpx += 2.0f * re_x;
    dpy += 2.0f * re_y;
    dpz += 2.0f * re_z;
  }
}

static __global__ void find_descriptors_erope_train(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors,
  float* g_cached_A_real,
  float* g_cached_A_imag)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N)
    return;

  const int t1 = g_type[n1];
  for (int l = 0; l < paramb.efa_l_max; ++l) {
    for (int ir = 0; ir < paramb.efa_num_radial; ++ir) {
      efa::EFAComplex a[9];
      for (int m = 0; m < 9; ++m)
        a[m] = efa::EFAComplex();
      const float omega = paramb.efa_omega_max * float(ir + 1) /
                          float(paramb.efa_num_radial);
      for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
        const int index = g_NN_sum[n1] + i1;
        const int n2 = g_NL[index];
        const int t2 = g_type[n2];
        const float x = g_x12[index];
        const float y = g_y12[index];
        const float z = g_z12[index];
        const float r = sqrtf(x * x + y * y + z * z);
        const float weight = efa_cutoff(r, paramb.rc_radial) *
          annmb.c[(l * paramb.efa_num_radial + ir) * paramb.num_types_sq +
                  t1 * paramb.num_types + t2];
        for (int m = -l; m <= l; ++m) {
          const efa::EFAComplex k = efa_weighted_kernel(x, y, z, omega, l, m, weight);
          a[m + l].re += k.re;
          a[m + l].im += k.im;
        }
      }
      const int d = l * paramb.efa_num_radial + ir;
      float power = 0.0f;
      for (int m = -l; m <= l; ++m)
        power += efa::efa_cabs2(a[m + l]);
      g_descriptors[n1 + d * N] = power;
      // Cache A(l,m) for reuse in force kernel
      const int cache_base = n1 + d * 9 * N;
      for (int m = 0; m < 9; ++m) {
        g_cached_A_real[cache_base + m * N] = a[m].re;
        g_cached_A_imag[cache_base + m * N] = a[m].im;
      }
    }
  }
}

static __global__ void find_force_erope_train(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_attention_grad,
  const float* __restrict__ g_D_real,
  const float* __restrict__ g_cached_A_real,
  const float* __restrict__ g_cached_A_imag,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N)
    return;

  const int t1 = g_type[n1];
  float virial[6] = {0.0f};
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    const int index = g_NN_sum[n1] + i1;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const float x = g_x12[index], y = g_y12[index], z = g_z12[index];
    float f12[3] = {0.0f, 0.0f, 0.0f};
    for (int l = 0; l < paramb.efa_l_max; ++l) {
      for (int ir = 0; ir < paramb.efa_num_radial; ++ir) {
        const int d = l * paramb.efa_num_radial + ir;
        const float omega = paramb.efa_omega_max * float(ir + 1) /
                            float(paramb.efa_num_radial);
        const float coeff = annmb.c[d * paramb.num_types_sq + t1 * paramb.num_types + t2];
        // Read cached A(l,m) from descriptor pass
        efa::EFAComplex a[9];
        {
          const int cache_base = n1 + d * 9 * N;
          for (int m = 0; m < 9; ++m) {
            a[m] = efa::EFAComplex(g_cached_A_real[cache_base + m * N],
                                   g_cached_A_imag[cache_base + m * N]);
          }
        }
#ifdef EFA_USE_FINITE_DIFF
        const float h = 1.0e-3f;
        for (int axis = 0; axis < 3; ++axis) {
          float xp = x, yp = y, zp = z;
          float xm = x, ym = y, zm = z;
          if (axis == 0) { xp += h; xm -= h; }
          if (axis == 1) { yp += h; ym -= h; }
          if (axis == 2) { zp += h; zm -= h; }
          const float rp = sqrtf(xp * xp + yp * yp + zp * zp);
          const float rm = sqrtf(xm * xm + ym * ym + zm * zm);
          const float wp = coeff * efa_cutoff(rp, paramb.rc_radial);
          const float wm = coeff * efa_cutoff(rm, paramb.rc_radial);
          float dp = 0.0f;
          for (int m = -l; m <= l; ++m) {
            const efa::EFAComplex kp = efa_weighted_kernel(xp, yp, zp, omega, l, m, wp);
            const efa::EFAComplex km = efa_weighted_kernel(xm, ym, zm, omega, l, m, wm);
            const float da_re = (kp.re - km.re) / (2.0f * h);
            const float da_im = (kp.im - km.im) / (2.0f * h);
            dp += 2.0f * (a[m + l].re * da_re + a[m + l].im * da_im);
          }
          const float scale = g_attention_grad[n1 + d * N] * g_D_real[n1];
          f12[axis] += scale * dp;
        }
        (void)coeff;
#else
        // Analytic derivative of the power spectrum w.r.t. r_vec(n2-n1)
        float dpx, dpy, dpz;
        efa_power_derivative_analytic_train(a, x, y, z, omega, l, coeff, paramb.rc_radial, dpx, dpy, dpz);
        const float scale = g_attention_grad[n1 + d * N] * g_D_real[n1];
        f12[0] += scale * dpx;
        f12[1] += scale * dpy;
        f12[2] += scale * dpz;
#endif
      }
    }
    atomicAdd(&g_fx[n1], f12[0]);
    atomicAdd(&g_fy[n1], f12[1]);
    atomicAdd(&g_fz[n1], f12[2]);
    atomicAdd(&g_fx[n2], -f12[0]);
    atomicAdd(&g_fy[n2], -f12[1]);
    atomicAdd(&g_fz[n2], -f12[2]);
    virial[0] -= x * f12[0];
    virial[1] -= y * f12[1];
    virial[2] -= z * f12[2];
    virial[3] -= x * f12[1];
    virial[4] -= y * f12[2];
    virial[5] -= z * f12[0];
  }
  g_virial[n1] += virial[0];
  g_virial[n1 + N] += virial[1];
  g_virial[n1 + 2 * N] += virial[2];
  g_virial[n1 + 3 * N] += virial[3];
  g_virial[n1 + 4 * N] += virial[4];
  g_virial[n1 + 5 * N] += virial[5];
}

// ============================================================================
// Descriptor kernels (identical to NEP/NEP_Charge; replicated here because the
// upstream versions are file-static).
// ============================================================================
static __global__ void find_descriptors_radial(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_NUM_N] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float x12 = g_x12[index];
      float y12 = g_y12[index];
      float z12 = g_z12[index];
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = paramb.rc_radial;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
    }
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      g_descriptors[n1 + n * N] = q[n];
    }
  }
}

static __global__ void find_descriptors_angular(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors,
  float* g_sum_fxyz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_DIM_ANGULAR] = {0.0f};
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < neighbor_number; ++i1) {
        int index = g_NN_sum[n1] + i1;
        int n2 = g_NL[index];
        float x12 = g_x12[index];
        float y12 = g_y12[index];
        float z12 = g_z12[index];
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = paramb.rc_angular;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112,
             paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
             paramb.n_max_angular + 1, n, s, q);
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
      }
    }
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      for (int l = 0; l < paramb.num_L; ++l) {
        int ln = l * (paramb.n_max_angular + 1) + n;
        g_descriptors[n1 + ((paramb.n_max_radial + 1) + ln) * N] = q[ln];
      }
    }
  }
}

// ============================================================================
// ANN kernel: descriptor -> per-atom attention weight q_n (scalar output)
//
// Uses the same single-hidden-layer topology as NEP, but with a single scalar
// output (q_n) instead of an energy.  The chain rule for back-propagation is
//   dE_EFA / d(descriptor_d) = (dE_EFA / d q_n) * (d q_n / d descriptor_d)
//                            = D_real[n] * q_derivative[d]
// which matches the charge pathway in nep_charge.cu.
// ============================================================================
static __device__ void apply_ann_one_layer_efa(
  const int N_des,
  const int N_neu,
  const float* w0,
  const float* b0,
  const float* w1,
  const float b1,
  const float* q,
  float& qn,
  float* qn_derivative)
{
  qn = -b1;
  for (int n = 0; n < N_neu; ++n) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += w0[n * N_des + d] * q[d];
    }
    float x1 = tanh(w0_times_q - b0[n]);
    float tanh_der = 1.0f - x1 * x1;
    qn += w1[n] * x1;
    for (int d = 0; d < N_des; ++d) {
      qn_derivative[d] += w1[n] * tanh_der * w0[n * N_des + d];
    }
  }
}

static __global__ void apply_ann_efa(
  const int N,
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  float* g_attention_weight,
  float* g_attention_grad)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int type = g_type[n1];
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    float qn = 0.0f;
    float qn_der[MAX_DIM] = {0.0f};
    apply_ann_one_layer_efa(
      annmb.dim, annmb.num_neurons1, annmb.w0[type], annmb.b0[type],
      annmb.w1[type], annmb.b1[0], q, qn, qn_der);
    g_attention_weight[n1] = qn;
    for (int d = 0; d < annmb.dim; ++d) {
      g_attention_grad[n1 + d * N] = qn_der[d] * g_q_scaler[d];
    }
  }
}

// Enforce per-structure charge neutrality (sum q_n = 0) by subtracting the
// mean; this matches the zero_total_charge kernel in nep_charge.cu.
static __global__ void zero_total_attention(
  const int Nc,
  const int* Na,
  const int* Na_sum,
  float* g_attention,
  float* g_attention_shifted)
{
  __shared__ float s[1024];
  for (int nc = 0; nc < Nc; ++nc) {
    int N1 = Na_sum[nc];
    int N2 = N1 + Na[nc];
    int number_of_batches = (N2 - N1 - 1) / 1024 + 1;
    float sum = 0.0f;
    for (int batch = 0; batch < number_of_batches; ++batch) {
      int n = threadIdx.x + batch * 1024 + N1;
      s[threadIdx.x] = 0.0f;
      if (n < N2) {
        s[threadIdx.x] = g_attention[n];
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
          s[threadIdx.x] += s[threadIdx.x + offset];
        }
        __syncthreads();
      }
      sum += s[0];
      __syncthreads();
    }
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
      s_mean = sum / float(N2 - N1);
    }
    __syncthreads();
    for (int batch = 0; batch < number_of_batches; ++batch) {
      int n = threadIdx.x + batch * 1024 + N1;
      if (n < N2) {
        float shifted = g_attention[n] - s_mean;
        g_attention_shifted[n] = shifted;
      }
    }
  }
}

// Subtract the per-structure mean of D_real so the chain rule through the
// neutrality constraint stays consistent (matches zero_mean_D_real_train).
static __global__ void zero_mean_D_real(
  const int Nc,
  const int* Na,
  const int* Na_sum,
  float* g_D_real)
{
  __shared__ float s[1024];
  for (int nc = 0; nc < Nc; ++nc) {
    int N1 = Na_sum[nc];
    int N2 = N1 + Na[nc];
    int number_of_batches = (N2 - N1 - 1) / 1024 + 1;
    float sum = 0.0f;
    for (int batch = 0; batch < number_of_batches; ++batch) {
      int n = threadIdx.x + batch * 1024 + N1;
      s[threadIdx.x] = 0.0f;
      if (n < N2) {
        s[threadIdx.x] = g_D_real[n];
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
          s[threadIdx.x] += s[threadIdx.x + offset];
        }
        __syncthreads();
      }
      sum += s[0];
      __syncthreads();
    }
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
      s_mean = sum / float(N2 - N1);
    }
    __syncthreads();
    for (int batch = 0; batch < number_of_batches; ++batch) {
      int n = threadIdx.x + batch * 1024 + N1;
      if (n < N2) {
        g_D_real[n] -= s_mean;
      }
    }
  }
}

// Placeholder for find_max_min (q_scaler) helper; identical to nep_charge.cu
static __global__ void find_max_min(const int N, const float* g_q, float* g_q_scaler)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_max[1024];
  __shared__ float s_min[1024];
  s_max[tid] = -1000000.0f;
  s_min[tid] = +1000000.0f;
  const int stride = 1024;
  const int number_of_rounds = (N - 1) / stride + 1;
  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = round * stride + tid;
    if (n < N) {
      const int m = n + N * bid;
      float q = g_q[m];
      if (q > s_max[tid]) s_max[tid] = q;
      if (q < s_min[tid]) s_min[tid] = q;
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_max[tid] < s_max[tid + offset]) s_max[tid] = s_max[tid + offset];
      if (s_min[tid] > s_min[tid + offset]) s_min[tid] = s_min[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_q_scaler[bid] = min(g_q_scaler[bid], 1.0f / (s_max[0] - s_min[0]));
  }
}

// ============================================================================
// Ewald reciprocal-space kernels (mirrors nep_charge.cu but uses the EFA
// per-atom attention weight as the structure-factor weight).  The kernel
// weight G carries the Ewald Gaussian screening exp(-k^2/(4 alpha^2)) / k^2.
// ============================================================================

static __device__ void cross_product(const float a[3], const float b[3], float c[3])
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

static __device__ float get_area(const float* a, const float* b)
{
  const float s1 = a[1] * b[2] - a[2] * b[1];
  const float s2 = a[2] * b[0] - a[0] * b[2];
  const float s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

static __global__ void find_k_and_G(
  const int Nc,
  const int num_kpoints_max,
  const float alpha,
  const float alpha_factor,
  const float k_max,
  const float lambda_e,
  const float* g_box,
  int* g_num_kpoints,
  float* g_kx,
  float* g_ky,
  float* g_kz,
  float* g_G)
{
  int nc = blockIdx.x * blockDim.x + threadIdx.x;
  if (nc < Nc) {
    const float* box = g_box + 9 * nc;
    float a1[3] = {0.0f}, a2[3] = {0.0f}, a3[3] = {0.0f};
    float det = box[0] * (box[4] * box[8] - box[5] * box[7]) +
                box[1] * (box[5] * box[6] - box[3] * box[8]) +
                box[2] * (box[3] * box[7] - box[4] * box[6]);
    a1[0] = box[0]; a1[1] = box[3]; a1[2] = box[6];
    a2[0] = box[1]; a2[1] = box[4]; a2[2] = box[7];
    a3[0] = box[2]; a3[1] = box[5]; a3[2] = box[8];
    float b1[3], b2[3], b3[3];
    cross_product(a2, a3, b1);
    cross_product(a3, a1, b2);
    cross_product(a1, a2, b3);
    const float two_pi = 6.2831853f;
    const float two_pi_over_det = two_pi / det;
    for (int d = 0; d < 3; ++d) {
      b1[d] *= two_pi_over_det;
      b2[d] *= two_pi_over_det;
      b3[d] *= two_pi_over_det;
    }
    const float volume_k = two_pi * two_pi * two_pi / fabsf(det);
    int n1_max = int(k_max * get_area(b2, b3) / volume_k);
    int n2_max = int(k_max * get_area(b3, b1) / volume_k);
    int n3_max = int(k_max * get_area(b1, b2) / volume_k);
    float ksq_max = k_max * k_max;
    int count = 0;
    for (int n1 = 0; n1 <= n1_max; ++n1) {
      for (int n2 = -n2_max; n2 <= n2_max; ++n2) {
        for (int n3 = -n3_max; n3 <= n3_max; ++n3) {
          const int nsq = n1 * n1 + n2 * n2 + n3 * n3;
          if (nsq == 0 || (n1 == 0 && n2 < 0) || (n1 == 0 && n2 == 0 && n3 < 0))
            continue;
          const float kx = n1 * b1[0] + n2 * b2[0] + n3 * b3[0];
          const float ky = n1 * b1[1] + n2 * b2[1] + n3 * b3[1];
          const float kz = n1 * b1[2] + n2 * b2[2] + n3 * b3[2];
          const float ksq = kx * kx + ky * ky + kz * kz;
          if (ksq < ksq_max) {
            if (count >= num_kpoints_max) {
              g_num_kpoints[nc] = -1;
              return;
            }
            int idx = nc * num_kpoints_max + count;
            g_kx[idx] = kx;
            g_ky[idx] = ky;
            g_kz[idx] = kz;
            float G = fabsf(two_pi_over_det) / ksq * expf(-ksq * alpha_factor);
            g_G[idx] = 2.0f * G * lambda_e;
            ++count;
          }
        }
      }
    }
    g_num_kpoints[nc] = count;
  }
}

static __global__ void find_structure_factor(
  const int num_kpoints_max,
  const int* Na,
  const int* Na_sum,
  const float* g_attention,
  const float* g_x,
  const float* g_y,
  const float* g_z,
  const int* g_num_kpoints,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  float* g_S_real,
  float* g_S_imag)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int num_kpoints = g_num_kpoints[blockIdx.x];
  int number_of_batches = (num_kpoints - 1) / 1024 + 1;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int nk = threadIdx.x + batch * 1024;
    if (nk < num_kpoints) {
      int nc_nk = blockIdx.x * num_kpoints_max + nk;
      float S_real = 0.0f;
      float S_imag = 0.0f;
      for (int n = N1; n < N2; ++n) {
        float kr = g_kx[nc_nk] * g_x[n] + g_ky[nc_nk] * g_y[n] + g_kz[nc_nk] * g_z[n];
        const float qn = g_attention[n];
        float sin_kr = sinf(kr);
        float cos_kr = cosf(kr);
        S_real += qn * cos_kr;
        S_imag -= qn * sin_kr;
      }
      g_S_real[nc_nk] = S_real;
      g_S_imag[nc_nk] = S_imag;
    }
  }
}

// EFA scaling coefficient replaces K_C_SP from the charge model.  Because the
// EFA energy is a learnable attention score rather than a Coulomb energy, we
// factor of 1.0 here; lambda_e is folded into G(k) when the mesh is built.
#define EFA_KC 1.0f

static __global__ void find_force_attention_reciprocal_space(
  const int N,
  const int num_kpoints_max,
  const float alpha_factor,
  const int* Na,
  const int* Na_sum,
  const float* g_attention,
  const float* g_x,
  const float* g_y,
  const float* g_z,
  const int* g_num_kpoints,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const float* g_S_real,
  const float* g_S_imag,
  float* g_D_real,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  float* g_pe)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int number_of_batches = (N2 - N1 - 1) / 1024 + 1;
  int num_kpoints = g_num_kpoints[blockIdx.x];
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = threadIdx.x + batch * 1024 + N1;
    if (n < N2) {
      float temp_energy_sum = 0.0f;
      float temp_virial_sum[6] = {0.0f};
      float temp_force_sum[3] = {0.0f};
      float temp_D_real_sum = 0.0f;
      for (int nk = 0; nk < num_kpoints; ++nk) {
        const int nc_nk = blockIdx.x * num_kpoints_max + nk;
        const float kx = g_kx[nc_nk];
        const float ky = g_ky[nc_nk];
        const float kz = g_kz[nc_nk];
        const float kr = kx * g_x[n] + ky * g_y[n] + kz * g_z[n];
        const float G = g_G[nc_nk];
        const float S_real = g_S_real[nc_nk];
        const float S_imag = g_S_imag[nc_nk];
        float sin_kr = sinf(kr);
        float cos_kr = cosf(kr);
        const float imag_term = G * (S_real * sin_kr + S_imag * cos_kr);
        const float GSS = G * (S_real * S_real + S_imag * S_imag);
        temp_energy_sum += GSS;
        const float alpha_k_factor = 2.0f * alpha_factor + 2.0f / (kx * kx + ky * ky + kz * kz);
        temp_virial_sum[0] += GSS * (1.0f - alpha_k_factor * kx * kx);
        temp_virial_sum[1] += GSS * (1.0f - alpha_k_factor * ky * ky);
        temp_virial_sum[2] += GSS * (1.0f - alpha_k_factor * kz * kz);
        temp_virial_sum[3] -= GSS * (alpha_k_factor * kx * ky);
        temp_virial_sum[4] -= GSS * (alpha_k_factor * ky * kz);
        temp_virial_sum[5] -= GSS * (alpha_k_factor * kz * kx);
        temp_D_real_sum += G * (S_real * cos_kr - S_imag * sin_kr);
        temp_force_sum[0] += kx * imag_term;
        temp_force_sum[1] += ky * imag_term;
        temp_force_sum[2] += kz * imag_term;
      }
      g_pe[n] += EFA_KC * temp_energy_sum / float(N2 - N1);
      for (int d = 0; d < 6; ++d) {
        g_virial[n + N * d] += EFA_KC * temp_virial_sum[d] / float(N2 - N1);
      }
      g_D_real[n] = 2.0f * EFA_KC * temp_D_real_sum;
      const float qn = g_attention[n];
      const float qn_factor = EFA_KC * 2.0f * qn;
      g_fx[n] += qn_factor * temp_force_sum[0];
      g_fy[n] += qn_factor * temp_force_sum[1];
      g_fz[n] += qn_factor * temp_force_sum[2];
    }
  }
}

// ============================================================================
// Back-propagation kernels: chain rule from dE_EFA/dq_n (D_real) into the
// descriptor gradient Fp, then into atomic forces.  These mirror the
// find_force_radial / find_force_angular kernels in nep_charge.cu.
//
// The effective descriptor gradient is
//   Fp_eff[n,d] = attention_grad[n,d] * D_real[n]
// because E_EFA depends on descriptor d only through q_n, and
//   dE_EFA / d(descriptor_d) = (dE_EFA / d q_n) * (d q_n / d descriptor_d)
//                            = D_real[n] * attention_grad[n,d]
// which is exactly the charge pathway (g_Fp + g_charge_derivative * g_D_real).
// ============================================================================
static __global__ void find_force_radial_efa(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_attention_grad,
  const float* g_D_real,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int neighbor_number = g_NN[n1];
    float s_virial_xx = 0.0f, s_virial_yy = 0.0f, s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f, s_virial_yz = 0.0f, s_virial_zx = 0.0f;
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float rc = paramb.rc_radial;
      float rcinv = 1.0f / rc;
      float fc12, fcp12;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        // EFA has no direct NEP energy, so Fp = 0 and the chain rule reduces
        // to attention_grad * D_real (the charge-derivative * D_real term).
        float tmp12 = g_attention_grad[n1 + n * N] * g_D_real[n1];
        tmp12 *= gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);
      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
    }
    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
  }
}

static __global__ void find_force_angular_efa(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_attention_grad,
  const float* g_D_real,
  const float* g_sum_fxyz,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    float s_virial_xx = 0.0f, s_virial_yy = 0.0f, s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f, s_virial_yz = 0.0f, s_virial_zx = 0.0f;
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      // EFA chain rule: Fp_eff = attention_grad * D_real (no direct NEP Fp)
      Fp[d] = g_attention_grad[(paramb.n_max_radial + 1 + d) * N + n1] * g_D_real[n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }
    int neighbor_number = g_NN[n1];
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = paramb.rc_angular;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float f12[3] = {0.0f};
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112,
          paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
          paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12,
          Fp, sum_fxyz, f12);
      }
      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);
      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
    }
    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
  }
}

// MARKER_EFA_KERNELS

// ============================================================================
// Constructor
// ============================================================================
NEP_EFA::NEP_EFA(
  Parameters& para,
  int N,
  int Nc,
  int version,
  int deviceCount)
  : nep(para, N, version, deviceCount)  // E_NEP: standard NEP energy/forces/ZBL
{
  // EFA-side parameter metadata (scalar cutoff style, like NEP_Charge)
  paramb.version = version;
  paramb.rc_radial = para.rc_radial[0];
  paramb.rc_angular = para.rc_angular[0];
  paramb.use_typewise_cutoff_zbl = para.use_typewise_cutoff_zbl;
  paramb.typewise_cutoff_zbl_factor = para.typewise_cutoff_zbl_factor;
  paramb.num_types = para.num_types;
  paramb.n_max_radial = para.n_max_radial;
  paramb.n_max_angular = para.n_max_angular;
  paramb.L_max = para.L_max;
  paramb.has_q_222 = para.has_q_222;
  paramb.has_q_1111 = para.has_q_1111;
  paramb.has_q_112 = para.has_q_112;
  paramb.has_q_123 = para.has_q_123;
  paramb.has_q_233 = para.has_q_233;
  paramb.has_q_134 = para.has_q_134;
  paramb.num_L = paramb.L_max;
  if (para.has_q_222) paramb.num_L += 1;
  if (para.has_q_1111) paramb.num_L += 1;
  if (para.has_q_112) paramb.num_L += 1;
  if (para.has_q_123) paramb.num_L += 1;
  if (para.has_q_233) paramb.num_L += 1;
  if (para.has_q_134) paramb.num_L += 1;
  paramb.dim_angular = (para.n_max_angular + 1) * paramb.num_L;
  paramb.basis_size_radial = para.basis_size_radial;
  paramb.basis_size_angular = para.basis_size_angular;
  paramb.num_types_sq = para.num_types * para.num_types;
  paramb.num_c_radial =
    paramb.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);

  // EFA-specific parameters
  paramb.efa_l_max = para.efa_l_max;
  paramb.efa_num_radial = para.efa_num_radial;
  paramb.efa_dim = para.efa_dim;
  paramb.efa_omega_max = para.efa_omega_max;
  paramb.efa_num_kpoints_max = efa_para.num_kpoints_max;

  // These values are serialized in nep.txt and must match the MD-side parser.
  efa_para.alpha = para.efa_alpha;
  efa_para.alpha_factor = 0.25f / (efa_para.alpha * efa_para.alpha);
  efa_para.omega_max = para.efa_omega_max;
  efa_para.lambda_e = para.efa_lambda_e;

  for (int device_id = 0; device_id < deviceCount; device_id++) {
    gpuSetDevice(device_id);
    annmb[device_id].dim = para.efa_dim;
    annmb[device_id].num_neurons1 = para.num_neurons1;
    annmb[device_id].num_para = para.number_of_variables_efa;

    nep_data[device_id].descriptors.resize(N * annmb[device_id].dim);
    nep_data[device_id].attention_weight.resize(N);
    nep_data[device_id].attention_grad.resize(N * annmb[device_id].dim);
    nep_data[device_id].sum_fxyz.resize(1);
    nep_data[device_id].parameters.resize(annmb[device_id].num_para);
    nep_data[device_id].kx.resize(Nc * efa_para.num_kpoints_max);
    nep_data[device_id].ky.resize(Nc * efa_para.num_kpoints_max);
    nep_data[device_id].kz.resize(Nc * efa_para.num_kpoints_max);
    nep_data[device_id].G.resize(Nc * efa_para.num_kpoints_max);
    nep_data[device_id].S_real.resize(Nc * efa_para.num_kpoints_max);
    nep_data[device_id].S_imag.resize(Nc * efa_para.num_kpoints_max);
    nep_data[device_id].D_real.resize(N);
    nep_data[device_id].cached_A_real.resize(N * annmb[device_id].dim * 9);
    nep_data[device_id].cached_A_imag.resize(N * annmb[device_id].dim * 9);
    nep_data[device_id].num_kpoints.resize(Nc);
  }
}

void NEP_EFA::update_potential(float* parameters, ANN& ann)
{
  // Single-output ANN: descriptor -> scalar q_n
  // Layout: [per-type (w0, b0, w1)] [shared b1] [descriptor coefficients c]
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1; // single output
  }
  ann.b1 = pointer;
  pointer += 1;
  ann.c = pointer;
}

// ============================================================================
// find_force: E_NEP (via NEP member) + E_EFA (this class)
// ============================================================================
// ============================================================================
void NEP_EFA::find_force(
  Parameters& para,
  const float* parameters,
  std::vector<Dataset>& dataset,
  bool calculate_q_scaler,
  int device_in_this_iter)
{
  // Step 1: compute E_NEP (energy, forces, virial, ZBL) via the NEP member.
  // This zeros the per-atom force/virial buffers and accumulates NEP's
  // contribution.  It also computes NEP's q_scaler when calculate_q_scaler.
  nep.find_force(para, parameters, dataset, calculate_q_scaler, device_in_this_iter);

  // Step 2: EFA pass.  Adds E_EFA (and its forces/virial) on top of NEP's.
  // The EFA parameters are the trailing slice of each device's parameter block:
  //   [ NEP params | EFA params ]
  //   <--- number_of_variables - number_of_variables_efa --->
  //                               <--- number_of_variables_efa --->
  const int efa_offset = para.number_of_variables - para.number_of_variables_efa;

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    nep_data[device_id].parameters.copy_from_host(
      parameters + device_id * para.number_of_variables + efa_offset);
    update_potential(nep_data[device_id].parameters.data(), annmb[device_id]);
  }

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    const int block_size = 32;
    const int grid_size = (dataset[device_id].N - 1) / block_size + 1;

    // 1. ERoPE channels -> rotationally invariant power spectrum.
    find_descriptors_erope_train<<<grid_size, block_size>>>(
      dataset[device_id].N,
      dataset[device_id].NN_radial_sum.data(),
      dataset[device_id].NN_radial.data(),
      dataset[device_id].NL_radial.data(),
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      dataset[device_id].x12_radial.data(),
      dataset[device_id].y12_radial.data(),
      dataset[device_id].z12_radial.data(),
      nep_data[device_id].descriptors.data(),
      nep_data[device_id].cached_A_real.data(),
      nep_data[device_id].cached_A_imag.data());
    GPU_CHECK_KERNEL

    if (calculate_q_scaler) {
      find_max_min<<<annmb[device_id].dim, 1024>>>(
        dataset[device_id].N,
        nep_data[device_id].descriptors.data(),
        para.q_scaler_efa_gpu[device_id].data());
      GPU_CHECK_KERNEL
    }

    // 2. ANN: descriptor -> per-atom attention weight q_n
    // (NO zero_force here — NEP already zeroed and accumulated.)
    apply_ann_efa<<<grid_size, block_size>>>(
      dataset[device_id].N,
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      nep_data[device_id].descriptors.data(),
      para.q_scaler_efa_gpu[device_id].data(),
      nep_data[device_id].attention_weight.data(),
      nep_data[device_id].attention_grad.data());
    GPU_CHECK_KERNEL

    // 3. enforce per-structure neutrality (sum q_n = 0)
    zero_total_attention<<<dataset[device_id].Nc, 1024>>>(
      dataset[device_id].Nc,
      dataset[device_id].Na.data(),
      dataset[device_id].Na_sum.data(),
      nep_data[device_id].attention_weight.data(),
      dataset[device_id].efa_attention_shifted.data());
    GPU_CHECK_KERNEL

    // 4. reciprocal space: q_n -> E_EFA, forces, virial, D_real
    find_k_and_G<<<(dataset[device_id].Nc - 1) / 64 + 1, 64>>>(
      dataset[device_id].Nc,
      efa_para.num_kpoints_max,
      efa_para.alpha,
      efa_para.alpha_factor,
      efa_para.omega_max,
      efa_para.lambda_e,
      dataset[device_id].box_original.data(),
      nep_data[device_id].num_kpoints.data(),
      nep_data[device_id].kx.data(),
      nep_data[device_id].ky.data(),
      nep_data[device_id].kz.data(),
      nep_data[device_id].G.data());
    GPU_CHECK_KERNEL

    std::vector<int> num_kpoints_cpu(dataset[device_id].Nc);
    nep_data[device_id].num_kpoints.copy_to_host(num_kpoints_cpu.data());
    for (int nc = 0; nc < dataset[device_id].Nc; ++nc) {
      if (num_kpoints_cpu[nc] < 0) {
        PRINT_INPUT_ERROR(
          "EFA reciprocal-space mesh exceeds efa_num_kpoints_max (50000). "
          "Increase the box size or reduce efa_omega_max.\n");
      }
    }

    find_structure_factor<<<dataset[device_id].Nc, 1024>>>(
      efa_para.num_kpoints_max,
      dataset[device_id].Na.data(),
      dataset[device_id].Na_sum.data(),
      dataset[device_id].efa_attention_shifted.data(),
      dataset[device_id].r.data(),
      dataset[device_id].r.data() + dataset[device_id].N,
      dataset[device_id].r.data() + dataset[device_id].N * 2,
      nep_data[device_id].num_kpoints.data(),
      nep_data[device_id].kx.data(),
      nep_data[device_id].ky.data(),
      nep_data[device_id].kz.data(),
      nep_data[device_id].S_real.data(),
      nep_data[device_id].S_imag.data());
    GPU_CHECK_KERNEL

    find_force_attention_reciprocal_space<<<dataset[device_id].Nc, 1024>>>(
      dataset[device_id].N,
      efa_para.num_kpoints_max,
      efa_para.alpha_factor,
      dataset[device_id].Na.data(),
      dataset[device_id].Na_sum.data(),
      dataset[device_id].efa_attention_shifted.data(),
      dataset[device_id].r.data(),
      dataset[device_id].r.data() + dataset[device_id].N,
      dataset[device_id].r.data() + dataset[device_id].N * 2,
      nep_data[device_id].num_kpoints.data(),
      nep_data[device_id].kx.data(),
      nep_data[device_id].ky.data(),
      nep_data[device_id].kz.data(),
      nep_data[device_id].G.data(),
      nep_data[device_id].S_real.data(),
      nep_data[device_id].S_imag.data(),
      nep_data[device_id].D_real.data(),
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data(),
      dataset[device_id].energy.data());
    GPU_CHECK_KERNEL

    // 5. back-prop: D_real through ANN into descriptor forces
    zero_mean_D_real<<<dataset[device_id].Nc, 1024>>>(
      dataset[device_id].Nc,
      dataset[device_id].Na.data(),
      dataset[device_id].Na_sum.data(),
      nep_data[device_id].D_real.data());
    GPU_CHECK_KERNEL

    find_force_erope_train<<<grid_size, block_size>>>(
      dataset[device_id].N,
      dataset[device_id].NN_radial_sum.data(),
      dataset[device_id].NN_radial.data(),
      dataset[device_id].NL_radial.data(),
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      dataset[device_id].x12_radial.data(),
      dataset[device_id].y12_radial.data(),
      dataset[device_id].z12_radial.data(),
      nep_data[device_id].attention_grad.data(),
      nep_data[device_id].D_real.data(),
      nep_data[device_id].cached_A_real.data(),
      nep_data[device_id].cached_A_imag.data(),
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data());
    GPU_CHECK_KERNEL
  }
}
