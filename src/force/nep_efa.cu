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
NEP_EFA (MD side) implementation.

E_total = E_NEP + E_EFA, where E_NEP is computed by the composed NEP member and
E_EFA is a global, linear-complexity attention energy computed in reciprocal
(k-) space.  The forward pass is:

  1. NEP::compute()                              -> E_NEP, forces, virial
  2. find_descriptors_erope_{large,small}_box    -> power-spectrum features
  3. zero_total_attention_single                 -> enforce sum q_n = 0
  4. find_k_and_G_single                         -> reciprocal-space mesh
  5. find_structure_factor_single                -> S(k) = sum_n q_n exp(-i k.r_n)
  6. find_force_attention_reciprocal_space_single-> E_EFA, forces, virial, D_real
  7. zero_mean_D_real_single                     -> chain-rule correction
  8. find_force_erope_{large,small}_box          -> back-prop -> forces

Steps 2 and 8/9 use the neighbor lists built by NEP::compute() (either
nep.nep_data for large boxes or nep.small_box_data for small boxes).
------------------------------------------------------------------------------*/

#include "neighbor.cuh"
#include "nep_efa.cuh"
#include "utilities/common.cuh"
#include "utilities/efa_utilities.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

#define EFA_KC 1.0f

// ============================================================================
// ANN helper: single-hidden-layer, single-output (q_n) forward pass with
// derivative accumulation for back-propagation.
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

static __device__ __forceinline__ float efa_cutoff_md(float r, float rc)
{
  float fc = 0.0f;
  find_fc(rc, 1.0f / rc, r, fc);
  return fc;
}

static __device__ __forceinline__ efa::EFAComplex efa_weighted_kernel_md(
  float x, float y, float z, float omega, int l, int m, float weight)
{
  return efa::efa_cscale(efa::efa_erope_value(x, y, z, omega, l, m), weight);
}

static __global__ void find_descriptors_erope_large_box(
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  float* g_descriptors,
  float* g_cached_A_real,
  float* g_cached_A_imag)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2)
    return;
  const int t1 = g_type[n1];
  for (int l = 0; l < paramb.efa_l_max; ++l) {
    for (int ir = 0; ir < paramb.efa_num_radial; ++ir) {
      efa::EFAComplex a[9];
      for (int m = 0; m < 9; ++m) a[m] = efa::EFAComplex();
      const float omega = paramb.efa_omega_max * float(ir + 1) /
                          float(paramb.efa_num_radial);
      for (int i = 0; i < g_NN[n1]; ++i) {
        const int index = i * N + n1;
        const int n2 = g_NL[index];
        float x = float(g_x[n2] - g_x[n1]);
        float y = float(g_y[n2] - g_y[n1]);
        float z = float(g_z[n2] - g_z[n1]);
        apply_mic(box, x, y, z);
        const float r = sqrtf(x * x + y * y + z * z);
        const int t2 = g_type[n2];
        const float c = annmb.c[(l * paramb.efa_num_radial + ir) * paramb.num_types_sq +
                                 t1 * paramb.num_types + t2];
        const float weight = c * efa_cutoff_md(r, paramb.rc_radial);
        for (int m = -l; m <= l; ++m) {
          const efa::EFAComplex k = efa_weighted_kernel_md(x, y, z, omega, l, m, weight);
          a[m + l].re += k.re;
          a[m + l].im += k.im;
        }
      }
      float power = 0.0f;
      for (int m = -l; m <= l; ++m) power += efa::efa_cabs2(a[m + l]);
      g_descriptors[n1 + (l * paramb.efa_num_radial + ir) * N] = power;
      // Cache A(l,m) for reuse in force kernel
      const int d = l * paramb.efa_num_radial + ir;
      const int cache_base = n1 + d * 9 * N;
      for (int m = 0; m < 9; ++m) {
        g_cached_A_real[cache_base + m * N] = a[m].re;
        g_cached_A_imag[cache_base + m * N] = a[m].im;
      }
    }
  }
}

static __global__ void find_descriptors_erope_small_box(
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors,
  float* g_cached_A_real,
  float* g_cached_A_imag)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2)
    return;
  const int t1 = g_type[n1];
  for (int l = 0; l < paramb.efa_l_max; ++l) {
    for (int ir = 0; ir < paramb.efa_num_radial; ++ir) {
      efa::EFAComplex a[9];
      for (int m = 0; m < 9; ++m) a[m] = efa::EFAComplex();
      const float omega = paramb.efa_omega_max * float(ir + 1) /
                          float(paramb.efa_num_radial);
      for (int i = 0; i < g_NN[n1]; ++i) {
        const int index = i * N + n1;
        const int n2 = g_NL[index];
        const float x = g_x12[index], y = g_y12[index], z = g_z12[index];
        const float r = sqrtf(x * x + y * y + z * z);
        const int t2 = g_type[n2];
        const float c = annmb.c[(l * paramb.efa_num_radial + ir) * paramb.num_types_sq +
                                 t1 * paramb.num_types + t2];
        const float weight = c * efa_cutoff_md(r, paramb.rc_radial);
        for (int m = -l; m <= l; ++m) {
          const efa::EFAComplex k = efa_weighted_kernel_md(x, y, z, omega, l, m, weight);
          a[m + l].re += k.re;
          a[m + l].im += k.im;
        }
      }
      const int d = l * paramb.efa_num_radial + ir;
      float power = 0.0f;
      for (int m = -l; m <= l; ++m) power += efa::efa_cabs2(a[m + l]);
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

static __global__ void apply_ann_efa_descriptor(
  const int N,
  const NEP_EFA::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  float* g_attention,
  float* g_attention_grad)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) return;
  const int type = g_type[n1];
  float q[MAX_DIM] = {0.0f};
  float dq[MAX_DIM] = {0.0f};
  for (int d = 0; d < annmb.dim; ++d) q[d] = g_descriptors[n1 + d * N] * annmb.q_scaler_efa[d];
  float qn = 0.0f;
  apply_ann_one_layer_efa(annmb.dim, annmb.num_neurons1, annmb.w0[type], annmb.b0[type],
                           annmb.w1[type], annmb.b1[0], q, qn, dq);
  g_attention[n1] = qn;
  for (int d = 0; d < annmb.dim; ++d)
    g_attention_grad[n1 + d * N] = dq[d] * annmb.q_scaler_efa[d];
}

#ifdef EFA_USE_FINITE_DIFF
static __device__ __forceinline__ float efa_power_derivative_md(
  const efa::EFAComplex* a, float xp, float yp, float zp, float xm, float ym, float zm,
  float omega, int l, float coeff, float rc, int axis)
{
  const float h = 1.0e-3f;
  const float rp = sqrtf(xp * xp + yp * yp + zp * zp);
  const float rm = sqrtf(xm * xm + ym * ym + zm * zm);
  const float wp = coeff * efa_cutoff_md(rp, rc);
  const float wm = coeff * efa_cutoff_md(rm, rc);
  float dp = 0.0f;
  for (int m = -l; m <= l; ++m) {
    const efa::EFAComplex kp = efa_weighted_kernel_md(xp, yp, zp, omega, l, m, wp);
    const efa::EFAComplex km = efa_weighted_kernel_md(xm, ym, zm, omega, l, m, wm);
    dp += 2.0f * (a[m + l].re * (kp.re - km.re) + a[m + l].im * (kp.im - km.im)) /
          (2.0f * h);
  }
  (void)axis;
  return dp;
}
#endif // EFA_USE_FINITE_DIFF

// Analytic derivative of the power spectrum w.r.t. the pair displacement r_vec.
// Computes d(|a_lm|^2)/d(r_vec) = 2 * Re[ conj(a_lm) * d(a_lm)/d(r_vec) ]
// where d(a_lm)/d(r_vec) = coeff * [ f_cut'(r) * r_hat * K_lm + f_cut(r) * dK_lm/d(r_vec) ]
// K_lm and dK_lm are computed analytically by efa_erope_derivative.
static __device__ __forceinline__ void efa_power_derivative_analytic(
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

static __global__ void find_force_erope_large_box(
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_attention_grad,
  const float* __restrict__ g_D_real,
  const float* __restrict__ g_cached_A_real,
  const float* __restrict__ g_cached_A_imag,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;
  const int t1 = g_type[n1];
  double virial[9] = {0.0};
  for (int i = 0; i < g_NN[n1]; ++i) {
    const int index = i * N + n1;
    const int n2 = g_NL[index];
    float x = float(g_x[n2] - g_x[n1]);
    float y = float(g_y[n2] - g_y[n1]);
    float z = float(g_z[n2] - g_z[n1]);
    apply_mic(box, x, y, z);
    const int t2 = g_type[n2];
    float f12[3] = {0.0f, 0.0f, 0.0f};
    for (int l = 0; l < paramb.efa_l_max; ++l) {
      for (int ir = 0; ir < paramb.efa_num_radial; ++ir) {
        const int d = l * paramb.efa_num_radial + ir;
        const float omega = paramb.efa_omega_max * float(ir + 1) /
                            float(paramb.efa_num_radial);
        // Read cached A(l,m) from descriptor pass (avoids O(N*Nz) recompute)
        efa::EFAComplex a[9];
        const int cache_base = n1 + d * 9 * N;
        for (int m = 0; m < 9; ++m) {
          a[m] = efa::EFAComplex(g_cached_A_real[cache_base + m * N],
                                 g_cached_A_imag[cache_base + m * N]);
        }
        const float c = annmb.c[d * paramb.num_types_sq + t1 * paramb.num_types + t2];
#ifdef EFA_USE_FINITE_DIFF
        float xp = x, yp = y, zp = z, xm = x, ym = y, zm = z;
        const float h = 1.0e-3f;
        xp += h; xm -= h;
        f12[0] += g_attention_grad[n1 + d * N] * g_D_real[n1] *
                  efa_power_derivative_md(a, xp, yp, zp, xm, ym, zm, omega, l, c, paramb.rc_radial, 0);
        xp = x; xm = x; yp += h; ym -= h;
        f12[1] += g_attention_grad[n1 + d * N] * g_D_real[n1] *
                  efa_power_derivative_md(a, xp, yp, zp, xm, ym, zm, omega, l, c, paramb.rc_radial, 1);
        yp = y; ym = y; zp += h; zm -= h;
        f12[2] += g_attention_grad[n1 + d * N] * g_D_real[n1] *
                  efa_power_derivative_md(a, xp, yp, zp, xm, ym, zm, omega, l, c, paramb.rc_radial, 2);
#else
        // Analytic derivative of the power spectrum w.r.t. r_vec(n2-n1)
        float dpx, dpy, dpz;
        efa_power_derivative_analytic(a, x, y, z, omega, l, c, paramb.rc_radial, dpx, dpy, dpz);
        f12[0] += g_attention_grad[n1 + d * N] * g_D_real[n1] * dpx;
        f12[1] += g_attention_grad[n1 + d * N] * g_D_real[n1] * dpy;
        f12[2] += g_attention_grad[n1 + d * N] * g_D_real[n1] * dpz;
#endif
      }
    }
    g_fx[n1] += f12[0]; g_fy[n1] += f12[1]; g_fz[n1] += f12[2];
    atomicAdd(&g_fx[n2], double(-f12[0]));
    atomicAdd(&g_fy[n2], double(-f12[1]));
    atomicAdd(&g_fz[n2], double(-f12[2]));
    virial[0] -= x * f12[0]; virial[1] -= y * f12[1]; virial[2] -= z * f12[2];
    virial[3] -= x * f12[1]; virial[4] -= x * f12[2]; virial[5] -= y * f12[2];
    virial[6] -= y * f12[0]; virial[7] -= z * f12[0]; virial[8] -= z * f12[1];
  }
  for (int d = 0; d < 9; ++d) g_virial[n1 + d * N] += virial[d];
}

static __global__ void find_force_erope_small_box(
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_attention_grad,
  const float* __restrict__ g_D_real,
  const float* __restrict__ g_cached_A_real,
  const float* __restrict__ g_cached_A_imag,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;
  const int t1 = g_type[n1];
  double virial[9] = {0.0};
  for (int i = 0; i < g_NN[n1]; ++i) {
    const int index = i * N + n1;
    const int n2 = g_NL[index];
    const float x = g_x12[index], y = g_y12[index], z = g_z12[index];
    const int t2 = g_type[n2];
    float f12[3] = {0.0f, 0.0f, 0.0f};
    for (int l = 0; l < paramb.efa_l_max; ++l) {
      for (int ir = 0; ir < paramb.efa_num_radial; ++ir) {
        const int d = l * paramb.efa_num_radial + ir;
        const float omega = paramb.efa_omega_max * float(ir + 1) /
                            float(paramb.efa_num_radial);
        // Read cached A(l,m) from descriptor pass (avoids O(N*Nz) recompute)
        efa::EFAComplex a[9];
        const int cache_base = n1 + d * 9 * N;
        for (int m = 0; m < 9; ++m) {
          a[m] = efa::EFAComplex(g_cached_A_real[cache_base + m * N],
                                 g_cached_A_imag[cache_base + m * N]);
        }
        const float c = annmb.c[d * paramb.num_types_sq + t1 * paramb.num_types + t2];
#ifdef EFA_USE_FINITE_DIFF
        const float h = 1.0e-3f;
        float xp = x + h, xm = x - h;
        f12[0] += g_attention_grad[n1 + d * N] * g_D_real[n1] *
                  efa_power_derivative_md(a, xp, y, z, xm, y, z, omega, l, c, paramb.rc_radial, 0);
        float yp = y + h, ym = y - h;
        f12[1] += g_attention_grad[n1 + d * N] * g_D_real[n1] *
                  efa_power_derivative_md(a, x, yp, z, x, ym, z, omega, l, c, paramb.rc_radial, 1);
        float zp = z + h, zm = z - h;
        f12[2] += g_attention_grad[n1 + d * N] * g_D_real[n1] *
                  efa_power_derivative_md(a, x, y, zp, x, y, zm, omega, l, c, paramb.rc_radial, 2);
#else
        // Analytic derivative of the power spectrum w.r.t. r_vec(n2-n1)
        float dpx, dpy, dpz;
        efa_power_derivative_analytic(a, x, y, z, omega, l, c, paramb.rc_radial, dpx, dpy, dpz);
        f12[0] += g_attention_grad[n1 + d * N] * g_D_real[n1] * dpx;
        f12[1] += g_attention_grad[n1 + d * N] * g_D_real[n1] * dpy;
        f12[2] += g_attention_grad[n1 + d * N] * g_D_real[n1] * dpz;
#endif
      }
    }
    atomicAdd(&g_fx[n1], double(f12[0])); atomicAdd(&g_fy[n1], double(f12[1]));
    atomicAdd(&g_fz[n1], double(f12[2]));
    atomicAdd(&g_fx[n2], double(-f12[0])); atomicAdd(&g_fy[n2], double(-f12[1]));
    atomicAdd(&g_fz[n2], double(-f12[2]));
    virial[0] -= x * f12[0]; virial[1] -= y * f12[1]; virial[2] -= z * f12[2];
    virial[3] -= x * f12[1]; virial[4] -= x * f12[2]; virial[5] -= y * f12[2];
    virial[6] -= y * f12[0]; virial[7] -= z * f12[0]; virial[8] -= z * f12[1];
  }
  for (int d = 0; d < 9; ++d) atomicAdd(&g_virial[n1 + d * N], virial[d]);
}

// ============================================================================
// Combined descriptor + ANN kernel (LARGE BOX).
// Recomputes r12 from absolute (double) positions inside the kernel.
// ============================================================================
static __global__ void find_descriptor_efa_large_box(
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  float* g_attention,
  float* g_attention_grad,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    float q[MAX_DIM] = {0.0f};

    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      float x12 = g_x[n2] - g_x[n1];
      float y12 = g_y[n2] - g_y[n1];
      float z12 = g_z[n2] - g_z[n1];
      apply_mic(box, x12, y12, z12);
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

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        float x12 = g_x[n2] - g_x[n1];
        float y12 = g_y[n2] - g_y[n1];
        float z12 = g_z[n2] - g_z[n1];
        apply_mic(box, x12, y12, z12);
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
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112,
        paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
        paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
      }
    }

    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler_efa[d];
    }

    float qn = 0.0f;
    float qn_der[MAX_DIM] = {0.0f};
    apply_ann_one_layer_efa(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1],
      annmb.w1[t1], annmb.b1[0], q, qn, qn_der);
    g_attention[n1] = qn;
    for (int d = 0; d < annmb.dim; ++d) {
      g_attention_grad[d * N + n1] = qn_der[d] * annmb.q_scaler_efa[d];
    }
  }
}

// ============================================================================
// Enforce sum_n q_n = 0 over the local atom range [N1, N2) by subtracting the
// mean.  Single-structure version (one block, block-reduce).
// ============================================================================
static __global__ void zero_total_attention_single(
  const int N,
  const int N1,
  const int N2,
  float* g_attention)
{
  int tid = threadIdx.x;
  int number_of_batches = (N2 - N1 - 1) / 1024 + 1;
  __shared__ float s_charge[1024];
  float charge = 0.0f;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024 + N1;
    if (n < N2) {
      charge += g_attention[n];
    }
  }
  s_charge[tid] = charge;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_charge[tid] += s_charge[tid + offset];
    }
    __syncthreads();
  }
  float mean_q = s_charge[0] / float(N2 - N1);
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024 + N1;
    if (n < N2) {
      g_attention[n] -= mean_q;
    }
  }
}

// ============================================================================
// Subtract the per-structure mean of D_real (chain-rule correction for the
// neutrality constraint).  Uses double accumulator for precision.
// ============================================================================
static __global__ void zero_mean_D_real_single(
  const int N,
  const int N1,
  const int N2,
  float* g_D_real)
{
  int tid = threadIdx.x;
  int number_of_batches = (N2 - N1 - 1) / 1024 + 1;
  __shared__ double s_sum[1024];
  double sum = 0.0;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024 + N1;
    if (n < N2) {
      sum += (double)g_D_real[n];
    }
  }
  s_sum[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum[tid] += s_sum[tid + offset];
    }
    __syncthreads();
  }
  float mean_D = (float)(s_sum[0] / double(N2 - N1));
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024 + N1;
    if (n < N2) {
      g_D_real[n] -= mean_D;
    }
  }
}

// ============================================================================
// Reciprocal-space structure factor: S(k) = sum_n q_n exp(-i k.r_n).
// Single-structure version, one block per k-point.
// ============================================================================
static __global__ void find_structure_factor_single(
  const int num_kpoints,
  const int N1,
  const int N2,
  const float* g_attention,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  float* g_S_real,
  float* g_S_imag)
{
  int nk = blockIdx.x * blockDim.x + threadIdx.x;
  if (nk < num_kpoints) {
    float S_real = 0.0f;
    float S_imag = 0.0f;
    for (int n = N1; n < N2; ++n) {
      float kr = g_kx[nk] * float(g_x[n]) + g_ky[nk] * float(g_y[n]) + g_kz[nk] * float(g_z[n]);
      const float qn = g_attention[n];
      float sin_kr = sin(kr);
      float cos_kr = cos(kr);
      S_real += qn * cos_kr;
      S_imag -= qn * sin_kr;
    }
    g_S_real[nk] = S_real;
    g_S_imag[nk] = S_imag;
  }
}

// ============================================================================
// EFA reciprocal-space energy, forces, virial, and D_real = dE/dq_n.
//
// E_EFA_total = sum_k G(k) |S(k)|^2  (a global "attention score" energy).
// Per-atom energy    = E_EFA_total / N_atoms.
// Per-atom virial    = virial_total  / N_atoms.
// Force on atom n     = 2 q_n sum_k G k_vec (S_real sin_kr + S_imag cos_kr).
// D_real[n] = dE/dq_n = 2 sum_k G (S_real cos_kr - S_imag sin_kr).
// ============================================================================
static __global__ void find_force_attention_reciprocal_space_single(
  const int N,
  const int N1,
  const int N2,
  const int num_kpoints,
  const float alpha_factor,
  const float* g_attention,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const float* g_S_real,
  const float* g_S_imag,
  float* g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const float qn = g_attention[n];
    float temp_energy_sum = 0.0f;
    float temp_virial_sum[6] = {0.0f};
    float temp_force_sum[3] = {0.0f};
    float temp_D_real_sum = 0.0f;
    for (int nk = 0; nk < num_kpoints; ++nk) {
      const float kx = g_kx[nk];
      const float ky = g_ky[nk];
      const float kz = g_kz[nk];
      const float kr = kx * float(g_x[n]) + ky * float(g_y[n]) + kz * float(g_z[n]);
      const float G = g_G[nk];
      const float S_real = g_S_real[nk];
      const float S_imag = g_S_imag[nk];
      float sin_kr = sin(kr);
      float cos_kr = cos(kr);
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
    const float inv_N = 1.0f / float(N2 - N1);
    g_pe[n] += EFA_KC * temp_energy_sum * inv_N;
    g_virial[n + 0 * N] += EFA_KC * temp_virial_sum[0] * inv_N;
    g_virial[n + 1 * N] += EFA_KC * temp_virial_sum[1] * inv_N;
    g_virial[n + 2 * N] += EFA_KC * temp_virial_sum[2] * inv_N;
    g_virial[n + 3 * N] += EFA_KC * temp_virial_sum[3] * inv_N;
    g_virial[n + 4 * N] += EFA_KC * temp_virial_sum[5] * inv_N;
    g_virial[n + 5 * N] += EFA_KC * temp_virial_sum[4] * inv_N;
    g_virial[n + 6 * N] += EFA_KC * temp_virial_sum[3] * inv_N;
    g_virial[n + 7 * N] += EFA_KC * temp_virial_sum[5] * inv_N;
    g_virial[n + 8 * N] += EFA_KC * temp_virial_sum[4] * inv_N;
    g_D_real[n] = 2.0f * EFA_KC * temp_D_real_sum;
    const float qn_factor = EFA_KC * 2.0f * qn;
    g_fx[n] += qn_factor * temp_force_sum[0];
    g_fy[n] += qn_factor * temp_force_sum[1];
    g_fz[n] += qn_factor * temp_force_sum[2];
  }
}

// ============================================================================
// Back-prop: radial descriptor -> forces (LARGE BOX).
// Recomputes r12 from double positions.  EFA has no direct NEP Fp, so the
// effective descriptor gradient is just attention_grad * D_real.
// ============================================================================
static __global__ void find_force_radial_efa_large_box(
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_attention_grad,
  const float* g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    float s_fx = 0.0f, s_fy = 0.0f, s_fz = 0.0f;
    float s_sxx = 0.0f, s_syy = 0.0f, s_szz = 0.0f;
    float s_sxy = 0.0f, s_sxz = 0.0f, s_syz = 0.0f;
    float s_syx = 0.0f, s_szx = 0.0f, s_szy = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    const float D_real_n1 = g_D_real[n1];
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      int t2 = g_type[n2];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      float rc = paramb.rc_radial;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fnp12, fnp12);
      float f12[3] = {0.0f};
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        float tmp12 = g_attention_grad[n * N + n1] * D_real_n1;
        tmp12 *= gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
      s_fx += f12[0];
      s_fy += f12[1];
      s_fz += f12[2];
      s_sxx -= r12[0] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syz -= r12[1] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      atomicAdd(&g_fx[n2], double(-f12[0]));
      atomicAdd(&g_fy[n2], double(-f12[1]));
      atomicAdd(&g_fz[n2], double(-f12[2]));
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
  }
}

// ============================================================================
// Back-prop: radial descriptor -> forces (SMALL BOX).
// Reads precomputed r12 from small_box_data.
// ============================================================================
static __global__ void find_force_radial_efa_small_box(
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_radial,
  const float* __restrict__ g_y12_radial,
  const float* __restrict__ g_z12_radial,
  const float* __restrict__ g_attention_grad,
  const float* g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    const float D_real_n1 = g_D_real[n1];
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      float rc = paramb.rc_radial;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fnp12, fnp12);
      float f12[3] = {0.0f};
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        float tmp12 = g_attention_grad[n * N + n1] * D_real_n1;
        tmp12 *= gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
      double s_sxx = 0.0, s_syy = 0.0, s_szz = 0.0;
      double s_sxy = 0.0, s_sxz = 0.0, s_syz = 0.0;
      double s_syx = 0.0, s_szx = 0.0, s_szy = 0.0;
      s_sxx -= r12[0] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syz -= r12[1] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      atomicAdd(&g_fx[n1], double(f12[0]));
      atomicAdd(&g_fy[n1], double(f12[1]));
      atomicAdd(&g_fz[n1], double(f12[2]));
      atomicAdd(&g_fx[n2], double(-f12[0]));
      atomicAdd(&g_fy[n2], double(-f12[1]));
      atomicAdd(&g_fz[n2], double(-f12[2]));
      atomicAdd(&g_virial[n2 + 0 * N], s_sxx);
      atomicAdd(&g_virial[n2 + 1 * N], s_syy);
      atomicAdd(&g_virial[n2 + 2 * N], s_szz);
      atomicAdd(&g_virial[n2 + 3 * N], s_sxy);
      atomicAdd(&g_virial[n2 + 4 * N], s_sxz);
      atomicAdd(&g_virial[n2 + 5 * N], s_syz);
      atomicAdd(&g_virial[n2 + 6 * N], s_syx);
      atomicAdd(&g_virial[n2 + 7 * N], s_szx);
      atomicAdd(&g_virial[n2 + 8 * N], s_szy);
    }
  }
}

// ============================================================================
// Back-prop: angular descriptor -> partial forces (LARGE BOX).
// Stores per-pair partial forces f12 for later reduction.
// ============================================================================
static __global__ void find_force_angular_efa_large_box(
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_attention_grad,
  const float* g_D_real,
  const float* __restrict__ g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_attention_grad[(paramb.n_max_radial + 1 + d) * N + n1] * g_D_real[n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
      }
    }
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = paramb.rc_angular;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
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
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

// ============================================================================
// Back-prop: angular descriptor -> partial forces (SMALL BOX).
// ============================================================================
static __global__ void find_force_angular_efa_small_box(
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_angular,
  const float* __restrict__ g_y12_angular,
  const float* __restrict__ g_z12_angular,
  const float* __restrict__ g_attention_grad,
  const float* g_D_real,
  const float* __restrict__ g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_attention_grad[(paramb.n_max_radial + 1 + d) * N + n1] * g_D_real[n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
      }
    }
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      float r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = paramb.rc_angular;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
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
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

// ============================================================================
// Combined descriptor + ANN kernel (SMALL BOX).
// Reads precomputed r12 from nep.small_box_data.r12.
// ============================================================================
static __global__ void find_descriptor_efa_small_box(
  const NEP_EFA::ParaMB paramb,
  const NEP_EFA::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_radial,
  const float* __restrict__ g_y12_radial,
  const float* __restrict__ g_z12_radial,
  const float* __restrict__ g_x12_angular,
  const float* __restrict__ g_y12_angular,
  const float* __restrict__ g_z12_angular,
  float* g_attention,
  float* g_attention_grad,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    float q[MAX_DIM] = {0.0f};

    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      float r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
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

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        float r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
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
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
      }
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112,
        paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
        paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
      }
    }

    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler_efa[d];
    }

    float qn = 0.0f;
    float qn_der[MAX_DIM] = {0.0f};
    apply_ann_one_layer_efa(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1],
      annmb.w1[t1], annmb.b1[0], q, qn, qn_der);
    g_attention[n1] = qn;
    for (int d = 0; d < annmb.dim; ++d) {
      g_attention_grad[d * N + n1] = qn_der[d] * annmb.q_scaler_efa[d];
    }
  }
}

// ============================================================================
// Host-side helper: determine if the simulation box is "small" (any periodic
// direction thinner than 2.5 * rc).  Mirrors get_expanded_box in nep.cu.
// ============================================================================
static bool is_small_box_check(const double rc, const Box& box)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  bool small = false;
  if (box.pbc_x && thickness_x <= 2.5 * (rc + 1.0)) small = true;
  if (box.pbc_y && thickness_y <= 2.5 * (rc + 1.0)) small = true;
  if (box.pbc_z && thickness_z <= 2.5 * (rc + 1.0)) small = true;
  return small;
}

// ============================================================================
// Constructor
// ============================================================================
NEP_EFA::NEP_EFA(const char* file_potential, const int num_atoms) : nep(file_potential, num_atoms)
{
  // Copy relevant hyperparameters from the NEP member (already parsed from nep.txt).
  // Set N1/N2 for the composed NEP member. force.cu only sets these on the
  // outermost Potential (NEP_EFA), so the NEP member needs explicit init.
  nep.N1 = 0;
  nep.N2 = num_atoms;
  paramb.version = nep.paramb.version;
  paramb.rc_radial = nep.paramb.rc_radial[0];
  paramb.rc_angular = nep.paramb.rc_angular[0];
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  paramb.MN_radial = nep.paramb.MN_radial;
  paramb.MN_angular = nep.paramb.MN_angular;
  paramb.n_max_radial = nep.paramb.n_max_radial;
  paramb.n_max_angular = nep.paramb.n_max_angular;
  paramb.L_max = nep.paramb.L_max;
  paramb.has_q_222 = nep.paramb.has_q_222;
  paramb.has_q_1111 = nep.paramb.has_q_1111;
  paramb.has_q_112 = nep.paramb.has_q_112;
  paramb.has_q_123 = nep.paramb.has_q_123;
  paramb.has_q_233 = nep.paramb.has_q_233;
  paramb.has_q_134 = nep.paramb.has_q_134;
  paramb.num_L = nep.paramb.num_L;
  paramb.dim_angular = nep.paramb.dim_angular;
  paramb.basis_size_radial = nep.paramb.basis_size_radial;
  paramb.basis_size_angular = nep.paramb.basis_size_angular;
  paramb.num_types_sq = nep.paramb.num_types_sq;
  paramb.num_c_radial = nep.paramb.num_c_radial;
  paramb.num_types = nep.paramb.num_types;
  paramb.use_typewise_cutoff_zbl = nep.paramb.use_typewise_cutoff_zbl;
  paramb.typewise_cutoff_zbl_factor = nep.paramb.typewise_cutoff_zbl_factor;

  // Ewald screening parameter: same heuristic as NEP_Charge
  efa_para.alpha = float(PI) / paramb.rc_radial;
  efa_para.alpha_factor = 0.25f / (efa_para.alpha * efa_para.alpha);

  // Re-open nep.txt to read the EFA parameter block.
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    std::cout << "NEP_EFA: failed to re-open " << file_potential << std::endl;
    exit(1);
  }

  // Skip header line
  std::vector<std::string> tokens = get_tokens(input);

  // Skip optional ZBL line
  if (tokens[0] == "nep4_zbl_efa") {
    tokens = get_tokens(input);
  }

  // Skip cutoff, n_max, basis_size, l_max, ANN lines (5 lines)
  for (int i = 0; i < 5; ++i) {
    get_tokens(input);
  }

  // Read the efa hyperparameter line
  tokens = get_tokens(input);
  if (tokens.size() != 6 || tokens[0] != "efa") {
    std::cout << "NEP_EFA: expected 'efa l_max num_radial omega_max alpha lambda_e' line." << std::endl;
    exit(1);
  }
  paramb.efa_l_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.efa_num_radial = get_int_from_token(tokens[2], __FILE__, __LINE__);
  if (paramb.efa_l_max < 1 || paramb.efa_l_max > 4 || paramb.efa_num_radial < 1 ||
      paramb.efa_num_radial > 16) {
    std::cout << "NEP_EFA: efa l_max must be in [1, 4] and num_radial in [1, 16]."
              << std::endl;
    exit(1);
  }
  efa_para.omega_max = get_double_from_token(tokens[3], __FILE__, __LINE__);
  efa_para.alpha = get_double_from_token(tokens[4], __FILE__, __LINE__);
  efa_para.lambda_e = get_double_from_token(tokens[5], __FILE__, __LINE__);
  if (efa_para.omega_max <= 0.0f || efa_para.alpha <= 0.0f || efa_para.lambda_e < 0.0f) {
    std::cout << "NEP_EFA: efa omega_max and alpha must be positive, and lambda_e must be non-negative."
              << std::endl;
    exit(1);
  }
  efa_para.alpha_factor = 0.25f / (efa_para.alpha * efa_para.alpha);

  // EFA ANN dimensions are determined by the ERoPE power spectrum, not NEP's
  // invariant descriptor dimension.
  paramb.efa_dim = paramb.efa_l_max * paramb.efa_num_radial;
  paramb.efa_omega_max = efa_para.omega_max;
  const int efa_num_para_ann =
    (paramb.efa_dim + 2) * nep.annmb.num_neurons1 * paramb.num_types + 1;
  const int efa_num_para = efa_num_para_ann +
                           paramb.efa_dim * paramb.num_types_sq;
  annmb.dim = paramb.efa_dim;
  annmb.num_neurons1 = nep.annmb.num_neurons1;
  annmb.num_para = efa_num_para;
  annmb.num_para_ann = efa_num_para_ann;

  // Skip NEP params (annmb.num_para lines)
  for (int n = 0; n < nep.annmb.num_para; ++n) {
    get_tokens(input);
  }

  // Read EFA params
  std::vector<float> efa_params(efa_num_para + paramb.efa_dim);
  for (int n = 0; n < efa_num_para; ++n) {
    tokens = get_tokens(input);
    efa_params[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  // Skip NEP q_scaler (annmb.dim lines)
  for (int n = 0; n < nep.annmb.dim; ++n) {
    get_tokens(input);
  }

  // Read EFA q_scaler
  for (int n = 0; n < paramb.efa_dim; ++n) {
    tokens = get_tokens(input);
    efa_params[efa_num_para + n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  // Copy EFA params to device
  nep_efa_data.parameters.resize(efa_num_para + paramb.efa_dim);
  nep_efa_data.parameters.copy_from_host(efa_params.data());
  update_potential(nep_efa_data.parameters.data(), annmb);
  annmb.q_scaler_efa = nep_efa_data.parameters.data() + efa_num_para;

  // Allocate EFA data buffers
  const int N = num_atoms;
  nep_efa_data.descriptors.resize(N * annmb.dim);
  nep_efa_data.attention.resize(N);
  nep_efa_data.attention_grad.resize(N * annmb.dim);
  nep_efa_data.sum_fxyz.resize(
    N * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
  nep_efa_data.D_real.resize(N);
  // Cached A(l,m) forward values: N * efa_dim * 9 complex numbers (re + im separate)
  nep_efa_data.cached_A_real.resize(N * annmb.dim * 9);
  nep_efa_data.cached_A_imag.resize(N * annmb.dim * 9);
  nep_efa_data.f12x.resize(N * paramb.MN_angular);
  nep_efa_data.f12y.resize(N * paramb.MN_angular);
  nep_efa_data.f12z.resize(N * paramb.MN_angular);

  efa_para.num_kpoints_max = 1;
  efa_para.num_kpoints = 0;

  rc = paramb.rc_radial;
  N1 = 0;
  N2 = num_atoms;
}

// ============================================================================
NEP_EFA::~NEP_EFA(void)
{
  // nothing
}

// ============================================================================
// Point EFA ANN pointers into the flat parameter buffer.
// Single-output ANN: q_n = w1 . tanh(w0 . q - b0) - b1
// ============================================================================
void NEP_EFA::update_potential(float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
  }
  ann.b1 = pointer;
  pointer += 1;
  ann.c = pointer;
}

// ============================================================================
// Host-side: build the reciprocal-space k-mesh for the current box.
// ============================================================================
void NEP_EFA::find_k_and_G(const double* box)
{
  if (k_mesh_valid) {
    bool unchanged = true;
    for (int i = 0; i < 9; ++i) {
      if (k_mesh_box[i] != box[i]) {
        unchanged = false;
        break;
      }
    }
    if (unchanged) {
      return;
    }
  }

  float a1[3] = {0.0f}, a2[3] = {0.0f}, a3[3] = {0.0f};
  float det = box[0] * (box[4] * box[8] - box[5] * box[7]) +
              box[1] * (box[5] * box[6] - box[3] * box[8]) +
              box[2] * (box[3] * box[7] - box[4] * box[6]);
  a1[0] = box[0]; a1[1] = box[3]; a1[2] = box[6];
  a2[0] = box[1]; a2[1] = box[4]; a2[2] = box[7];
  a3[0] = box[2]; a3[1] = box[5]; a3[2] = box[8];

  float b1[3], b2[3], b3[3];
  b1[0] =  a2[1] * a3[2] - a2[2] * a3[1];
  b1[1] =  a2[2] * a3[0] - a2[0] * a3[2];
  b1[2] =  a2[0] * a3[1] - a2[1] * a3[0];
  b2[0] =  a3[1] * a1[2] - a3[2] * a1[1];
  b2[1] =  a3[2] * a1[0] - a3[0] * a1[2];
  b2[2] =  a3[0] * a1[1] - a3[1] * a1[0];
  b3[0] =  a1[1] * a2[2] - a1[2] * a2[1];
  b3[1] =  a1[2] * a2[0] - a1[0] * a2[2];
  b3[2] =  a1[0] * a2[1] - a1[1] * a2[0];

  const float two_pi = 6.2831853f;
  const float two_pi_over_det = two_pi / det;
  for (int d = 0; d < 3; ++d) {
    b1[d] *= two_pi_over_det;
    b2[d] *= two_pi_over_det;
    b3[d] *= two_pi_over_det;
  }

  const float volume_k = two_pi * two_pi * two_pi / fabsf(det);
  float area_b2b3 = sqrtf(
    (b2[1] * b3[2] - b2[2] * b3[1]) * (b2[1] * b3[2] - b2[2] * b3[1]) +
    (b2[2] * b3[0] - b2[0] * b3[2]) * (b2[2] * b3[0] - b2[0] * b3[2]) +
    (b2[0] * b3[1] - b2[1] * b3[0]) * (b2[0] * b3[1] - b2[1] * b3[0]));
  float area_b3b1 = sqrtf(
    (b3[1] * b1[2] - b3[2] * b1[1]) * (b3[1] * b1[2] - b3[2] * b1[1]) +
    (b3[2] * b1[0] - b3[0] * b1[2]) * (b3[2] * b1[0] - b3[0] * b1[2]) +
    (b3[0] * b1[1] - b3[1] * b1[0]) * (b3[0] * b1[1] - b3[1] * b1[0]));
  float area_b1b2 = sqrtf(
    (b1[1] * b2[2] - b1[2] * b2[1]) * (b1[1] * b2[2] - b1[2] * b2[1]) +
    (b1[2] * b2[0] - b1[0] * b2[2]) * (b1[2] * b2[0] - b1[0] * b2[2]) +
    (b1[0] * b2[1] - b1[1] * b2[0]) * (b1[0] * b2[1] - b1[1] * b2[0]));

  const float k_max = efa_para.omega_max;
  int n1_max = int(k_max * area_b2b3 / volume_k);
  int n2_max = int(k_max * area_b3b1 / volume_k);
  int n3_max = int(k_max * area_b1b2 / volume_k);
  float ksq_max = k_max * k_max;

  std::vector<float> cpu_kx, cpu_ky, cpu_kz, cpu_G;
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
          cpu_kx.emplace_back(kx);
          cpu_ky.emplace_back(ky);
          cpu_kz.emplace_back(kz);
          const float G = fabsf(two_pi_over_det) / ksq * expf(-ksq * efa_para.alpha_factor);
          cpu_G.emplace_back(2.0f * G * efa_para.lambda_e);
        }
      }
    }
  }

  efa_para.num_kpoints = int(cpu_kx.size());
  if (efa_para.num_kpoints > efa_para.num_kpoints_max) {
    efa_para.num_kpoints_max = efa_para.num_kpoints;
    nep_efa_data.kx.resize(efa_para.num_kpoints_max);
    nep_efa_data.ky.resize(efa_para.num_kpoints_max);
    nep_efa_data.kz.resize(efa_para.num_kpoints_max);
    nep_efa_data.G.resize(efa_para.num_kpoints_max);
    nep_efa_data.S_real.resize(efa_para.num_kpoints_max);
    nep_efa_data.S_imag.resize(efa_para.num_kpoints_max);
  }
  nep_efa_data.kx.copy_from_host(cpu_kx.data(), efa_para.num_kpoints);
  nep_efa_data.ky.copy_from_host(cpu_ky.data(), efa_para.num_kpoints);
  nep_efa_data.kz.copy_from_host(cpu_kz.data(), efa_para.num_kpoints);
  nep_efa_data.G.copy_from_host(cpu_G.data(), efa_para.num_kpoints);
  for (int i = 0; i < 9; ++i) {
    k_mesh_box[i] = box[i];
  }
  k_mesh_valid = true;
}

// ============================================================================
// EFA pass: descriptor -> ANN -> k-space energy/forces -> back-prop forces.
// ============================================================================
void NEP_EFA::compute_efa_pass(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const bool small_box = is_small_box_check(paramb.rc_radial, box);

  // Step 1: ERoPE channels -> power spectrum -> attention ANN.
  if (small_box) {
    const int big_neighbor_size = 2000;
    const int size_x12 = N * big_neighbor_size;
    find_descriptors_erope_small_box<<<grid_size, BLOCK_SIZE>>>(
      paramb, annmb, N, N1, N2,
      nep.small_box_data.NN_radial.data(),
      nep.small_box_data.NL_radial.data(),
      type.data(),
      nep.small_box_data.r12.data(),
      nep.small_box_data.r12.data() + size_x12,
      nep.small_box_data.r12.data() + size_x12 * 2,
      nep_efa_data.descriptors.data(),
      nep_efa_data.cached_A_real.data(),
      nep_efa_data.cached_A_imag.data());
  } else {
    find_descriptors_erope_large_box<<<grid_size, BLOCK_SIZE>>>(
      paramb, annmb, N, N1, N2, box,
      nep.nep_data.NN_radial.data(),
      nep.nep_data.NL_radial.data(),
      type.data(),
      position.data(),
      position.data() + N,
      position.data() + N * 2,
      nep_efa_data.descriptors.data(),
      nep_efa_data.cached_A_real.data(),
      nep_efa_data.cached_A_imag.data());
  }
  GPU_CHECK_KERNEL

  apply_ann_efa_descriptor<<<grid_size, BLOCK_SIZE>>>(
    N, annmb, type.data(), nep_efa_data.descriptors.data(),
    nep_efa_data.attention.data(), nep_efa_data.attention_grad.data());
  GPU_CHECK_KERNEL

  // Step 2: enforce neutrality (sum q_n = 0)
  zero_total_attention_single<<<1, 1024>>>(
    N, N1, N2, nep_efa_data.attention.data());
  GPU_CHECK_KERNEL

  // Step 3: reciprocal-space mesh
  find_k_and_G(box.cpu_h);

  // Step 4: structure factor S(k)
  find_structure_factor_single<<<(efa_para.num_kpoints - 1) / 64 + 1, 64>>>(
    efa_para.num_kpoints, N1, N2,
    nep_efa_data.attention.data(),
    position.data(),
    position.data() + N,
    position.data() + N * 2,
    nep_efa_data.kx.data(),
    nep_efa_data.ky.data(),
    nep_efa_data.kz.data(),
    nep_efa_data.S_real.data(),
    nep_efa_data.S_imag.data());
  GPU_CHECK_KERNEL

  // Step 5: reciprocal-space energy, forces, virial, D_real
  find_force_attention_reciprocal_space_single<<<grid_size, BLOCK_SIZE>>>(
    N, N1, N2, efa_para.num_kpoints, efa_para.alpha_factor,
    nep_efa_data.attention.data(),
    position.data(),
    position.data() + N,
    position.data() + N * 2,
    nep_efa_data.kx.data(),
    nep_efa_data.ky.data(),
    nep_efa_data.kz.data(),
    nep_efa_data.G.data(),
    nep_efa_data.S_real.data(),
    nep_efa_data.S_imag.data(),
    nep_efa_data.D_real.data(),
    force.data(),
    force.data() + N,
    force.data() + N * 2,
    virial.data(),
    potential.data());
  GPU_CHECK_KERNEL

  // Step 6: chain-rule correction
  zero_mean_D_real_single<<<1, 1024>>>(
    N, N1, N2, nep_efa_data.D_real.data());
  GPU_CHECK_KERNEL

  // Step 7: back-propagate the ERoPE power spectrum with the same descriptor
  // definition used above.  The finite-difference step is deliberately local
  // to the pair kernel so training and MD share the identical derivative.
  if (small_box) {
    const int big_neighbor_size = 2000;
    const int size_x12 = N * big_neighbor_size;
    find_force_erope_small_box<<<grid_size, BLOCK_SIZE>>>(
      paramb, annmb, N, N1, N2,
      nep.small_box_data.NN_radial.data(),
      nep.small_box_data.NL_radial.data(),
      type.data(),
      nep.small_box_data.r12.data(),
      nep.small_box_data.r12.data() + size_x12,
      nep.small_box_data.r12.data() + size_x12 * 2,
      nep_efa_data.attention_grad.data(),
      nep_efa_data.D_real.data(),
      nep_efa_data.cached_A_real.data(),
      nep_efa_data.cached_A_imag.data(),
      force.data(),
      force.data() + N,
      force.data() + N * 2,
      virial.data());
  } else {
    find_force_erope_large_box<<<grid_size, BLOCK_SIZE>>>(
      paramb, annmb, N, N1, N2, box,
      nep.nep_data.NN_radial.data(),
      nep.nep_data.NL_radial.data(),
      type.data(),
      position.data(),
      position.data() + N,
      position.data() + N * 2,
      nep_efa_data.attention_grad.data(),
      nep_efa_data.D_real.data(),
      nep_efa_data.cached_A_real.data(),
      nep_efa_data.cached_A_imag.data(),
      force.data(),
      force.data() + N,
      force.data() + N * 2,
      virial.data());
  }
  GPU_CHECK_KERNEL
}

// ============================================================================
// compute: E_NEP (via nep member) + E_EFA (via compute_efa_pass)
// ============================================================================
void NEP_EFA::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  if (!box.pbc_x || !box.pbc_y || !box.pbc_z) {
    PRINT_INPUT_ERROR("Cannot use non-periodic boundaries for NEP_EFA models.");
  }

  // Step 1: compute E_NEP (energy, forces, virial, ZBL) via NEP member.
  nep.compute(box, type, position, potential, force, virial);

  // Step 2: add E_EFA contributions on top.
  compute_efa_pass(box, type, position, potential, force, virial);
}
