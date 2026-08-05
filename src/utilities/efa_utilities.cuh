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
EFA (Euclidean Fast Attention) utilities for the neuro-evolution potential.

This header collects the device-friendly mathematical primitives required by
the equivariant EFA layer (see Nature Machine Intelligence 2026,
DOI 10.1038/s42256-026-01195-y):

  * complex arithmetic (value-type, device/host)
  * factorial cache (host, for CG coefficient pre-computation)
  * associated Legendre polynomials and polar prefactors (device)
  * complex spherical harmonics Y_lm(theta, phi) (device)
  * real spherical harmonics (device, used to build pre-contracted features)
  * spherical Bessel functions j_l(x) and their radial derivatives j_l'(x)
  * Clebsch-Gordan coefficient table builder (host, generic l1,l2 -> l3)
  * Euclidean rotation position encoding (ERoPE) kernel:
        integral over S^2 of exp(i omega u.r) Y_lm(u) du
            = 4*pi*i^l * j_l(omega*r) * Y_lm(r_hat)
  * complex-to-real packing helpers for the equivariant channel outputs

The mathematical identities used here are:
  * Plane-wave expansion:  integral_{S^2} exp(i k.u) Y_lm(u) du
        = 4*pi*i^l * j_l(|k|) * Y_lm(k_hat)
  * Spherical Bessel recurrence:
        j_0(x) = sin(x)/x,  j_1(x) = sin(x)/x^2 - cos(x)/x
        j_l(x) = (2l-1)/x * j_{l-1}(x) - j_{l-2}(x)
        j_l'(x) = j_{l-1}(x) - (l+1)/x * j_l(x)
  * Associated Legendre recurrence (for the polar part of Y_lm):
        P_l^l(x) = (-1)^l (2l-1)!! (1-x^2)^{l/2}
        P_{l+1}^l(x) = x*(2l+1)*P_l^l(x)
        P_{l}^{m}(x) = ((2l-1)*x*P_{l-1}^{m}(x) - (l+m-1)*P_{l-2}^{m}(x)) / (l-m)

All device functions are marked __device__ __host__ so the same header can be
shared between the training (main_nep) and MD (force) code paths, matching the
convention used in src/utilities/nep_utilities.cuh.
------------------------------------------------------------------------------*/

#pragma once

#include <cmath>
#include <vector>

namespace efa {

// ============================================================================
// Complex number (value type, device/host compatible, no dynamic memory)
// ============================================================================
struct EFAComplex {
  float re;
  float im;
  __device__ __host__ EFAComplex(float r = 0.0f, float i = 0.0f) : re(r), im(i) {}
};

__device__ __host__ __forceinline__ EFAComplex
efa_cadd(EFAComplex a, EFAComplex b)
{
  return EFAComplex(a.re + b.re, a.im + b.im);
}

__device__ __host__ __forceinline__ EFAComplex
efa_csub(EFAComplex a, EFAComplex b)
{
  return EFAComplex(a.re - b.re, a.im - b.im);
}

__device__ __host__ __forceinline__ EFAComplex
efa_cmul(EFAComplex a, EFAComplex b)
{
  return EFAComplex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

// a * real scalar
__device__ __host__ __forceinline__ EFAComplex
efa_cscale(EFAComplex a, float s)
{
  return EFAComplex(a.re * s, a.im * s);
}

__device__ __host__ __forceinline__ EFAComplex
efa_conj(EFAComplex a)
{
  return EFAComplex(a.re, -a.im);
}

__device__ __host__ __forceinline__ float
efa_cabs2(EFAComplex a)
{
  return a.re * a.re + a.im * a.im;
}

__device__ __host__ __forceinline__ float
efa_cabs(EFAComplex a)
{
  return sqrtf(efa_cabs2(a));
}

// complex exponential e^{i theta}
__device__ __host__ __forceinline__ EFAComplex
efa_cexp_i(float theta)
{
  return EFAComplex(cosf(theta), sinf(theta));
}

// ============================================================================
// Factorial (host-side, used only to pre-compute CG coefficient tables)
// ============================================================================
inline double efa_factorial(int n)
{
  // small-integer factorial; tables in CG generation never exceed ~6! here
  double f = 1.0;
  for (int k = 2; k <= n; ++k)
    f *= static_cast<double>(k);
  return f;
}

// ============================================================================
// Associated Legendre polynomials P_l^m(cos theta)
//
// Implements the standard forward-column recurrence (m -> m+1, l -> l+1) that
// is numerically stable for |x| <= 1.  This mirrors the device implementation
// in src/measure/orientorder.cu (_associated_legendre) but is also callable
// from host code so the CG builder can reuse it if needed.
// ============================================================================
__device__ __host__ __forceinline__ double
efa_associated_legendre(int l, int m, double x)
{
  if (l < 0 || m < 0 || m > l)
    return 0.0;
  double p = 1.0;
  if (m != 0) {
    double sqx = sqrt(1.0 - x * x);
    double sign = (m % 2 == 0) ? 1.0 : -1.0;
    double prod = 1.0;
    for (int i = 1; i <= m; ++i) {
      prod *= (2 * i - 1) * sqx;
    }
    p = sign * prod;
  }
  if (l == m)
    return p;
  double pm1 = x * (2 * m + 1) * p; // P_{m+1}^{m}
  if (l == m + 1)
    return pm1;
  double pm2 = 0.0;
  for (int i = m + 2; i <= l; ++i) {
    pm2 = ((2 * i - 1) * x * pm1 - (i + m - 1) * p) / static_cast<double>(i - m);
    p = pm1;
    pm1 = pm2;
  }
  return pm2;
}

// ============================================================================
// Polar prefactor: sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!) * P_l^m(cos theta)
//
// Returns the theta-dependent part of the (real-form) spherical harmonic,
// including the Condon-Shortley sign for the standard physics convention.
// This matches _polar_prefactor in src/measure/orientorder.cu so that the
// real-form Y_lm produced below agrees with the orientorder diagnostics.
// ============================================================================
__device__ __host__ __forceinline__ double
efa_polar_prefactor(int l, int m, double costheta)
{
  const double MY_PI = 3.14159265358979323846;
  int mabs = abs(m);
  double prefactor = 1.0;
  for (int i = l - mabs + 1; i <= l + mabs; ++i)
    prefactor *= i;
  prefactor = sqrt((2 * l + 1) / (4 * MY_PI * prefactor));
  double al = efa_associated_legendre(l, mabs, costheta);
  prefactor *= al;
  if (m < 0 && (m % 2 != 0))
    prefactor = -prefactor;
  return prefactor;
}

// ============================================================================
// Complex spherical harmonics Y_lm(theta, phi) (physics convention)
//
//   Y_lm(theta, phi) = (-1)^m * sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!)
//                       * P_l^m(cos theta) * exp(i m phi)
//
// The (-1)^m Condon-Shortley phase is folded into the prefactor.  We evaluate
// the theta part once and then apply e^{i m phi}; callers that need several
// (l, m) channels for the same atom should loop m in the outer loop so that
// the e^{i m phi} recurrence can be reused.
//
// l_max = 3 is the supported maximum, so (l,m) fits in 16 channels.
// ============================================================================
__device__ __host__ __forceinline__ EFAComplex
efa_ylm_complex(int l, int m, double costheta, double phi)
{
  int mabs = abs(m);
  double prefactor = 1.0;
  for (int i = l - mabs + 1; i <= l + mabs; ++i)
    prefactor *= i;
  prefactor = sqrt((2 * l + 1) / (4.0 * 3.14159265358979323846 * prefactor));
  double al = efa_associated_legendre(l, mabs, costheta);
  if (m < 0 && (m % 2 != 0))
    prefactor = -prefactor;
  else if (m > 0 && (m % 2 != 0))
    prefactor = -prefactor;
  double ang = static_cast<double>(m) * phi;
  return EFAComplex(
    static_cast<float>(prefactor * al * cos(ang)),
    static_cast<float>(prefactor * al * sin(ang)));
}

// ============================================================================
// Real spherical harmonics (Condon-Shortley phase, real-form used by NEP)
//
//   m > 0 : sqrt(2) * Re[Y_lm]
//   m = 0 : Y_l0
//   m < 0 : sqrt(2) * Im[Y_lm]
//
// This matches the orientorder convention so that the equivariant pre-feature
// tensors agree with the rest of the code base.
// ============================================================================
__device__ __host__ __forceinline__ double
efa_ylm_real(int l, int m, double costheta, double phi)
{
  double polar = efa_polar_prefactor(l, m, costheta);
  if (m == 0)
    return polar;
  double sqrt2 = 1.41421356237309514547;
  int mabs = abs(m);
  double ang = static_cast<double>(mabs) * phi;
  if (m > 0)
    return sqrt2 * polar * cos(ang);
  else
    return sqrt2 * polar * sin(ang);
}

// ============================================================================
// Spherical Bessel function j_l(x) and its radial derivative j_l'(x)
//
// Forward recurrence is used because it is numerically stable for the small
// arguments (x = omega * r with omega on the order of a few inverse Angstroms
// and r on the order of a few Angstroms) that appear in the EFA layer.
//
//   j_0(x) = sin(x)/x,   j_1(x) = sin(x)/x^2 - cos(x)/x
//   j_l(x) = (2l-1)/x * j_{l-1}(x) - j_{l-2}(x)
//   j_l'(x) = j_{l-1}(x) - (l+1)/x * j_l(x)
//
// Both j_l and j_l' are returned in one pass so the force kernels can avoid
// recomputing the recurrence.  For x ~ 0 we use the small-x limits
//   j_0(x) -> 1,  j_l(x) -> x^l/(2l+1)!! for l >= 1.
// ============================================================================
__device__ __host__ __forceinline__ void
efa_jl_and_jlp(int l, float x, float& jl, float& jlp)
{
  const float EPS = 1.0e-8f;
  if (x < EPS) {
    // small-x limit
    if (l == 0) {
      jl = 1.0f;
      jlp = 0.0f;
    } else {
      // j_l -> x^l / (2l+1)!!   =>   j_l' = l * x^{l-1} / (2l+1)!!
      double fact = 1.0;
      for (int k = 0; k <= l; ++k)
        fact *= (2 * k + 1);
      double denom = fact; // (2l+1)!!
      double xl = 1.0;
      for (int k = 0; k < l; ++k)
        xl *= x;
      jl = static_cast<float>(xl / denom);
      if (l == 1)
        jlp = 1.0f / 3.0f;
      else {
        double xlm1 = 1.0;
        for (int k = 0; k < l - 1; ++k)
          xlm1 *= x;
        jlp = static_cast<float>(static_cast<double>(l) * xlm1 / denom);
      }
    }
    return;
  }

  float sx = sinf(x);
  float cx = cosf(x);
  float j0 = sx / x;
  float j1 = sx / (x * x) - cx / x;
  if (l == 0) {
    jl = j0;
    jlp = -j1; // j_0'(x) = -j_1(x)
    return;
  }
  if (l == 1) {
    jl = j1;
    jlp = j0 - 2.0f * j1 / x; // j_1'(x) = j_0(x) - 2/x * j_1(x)
    return;
  }
  float jm2 = j0;
  float jm1 = j1;
  float jl_cur = 0.0f;
  for (int k = 2; k <= l; ++k) {
    jl_cur = (2 * k - 1) / x * jm1 - jm2;
    jm2 = jm1;
    jm1 = jl_cur;
  }
  jl = jm1;
  // j_l'(x) = j_{l-1}(x) - (l+1)/x * j_l(x)
  jlp = jm2 - static_cast<float>(l + 1) * jl / x;
}

// only j_l(x) (no derivative)
__device__ __host__ __forceinline__ float
efa_jl(int l, float x)
{
  float jl, jlp;
  efa_jl_and_jlp(l, x, jl, jlp);
  return jl;
}

// ============================================================================
// Clebsch-Gordan coefficient table (host-side builder, generic l1,l2 -> l3)
//
// Builds a flat table of all CG coefficients C(l1,m1,l2,m2; l3,m3) needed for
// tensor products of equivariant channels up to l_max.  For l_max=3 the
// relevant coupling pairs are (l1,l2) with l1,l2 in {1,2,3} and
// |l1-l2| <= l3 <= l1+l2, l3 >= 1.
//
// The builder uses the standard Racah formula:
//   C = (-1)^{(l1-l2+m3)/2} * sqrt((2l3+1) *
//        (l1+l2-l3)! (l3+l1-l2)! (l3+l2-l1)! / (l1+l2+l3+1)!) *
//        sqrt((l3+m3)! (l3-m3)! (l1-m1)! (l1+m1)! (l2-m2)! (l2+m2)!) *
//        sum_k (-1)^k / [ k! (l1+l2-l3-k)! (l1-m1-k)! (l2+m2-k)!
//                         (l3-l2+m1+k)! (l3-l1-m2+k)! ]
//
// Returns a host vector of (l1,m1,l2,m2,l3,coeff) entries; the device kernel
// indexes this table by a precomputed offset map.
// ============================================================================
struct EFA_CG_Entry {
  int l1, m1, l2, m2, l3, m3;
  double coeff;
};

inline void efa_build_cg_table(int l_max, std::vector<EFA_CG_Entry>& table)
{
  table.clear();
  for (int l1 = 1; l1 <= l_max; ++l1) {
    for (int l2 = 1; l2 <= l_max; ++l2) {
      for (int l3 = std::abs(l1 - l2); l3 <= l1 + l2; ++l3) {
        if (l3 < 1)
          continue;
        for (int m1 = -l1; m1 <= l1; ++m1) {
          for (int m2 = -l2; m2 <= l2; ++m2) {
            int m3 = m1 + m2;
            if (m3 < -l3 || m3 > l3)
              continue;
            // Racah formula
            double prefac = (2 * l3 + 1) * efa_factorial(l1 + l2 - l3) *
                            efa_factorial(l3 + l1 - l2) *
                            efa_factorial(l3 + l2 - l1) /
                            efa_factorial(l1 + l2 + l3 + 1);
            double norm = prefac * efa_factorial(l3 + m3) *
                          efa_factorial(l3 - m3) * efa_factorial(l1 - m1) *
                          efa_factorial(l1 + m1) * efa_factorial(l2 - m2) *
                          efa_factorial(l2 + m2);
            if (norm < 0.0)
              continue;
            double sign_phase = 1.0;
            int phase_exp = (l1 - l2 + m3);
            if (phase_exp % 2 != 0)
              sign_phase = -1.0;
            double sum = 0.0;
            for (int k = std::max(0, std::max(-l3 + l2 - m1, -l3 + l1 + m2));
                 k <= std::min(l1 + l2 - l3, std::min(l1 - m1, l2 + m2));
                 ++k) {
              double term = 1.0 / (efa_factorial(k) * efa_factorial(l1 + l2 - l3 - k) *
                                    efa_factorial(l1 - m1 - k) * efa_factorial(l2 + m2 - k) *
                                    efa_factorial(l3 - l2 + m1 + k) * efa_factorial(l3 - l1 - m2 + k));
              double sgn = (k % 2 == 0) ? 1.0 : -1.0;
              sum += sgn * term;
            }
            double cg = sign_phase * sqrt(norm) * sum;
            // skip numerically zero entries to keep the table compact
            if (fabs(cg) > 1.0e-12) {
              table.push_back({l1, m1, l2, m2, l3, m3, cg});
            }
          }
        }
      }
    }
  }
}

// ============================================================================
// ERoPE (Euclidean Rotation Position Encoding) kernel
//
// Core identity (plane-wave expansion over S^2):
//
//   integral_{S^2} exp(i omega (u.r)) Y_lm(u) du
//        = 4 * pi * i^l * j_l(omega * r) * Y_lm(r_hat)
//
// where r_hat = r / |r| and j_l is the spherical Bessel function.
//
// This is the building block of the equivariant EFA layer: the global
// attention value for channel (l, m) at frequency omega is the sum over atom
// pairs of this kernel, weighted by learnable radial coefficients.
//
// Returns a complex number carrying both the i^l phase and the Y_lm(r_hat)
// factor.  Callers multiply by 4*pi themselves so that the radial part
// j_l(omega r) stays real and can be differentiated cleanly for the force
// kernel.
//
// Inputs:
//   rx, ry, rz  : pair displacement vector (Angstrom)
//   omega       : frequency (inverse Angstrom)
//   l, m        : channel indices
// Outputs:
//   jl          : j_l(omega * r)   (real, radial part)
//   ylm_r, ylm_i: Y_lm(r_hat)      (complex, angular part)
//   phase_r, phase_i : (i^l)        (complex phase)
// The full kernel is 4*pi * phase * jl * Y_lm(r_hat).
// ============================================================================
__device__ __host__ __forceinline__ void
efa_erope_kernel(
  float rx, float ry, float rz, float omega, int l, int m,
  float& jl, float& ylm_re, float& ylm_im, float& phase_re, float& phase_im)
{
  float r = sqrtf(rx * rx + ry * ry + rz * rz);
  // direction cosines (guard against r == 0)
  float rinv = (r < 1.0e-12f) ? 0.0f : 1.0f / r;
  float costheta = rz * rinv;
  float phi = atan2f(ry * rinv, rx * rinv);
  efa::EFAComplex ylm = efa_ylm_complex(l, m, costheta, phi);
  ylm_re = ylm.re;
  ylm_im = ylm.im;
  jl = efa_jl(l, omega * r);
  // i^l
  static const float IP[4][2] = {
    {1.0f, 0.0f},   // i^0 = 1
    {0.0f, 1.0f},   // i^1 = i
    {-1.0f, 0.0f},  // i^2 = -1
    {0.0f, -1.0f},  // i^3 = -i
  };
  int ll = ((l % 4) + 4) % 4;
  phase_re = IP[ll][0];
  phase_im = IP[ll][1];
}

// Convenience wrapper: returns the full ERoPE kernel value
//   K_lm(r, omega) = 4*pi * i^l * j_l(omega r) * Y_lm(r_hat)
__device__ __host__ __forceinline__ EFAComplex
efa_erope_value(float rx, float ry, float rz, float omega, int l, int m)
{
  float jl, ylm_re, ylm_im, ph_re, ph_im;
  efa_erope_kernel(rx, ry, rz, omega, l, m, jl, ylm_re, ylm_im, ph_re, ph_im);
  // (ph_re + i ph_im) * (ylm_re + i ylm_im) = real, imag
  float re = ph_re * ylm_re - ph_im * ylm_im;
  float im = ph_re * ylm_im + ph_im * ylm_re;
  const float four_pi = 12.566370614359172f;
  return EFAComplex(four_pi * jl * re, four_pi * jl * im);
}


// ============================================================================
// Analytic derivative of the ERoPE kernel K_lm w.r.t. the displacement vector
//
//   K_lm(r) = 4*pi * i^l * j_l(omega*r) * Y_lm(r_hat)
//
//   dK/d(r_vec) = 4*pi * i^l * [ omega*j_l'(omega*r) * r_hat * Y_lm
//                                 + j_l(omega*r) * dY_lm/d(r_vec) ]
//
// The angular derivatives dY_lm/d(theta) and dY_lm/d(phi) are computed by
// central finite differences (h = 1e-4) on efa_ylm_complex.  This is
// numerically stable and avoids duplicating the associated-Legendre derivative
// recurrence.  The chain-rule Jacobians d(theta)/d(r_vec) and d(phi)/d(r_vec)
// are analytic.
//
// When r ~ 0 the derivative is set to zero (the kernel is well-behaved but
// the angular parametrisation is singular).
// ============================================================================
__device__ __host__ __forceinline__ void
efa_erope_derivative(
  float rx, float ry, float rz, float omega, int l, int m,
  EFAComplex& K_out, EFAComplex& dK_dx, EFAComplex& dK_dy, EFAComplex& dK_dz)
{
  float r = sqrtf(rx * rx + ry * ry + rz * rz);
  float rinv = (r < 1.0e-12f) ? 0.0f : 1.0f / r;
  float costheta = rz * rinv;
  float phi = atan2f(ry * rinv, rx * rinv);

  // Spherical Bessel j_l(omega*r) and its radial derivative j_l'(omega*r)
  float jl, jlp;
  efa_jl_and_jlp(l, omega * r, jl, jlp);

  // Complex spherical harmonic Y_lm(theta, phi)
  EFAComplex ylm = efa_ylm_complex(l, m, costheta, phi);

  // i^l phase
  static const float IP[4][2] = {
    {1.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f}, {0.0f, -1.0f}
  };
  int ll = ((l % 4) + 4) % 4;
  float phase_re = IP[ll][0], phase_im = IP[ll][1];

  // phase * ylm
  float py_re = phase_re * ylm.re - phase_im * ylm.im;
  float py_im = phase_re * ylm.im + phase_im * ylm.re;

  const float four_pi = 12.566370614359172f;

  // K = 4*pi * i^l * jl * Y_lm
  K_out = EFAComplex(four_pi * jl * py_re, four_pi * jl * py_im);

  // --- Guard: if r ~ 0, derivative is zero ---
  if (r < 1.0e-12f) {
    dK_dx = dK_dy = dK_dz = EFAComplex(0.0f, 0.0f);
    return;
  }

  // --- Radial derivative: d(jl(omega*r))/d(r_vec) = omega * j_l'(omega*r) * r_hat ---
  float drjl = omega * jlp;  // d[jl(omega*r)]/dr
  float rhat_x = rx * rinv, rhat_y = ry * rinv, rhat_z = rz * rinv;

  // --- Angular derivatives of Y_lm ---
  // costheta = rz/r, so d(costheta)/d(r_vec) is:
  //   d(costheta)/d(rx) = -rz*rx / r^3
  //   d(costheta)/d(ry) = -rz*ry / r^3
  //   d(costheta)/d(rz) = (r^2 - rz^2) / r^3
  float r2 = r * r;
  float r3 = r2 * r;

  float dcostheta_dx = -rz * rx / r3;
  float dcostheta_dy = -rz * ry / r3;
  float dcostheta_dz = (r2 - rz * rz) / r3;

  // d(phi)/d(r_vec):  phi = atan2(ry, rx)
  //   d(phi)/d(rx) = -ry / (rx^2 + ry^2)
  //   d(phi)/d(ry) =  rx / (rx^2 + ry^2)
  //   d(phi)/d(rz) = 0
  float rho2 = rx * rx + ry * ry;  // = r^2 * sin^2(theta)
  float rho2_safe = (rho2 < 1.0e-16f) ? 1.0e-16f : rho2;
  float dphi_dx = -ry / rho2_safe;
  float dphi_dy = rx / rho2_safe;
  float dphi_dz = 0.0f;

  // dY_lm/d(costheta) and dY_lm/d(phi) via central finite differences.
  // We compute dY/d(costheta) directly (not dY/d(theta)) so that the chain
  // rule becomes: dY/d(r_vec) = dY/d(costheta) * d(costheta)/d(r_vec)
  //                            + dY/d(phi) * d(phi)/d(r_vec)
  float h_theta = 1.0e-4f;
  float ct_p = fminf(1.0f, costheta + h_theta);
  float ct_m = fmaxf(-1.0f, costheta - h_theta);
  EFAComplex ylm_tp = efa_ylm_complex(l, m, ct_p, phi);
  EFAComplex ylm_tm = efa_ylm_complex(l, m, ct_m, phi);
  float denom_theta = ct_p - ct_m;
  float dYdct_re = (ylm_tp.re - ylm_tm.re) / denom_theta;
  float dYdct_im = (ylm_tp.im - ylm_tm.im) / denom_theta;

  float h_phi = 1.0e-4f;
  EFAComplex ylm_pp = efa_ylm_complex(l, m, costheta, phi + h_phi);
  EFAComplex ylm_pm = efa_ylm_complex(l, m, costheta, phi - h_phi);
  float dYdphi_re = (ylm_pp.re - ylm_pm.re) / (2.0f * h_phi);
  float dYdphi_im = (ylm_pp.im - ylm_pm.im) / (2.0f * h_phi);

  // dY/d(r_vec) = dY/d(costheta) * d(costheta)/d(r_vec) + dY/d(phi) * d(phi)/d(r_vec)
  float dYdx_re = dYdct_re * dcostheta_dx + dYdphi_re * dphi_dx;
  float dYdx_im = dYdct_im * dcostheta_dx + dYdphi_im * dphi_dx;
  float dYdy_re = dYdct_re * dcostheta_dy + dYdphi_re * dphi_dy;
  float dYdy_im = dYdct_im * dcostheta_dy + dYdphi_im * dphi_dy;
  float dYdz_re = dYdct_re * dcostheta_dz + dYdphi_re * dphi_dz;
  float dYdz_im = dYdct_im * dcostheta_dz + dYdphi_im * dphi_dz;

  // dK/d(r_vec) = 4*pi * i^l * [ drjl * r_hat * Y_lm + jl * dY/d(r_vec) ]
  // = 4*pi * [ phase * (drjl * r_hat * ylm + jl * dY) ]
  // where phase * (complex) uses the complex multiplication rule.
  // Radial part: drjl * r_hat * Y_lm
  float rad_re_x = drjl * rhat_x * ylm.re;
  float rad_im_x = drjl * rhat_x * ylm.im;
  float rad_re_y = drjl * rhat_y * ylm.re;
  float rad_im_y = drjl * rhat_y * ylm.im;
  float rad_re_z = drjl * rhat_z * ylm.re;
  float rad_im_z = drjl * rhat_z * ylm.im;

  // Total inside brackets: radial + jl * dY
  float bx_re = rad_re_x + jl * dYdx_re;
  float bx_im = rad_im_x + jl * dYdx_im;
  float by_re = rad_re_y + jl * dYdy_re;
  float by_im = rad_im_y + jl * dYdy_im;
  float bz_re = rad_re_z + jl * dYdz_re;
  float bz_im = rad_im_z + jl * dYdz_im;

  // Multiply by 4*pi * phase (i^l)
  // (phase_re + i*phase_im) * (bx_re + i*bx_im) = ...
  dK_dx = EFAComplex(four_pi * (phase_re * bx_re - phase_im * bx_im),
                     four_pi * (phase_re * bx_im + phase_im * bx_re));
  dK_dy = EFAComplex(four_pi * (phase_re * by_re - phase_im * by_im),
                     four_pi * (phase_re * by_im + phase_im * by_re));
  dK_dz = EFAComplex(four_pi * (phase_re * bz_re - phase_im * bz_im),
                     four_pi * (phase_re * bz_im + phase_im * bz_re));
}

// ============================================================================
// Channel index helpers for the equivariant (l >= 1) block
//
// We pack all (l, m) channels with 1 <= l <= l_max into a flat array of size
//   num_channels = sum_{l=1}^{l_max} (2*l+1) = l_max*(l_max+2)
// For l_max = 3 this gives 3 + 5 + 7 = 15 channels.
// The ordering is (l=1: m=-1,0,1), (l=2: m=-2..2), (l=3: m=-3..3).
// ============================================================================
inline int efa_num_channels(int l_max)
{
  int n = 0;
  for (int l = 1; l <= l_max; ++l)
    n += 2 * l + 1;
  return n;
}

// offset of channel (l, m) within the flat block; m in [-l, l]
inline int efa_channel_offset(int l, int m)
{
  int off = 0;
  for (int ll = 1; ll < l; ++ll)
    off += 2 * ll + 1;
  return off + (m + l);
}

// inverse: given a flat index, recover (l, m)
inline void efa_channel_decode(int idx, int& l, int& m)
{
  int acc = 0;
  for (int ll = 1;; ++ll) {
    int width = 2 * ll + 1;
    if (idx < acc + width) {
      l = ll;
      m = (idx - acc) - ll;
      return;
    }
    acc += width;
  }
}

} // namespace efa
