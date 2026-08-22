# Full-GPU QCT/LSC-IVR Hessian Design

## 1. Objective

Replace the current mixed CPU/GPU automatic molecular Hessian path used by
QCT and LSC-IVR with a GPU-resident implementation for periodic condensed
phase systems. The implementation must also report progress during the long
finite-difference phase.

The target use case is a periodic solid with `N` atoms and Cartesian dimension
`D = 3N`, for example the 6x6x6 diamond cell:

- `N = 1728`
- `D = 5184`
- central finite difference force evaluations: `2D + 1 = 10369`
- dense double-precision Hessian: `D^2 * 8 = 206 MiB`

The first implementation target is periodic (`PBC`) LSC-IVR/QCT Wigner
sampling. Isolated-molecule rotational projection can remain on the existing
path until the periodic path is validated.

## 2. Current implementation

`src/phonon/molecular_hessian.cu` currently does the following:

1. Stores reference positions, displaced positions, force vectors, and the
   Hessian in host `std::vector<double>` objects.
2. Copies every displaced position to the GPU and copies every force vector
   back to the host.
3. Assembles and symmetrizes the Hessian on the CPU.
4. Builds the mass-weighted matrix and a dense vibrational complement on the
   CPU.
5. Forms the projected matrix with CPU triple loops.
6. Sends only the final projected matrix to the GPU eigensolver.

The current `build_vibrational_complement()` also creates a dense orthonormal
basis incrementally. This is avoidable for periodic solids, where only the
three mass-weighted translations need to be excluded.

## 3. Proposed GPU data flow

### 3.1 Device-resident finite differences

Allocate and reuse these device buffers:

```text
reference_position[D]
working_position[D]
force_reference[D]
force_positive[D]
force_negative[D]
hessian[D * D]
```

The reference position remains on the device for the entire Hessian build.
For each Cartesian column `c`:

1. Copy `reference_position` into `working_position` with a device-to-device
   copy or a copy kernel.
2. Add `+displacement` to coordinate `c` with a kernel.
3. Call the existing GPU force evaluator; leave the force on the device.
4. Add `-2*displacement` to coordinate `c` with a kernel.
5. Call the GPU force evaluator again.
6. Write
   `H[row, c] = (F_negative[row] - F_positive[row]) / (2*displacement)`
   with a CUDA kernel.

No force vector is copied to the host inside this loop. The reference force,
maximum-force reduction, and post-column checks are also device operations.
The working position must be restored to the reference position before the
next column and before returning to MD.

### 3.2 Symmetry and mass weighting

Use CUDA kernels to:

- symmetrize `H` in place;
- divide by `sqrt(m_row * m_column)` in place;
- optionally check the acoustic sum rule and maximum antisymmetric element.

For PBC, do not construct a dense `D x (D-3)` complement. Instead, use one of
these validated approaches:

1. **Preferred first implementation:** diagonalize the full symmetric
   mass-weighted matrix and identify the three translational modes by the
   smallest absolute eigenvalues. Preserve the existing mode ordering contract
   after sorting.
2. **Later option:** implement a GPU Householder projection that removes the
   three known translation vectors without materializing a dense complement.

The first approach avoids the current CPU `O(D^3)` complement construction and
keeps the implementation compatible with a dense symmetric eigensolver.

### 3.3 GPU eigensolver

Extend `src/utilities/cusolver_wrapper.cu` with an explicit double-precision
device API using the project CUDA solver wrapper. The API should:

- accept a device-resident symmetric matrix;
- support eigenvalues only and eigenvalues plus eigenvectors;
- return the solver `info` code and fail with a useful GPUMD error;
- expose workspace size before allocation;
- use a production symmetric eigensolver (`syevd` or the validated equivalent)
  rather than the current Jacobi-only convenience wrapper when appropriate.

Keep the matrix and eigenvectors on the device until QCT mode sampling has
finished. Host copies are permitted only for audit output and final scalar
validation.

## 4. Memory policy

The implementation must not allocate several full host-side copies of the
Hessian. For `D=5184`:

| Buffer | Approximate size |
|---|---:|
| Hessian, FP64 | 206 MiB |
| Full eigenvector matrix, FP64 | 206 MiB |
| One projected/work matrix, if required | 206 MiB |
| Position and force vectors | less than 1 MiB |

The preferred periodic path should keep at most two dense matrices plus the
solver workspace on the GPU. Query free GPU memory before allocation and emit a
clear error containing the requested and available bytes. Do not silently fall
back to the old CPU path after a partial allocation.

The text `qct_hessian.out` is large for this system. Write it in row or block
chunks from device memory after the numerical work completes; do not create a
second full host matrix solely for file output. `qct_eigenvector.out` is already
written as FP32 and should retain its existing file format unless a versioned
format change is required.

## 5. Progress reporting

Progress must be emitted during finite differences, not only before and after
the Hessian call. Add a small progress helper shared with the normal GPUMD
progress style.

Recommended output:

```text
Hessian progress: finite_difference  512/5184 columns (9.88%),
  force_evaluations=1025/10369, elapsed=42.1 s, ETA=384.7 s
```

Required behavior:

- print at the configured interval (default adaptive, about 12 updates);
- print the first and last column immediately;
- use a monotonic clock for elapsed time;
- calculate ETA only after measurable progress exists;
- flush stdout after each progress line so redirected `gpumd.log` is useful;
- report subsequent phases separately:
  `symmetrize`, `mass_weight`, `eigensolve`, `mode_reconstruction`, and
  `audit_output`;
- never print once per force evaluation or once per matrix element.

Suggested optional input keys:

```text
hessian_progress yes hessian_progress_interval 100
```

The default should be enabled for automatic QCT/LSC-IVR Hessians. A `no`
option can suppress periodic progress lines for batch logs, while fatal errors
must still be printed.

## 6. API changes

Proposed interface changes:

```cpp
struct Molecular_Hessian_Options {
  double displacement = 1.0e-3;
  bool device_resident = true;
  bool report_progress = true;
  int progress_interval = 0;
};

static Molecular_Hessian_Result compute(
  const Molecular_Hessian_Options& options,
  Force& force,
  Box& box,
  Atom& atom,
  std::vector<Group>& group);
```

The result object should own or reference device matrices until mode sampling
has consumed them. If ownership is transferred, define destruction order
explicitly so the solver workspace and device matrices are released before the
next GPUMD run phase.

The parser should accept progress settings through the existing QCT/LSC-IVR
harmonic options. Backward-compatible inputs that only specify
`hessian_displacement` must continue to work.

## 7. Correctness requirements

Before enabling the new path by default, compare it against the current path
on valid periodic cells, not a box smaller than twice the potential cutoff.
The minimum test matrix is:

1. 3x3x3 diamond Si or C, checking maximum force and Hessian elements;
2. a small isolated molecule, checking that the old six-rigid-mode behavior
   is unchanged;
3. a periodic cell with a nontrivial mass pattern, checking mass weighting;
4. a saddle/minimum classification case, checking imaginary-frequency signs.

Numerical acceptance criteria should include:

- maximum absolute Hessian difference versus the reference path;
- maximum relative eigenvalue difference for nonzero modes;
- eigenvector orthonormality;
- translational-mode count and acoustic sum-rule residual;
- deterministic output for a fixed structure and displacement;
- no device or host memory growth across repeated Hessian calls.

Add progress-output tests that capture `gpumd.log` and verify the presence of
the finite-difference phase, percentage, force-evaluation count, and final
phase lines. Tests must not depend on wall-clock ETA values.

## 8. Implementation order

1. Add device displacement, difference, reduction, symmetrization, and
   mass-weighting kernels.
2. Add a device-resident finite-difference Hessian path while retaining the
   existing CPU path behind an internal option.
3. Add progress reporting and GPU memory checks.
4. Add full-matrix device eigensolver support and periodic translation-mode
   handling.
5. Stream audit files from device buffers and release all resources cleanly.
6. Compare CPU and GPU paths on 3x3x3 systems.
7. Enable the GPU path for periodic LSC-IVR/QCT after correctness tests pass.
8. Benchmark the 6x6x6 diamond case and record force-evaluation throughput,
   eigensolver time, peak GPU memory, peak host memory, and total wall time.

## 9. Non-goals

- This change does not make a nonstationary reference structure valid for
  Wigner sampling.
- This change does not replace the dense Hessian with a sparse or local model.
- This change does not change the LSC-IVR estimator, heat-current definition,
  or anharmonic reweighting formula.
- This change does not hide insufficient GPU memory by silently moving the
  numerical path back to the CPU.

For systems beyond dense-Hessian scale, a future sparse or matrix-free path
must distinguish exact finite-cutoff sparsity from numerical truncation. Exact
sparse storage may preserve the thermal-conductivity physics; dropping small
couplings or using only a finite active-mode subset changes Wigner sampling and
must be validated against HAC convergence before being used for production
QCT/LSC-IVR.
