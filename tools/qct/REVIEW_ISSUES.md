# Native QCT Review Issues

> Historical review: the current branch assessment is
> [QCT and LSC-IVR Comprehensive Code Review V3](QCT_LSC_IVR_CODE_REVIEW_V3.md).
> This document is retained for traceability, but its status conclusions are
> superseded by V3.

Review target: `qct` commit `c8208013` plus the current working-tree changes.

Latest review: 2026-08-04. Local QCT Python tests pass (`10 passed`), and the
remote sai GPU regression pass was `10/10`. The tool execution paths below were
also checked directly; passing unit tests do not cover all of them.

This document separates resolved blocking issues from currently open risks and
scope limitations. It is not a claim that the current implementation already
provides a complete scattering package.

## Resolved

### QCT-001: Rotational quantum number changed by correction

Status: **closed**.

Semiclassical rotational velocity is stored separately from the correctable
vibrational velocity. The real-PES correction scales only the vibrational
component and reconstructs the velocity with reaction and rotational
components preserved. The rotational velocity is constructed from the sampled
diatomic geometry in `src/integrate/ensemble_qct.cu`.

### QCT-002: Missing phase-point velocities silently used 300 K velocities

Status: **closed**.

`Atom::has_velocity_in_xyz` records whether `model.xyz` supplied Cartesian
velocities. `phase_point` now rejects inputs without explicit `vel:R:3` data in
`src/integrate/ensemble_qct.cu` instead of accepting the generic velocity
initializer's fallback distribution.

### QCT-003: Single-replica correction differed from batch correction

Status: **closed**.

Single and multi-replica harmonic initialization now use the same sampler,
real-PES energy correction, deterministic retry seed sequence, and
`QCT_INITIAL v2` output path. The duplicated legacy single-replica implementation was
removed from `src/integrate/ensemble_qct.cu`.

### QCT-004: Reaction eigenvector sign differed between sampling and analysis

Status: **closed**.

The C++ Hessian path and Python mode loader use the same deterministic rule:
the largest absolute eigenvector component is positive. This removes arbitrary
sign changes from reaction-coordinate diagnostics.

## Open Blocking Issues

### QCT-005: Test suite cannot be collected

Severity: **high**. Status: **closed**.

The helper modules are tracked in `tools/qct/`, and the full QCT pytest suite
now collects and passes locally (34 tests).

### QCT-006: No repository-level GPU regression for the four fixes

Severity: **high**. Status: **open**.

The Python test covers analyzer behavior, but the repository does not contain
an automated GPU test that directly verifies: preserved Cartesian angular
momentum, rejection of missing `vel`, single-replica correction/retry
equivalence, and C++/Python reaction-coordinate sign agreement. Existing NEP89
directories are input fixtures, not a portable test runner.

Required action: add versioned GPU test scripts or a test harness with explicit
resource and software prerequisites.

## Open Physical Risks

### QCT-007: Energy correction assumes a valid mode decomposition

Severity: **medium**. Status: **partially closed**.

`apply_potential_correction()` decomposes the velocity into vibrational,
reaction, and rotational parts and rescales the vibrational kinetic energy.
This is physically consistent only when the supplied normal modes are
mass-weighted orthogonal and the rotational component is orthogonal to the
vibrational/reaction components. `validate_qct_modes()` currently checks mode
normalization and frequency but does not validate orthogonality or rigid-mode
contamination for user-authored `qct_modes.in`.

Partially resolved: `validate_qct_modes()` now checks pairwise
mass-weighted orthogonality between active stable modes, and
`classify_stationary_point()` checks active stable modes against the reaction
mode. The remaining risk is that rigid modes (translation/rotation) are not yet
checked against the active basis when supplied through user-authored
`qct_modes.in`.

### QCT-008: `dump_qct` temperature is not a molecular thermodynamic temperature

Severity: **medium**. Status: **closed**.

`src/measure/dump_qct.cu` reports
`2 * kinetic_energy / (3 * N * k_B)` for every replica. For an isolated
molecule this includes center-of-mass and rotational motion and uses `3N`
degrees of freedom even when rigid modes are excluded from QCT sampling. The
column is therefore a kinetic diagnostic, not a canonical molecular
temperature, but is currently named `temperature_K` without the distinction.

Resolved: the `qct_thermo.csv` column was renamed from `temperature_K` to
`kinetic_temperature_K` and documented as a raw `2K/(3N k_B)` diagnostic that
includes center-of-mass and rotational motion. `analyze_qct.py` reads either
column name for backward compatibility.

### QCT-009: Batch force execution is narrowly constrained

Severity: **medium**. Status: **open by design**.

`Force::configure_qct_batch()` accepts exactly one ordinary scalar NEP
potential. The batch path also requires non-periodic input and rejects normal
measurements, momentum correction, added forces, and hybrid MC/MD operations.
These constraints are reasonable for the first native implementation but must
remain visible in user-facing documentation and test coverage.

### QCT-010: Replica temperature and output reductions lack a policy

Severity: **medium**. Status: **open**.

`dump_qct` writes per-replica energy and a raw temperature diagnostic, but
standard GPUMD measurements are rejected for batch runs because they do not
have per-replica reductions. There is no common policy yet for replica-level
statistics, uncertainty estimates, failed trajectories, or output sampling
when a replica retries during initialization.

Required action: define the batch output contract before adding more
observables.

## Open Scientific Scope

### QCT-011: Scattering observables are absent

Severity: **high for scattering use**. Status: **open**.

The current tools classify connectivity, reaction completion, reaction
coordinates, fragment translation/rotation/vibration energy, and product mode
projections. They do not implement impact-parameter sampling, `bmax`,
scattering angles, reaction probability versus collision energy, ICS, or DCS.

### QCT-012: Initial-state sampling is incomplete for scattering

Severity: **high for scattering use**. Status: **open**.

The native sampler is harmonic and supports canonical, harmonic
microcanonical, fixed-mode energy, and diatomic EBK-like `v/J` initialization.
It does not provide the full thermal/microcanonical collision-state workflow,
diatomic WKB sampling, or a validated treatment of translational and
rotational reactant state distributions.

### QCT-013: Final-state quantum-state treatment is incomplete

Severity: **medium**. Status: **open**.

The analyzer can report continuous mode energies, nearest standard bins,
Gaussian weights, and below-ZPE flags when a product reference and modes are
provided. It does not yet provide a complete configurable HB/GB and soft/hard
state-to-state workflow, nor a validated ZPE-leakage correction policy.

### QCT-014: Morse benchmark is not part of GPUMD QCT

Severity: **medium for the planned teaching workflow**. Status: **open**.

No native Morse-potential QCT example or regression is currently included.
This blocks the intended two-atom analytic benchmark for sampling, integrator,
and scattering validation.

### QCT-015: Multi-GPU/MPI replica distribution is absent

Severity: **low for current scope; high for scale-out**. Status: **open**.

Current native batching fills one GPU with independent replicas. There is no
MPI or multi-GPU distribution layer, cross-rank aggregation, or global seed
contract. This should be implemented only after single-GPU batching and
resource estimation are stable.

## Documentation Consistency

### QCT-016: Tool/documentation drift

Severity: **medium**. Status: **closed**.

The files are present in the working tree, matching the README examples and
test imports. The `kinetic_temperature_K` rename is reflected in the README
and input documentation. Versioning of the restored helper files remains
covered by QCT-005.

## Latest Review Findings

### QCT-017: `phase zero` silently samples phase `pi/4`

Severity: **high**. Status: **open**.

`ensemble_qct.cu` records the user-facing mode as `phase zero`, but the sampler
actually assigns `0.25 * PI`. The generated `qct_initial.out` also records
`# phase zero`, so the recorded phase does not describe the generated Q/P
coordinates. This changes deterministic debugging trajectories and silently
changes the physical initial condition. The underlying real-PES correction
problem for a true zero phase must be handled explicitly rather than by
changing the requested phase.

### QCT-018: Batch benchmark execution crashes and does not measure memory

Severity: **high**. Status: **open**.

`benchmark_batch.py` uses `subprocess.run()`, then accesses `proc.pid`, but a
`CompletedProcess` has no `pid` attribute. A successful benchmark therefore
raises `AttributeError` before writing its result. In addition, the process has
already exited before `nvidia-smi` is queried, and `memory >= 0` is used instead
of comparing a peak allocation against 85 percent of GPU memory. The advertised
memory-safety recommendation is therefore not implemented.

### QCT-019: Batch comparison can report false equivalence

Severity: **medium**. Status: **open**.

`compare_batch.py` compares velocities and masses only when both the batch and
reference frames contain those fields. If one side omits a field, the check is
silently skipped and a zero error can be reported even when the available data
would disagree. The comparator should require compatible schemas and validate
frame metadata such as `Step` and `Time` before declaring equivalence.
