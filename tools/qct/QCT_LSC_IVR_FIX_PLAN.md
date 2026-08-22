# QCT and LSC-IVR Repair Plan

Date: 2026-08-10

Source review:
[QCT and LSC-IVR Comprehensive Code Review V3](QCT_LSC_IVR_CODE_REVIEW_V3.md)

## Objective

Restore memory safety, define the scientific estimator contracts, and make
the advertised single-GPU, periodic, and multi-GPU workflows reproducible.
The plan is ordered by dependency and risk, not by file location.

No affected feature should be documented as production-ready until its phase
exit criteria and regression tests pass on a clean checkout.

## Implementation Status

The following Phase 0 safety and build items are implemented in the current
worktree:

- QCT and LSC-IVR native batch validation now runs before initialization,
  covers both ensemble types, rejects unsupported measurements, rejects
  periodic native batch, and requires batch to be the first run in a process.
- Batch expansion resizes and initializes pre-existing optional unwrapped and
  temporary position buffers instead of leaving them at the old atom stride.
- MDI initialization calls use the current `Integrate::initialize` signature;
  the MDI target now compiles, and native batch is explicitly rejected there.
- The batch benchmark uses a live child process, records peak/total GPU memory
  when observable, reports `unknown` when it is not observable, and cleans up
  temporary workspaces on every exit path.
- Non-Wigner modal samples now populate the preallocated mode slots, and
  `phase zero` now records literal zero phase and zero modal momentum.
- The Python correlation path now requires strict Step/Time grids, includes
  the final frame, uses per-lag weight sums and ratio-estimator errors, aligns
  dipoles by `(replica, Step)`, and preserves extreme log-weight ratios.
- T=0 Wigner sampling now rejects anharmonic reweighting, and unsupported
  potential/total-energy/heat-current operators plus IR spectrum output are
  disabled until their data schemas and units are implemented.
- Automatic Hessian rigid-mode projection now removes only translations for
  PBC, and external eigenvector defaults use three excluded modes for PBC
  unless the user supplies an explicit value.
- CPU regression coverage includes the benchmark smoke path and the full QCT
  Python suite (`34 passed`); `gpumd` and `gpumd-mdi` compile successfully.

Remaining Phase 0 exit work is the GPU/compute-sanitizer guard matrix and a
clean-checkout validation of the full build. Periodic batch remains disabled
until the Phase 1 cutoff and image policy is implemented.

## Priority Matrix

| Phase | Issues | Primary outcome |
|---|---|---|
| 0 | QLR-001, QLR-004, QLR-006, QLR-007, QLR-012, QLR-016, QLR-017 | Block unsafe or misleading paths and restore builds |
| 1 | QLR-001, QLR-002, QLR-004, QLR-006 | Make batch ownership, memory, and replica isolation correct |
| 2 | QLR-003 | Correct periodic and molecular normal-mode spaces |
| 3 | QLR-007, QLR-014, QLR-015, QLR-018 | Correct sampling metadata, modal output, ZPE, and weights |
| 4 | QLR-008, QLR-009, QLR-010, QLR-012, QLR-013 | Define and implement correlation/observable contracts |
| 5 | QLR-005, QLR-010, QLR-011, QLR-019 | Rebuild process-level orchestration and strict merging |
| 6 | QLR-016, QLR-017, QLR-019, QLR-020 | Expand CI, integration tests, and release documentation |

## Phase 0: Safety Gates and Build Baseline

Goal: make unsafe combinations fail before allocating or producing output.
These are deliberately conservative guards that can be relaxed only after the
corresponding implementation phase passes.

### 0.1 Disable unsupported batch combinations

Files:

- `src/main_gpumd/run.cu`
- `src/force/force.cu`
- `src/measure/dump_dipole.cu`

Changes:

1. Change batch detection from `integrate.type == -13` to an ensemble
   capability check that covers both QCT and LSC-IVR.
2. Reject all standard measurements and global modifiers for any QCT-derived
   batch, including `compute_hac`, until they advertise per-replica support.
3. Reject batch `dump_dipole` until QLR-001 is implemented and tested.
4. Reject a second `run` after a native batch has expanded persistent state.
5. Reject periodic batch execution until Phase 1.4 supplies a correct cutoff
   and image policy.

Tests:

- parser/integration rejection for QCT and LSC batch with every unsupported
  property or modifier;
- batch plus a second run fails before the second initialization;
- batch plus two NEPs and `dump_dipole` fails with one precise error message.

Exit criteria: no known memory-unsafe batch path is reachable from `run.in`.

### 0.2 Hide invalid working-tree observables

Files:

- `src/measure/dump_qct.cu`
- `tools/qct/lsc_ivr.py`
- QCT/LSC input documentation

Changes:

1. Make ZPE monitoring genuinely opt-in. Until Phase 3 implements total modal
   energy, reject or mark the option unavailable rather than emit kinetic-only
   drift.
2. Remove `potential_energy`, `total_energy`, `heat_current`, and
   `heat_current_component` from the public operator registry until their
   input schema exists.
3. Keep `--ir-spectrum` experimental or disabled until Phase 4 defines the
   transform, alignment, units, and intensity convention.

Exit criteria: no public option silently returns zero, partial energy, or a
quantity with a misleading scientific name.

### 0.3 Restore auxiliary build and benchmark execution

Files:

- `src/main_mdi/run.cu`
- `src/makefile_mdi`
- `tools/qct/benchmark_batch.py`

Changes:

1. Update both MDI calls to the current `Integrate::initialize` API and repair
   the remaining MDI API drift until the full target builds.
2. Add the MDI target to a build-only CI job.
3. Replace the benchmark's `subprocess.run()` call with `Popen` plus a sampling
   loop, or drop process-memory claims. Measure total device memory and peak
   process use while the child is alive; implement the actual 85 percent test.
4. Always clean the temporary benchmark directory in `finally`.

Exit criteria: `gpumd`, `gpumd-mdi`, Python compilation, and a `/bin/true`
benchmark smoke test all pass in CI.

## Phase 1: Batch Memory, Isolation, and Lifetime

Goal: establish one explicit owner for replica layout and make every batch
consumer use it.

### 1.1 Introduce a batch layout contract

Recommended API:

```cpp
struct Replica_Layout {
  int atoms_per_replica;
  int replicas;
  int total_atoms;
  bool periodic;
};
```

Pass this object to force and measurement components instead of copying three
independent integer fields. Validate multiplication overflow once when it is
constructed.

The layout should expose capabilities such as:

- replica-isolated neighbor evaluation;
- per-replica reduction support;
- periodic-image support;
- whether a potential can evaluate scalar energy, dipole, or another model in
  the same layout.

### 1.2 Define batch run lifetime

Short-term contract: native batch is a terminal run. Set an explicit
`batch_consumed_simulation_state` flag and reject subsequent stateful commands.

Long-term contract: put expanded Atom/Group arrays and configured potentials
inside a scoped run context. Do not attempt an implicit collapse to one final
replica, because selecting which replica survives is scientifically ambiguous.
If a continuation feature is later required, make replica selection explicit.

During expansion, audit every optional Atom vector. Resize, replicate, or
clear it deliberately; never rely on `size() > 0` after the atom count changes.
At minimum cover unwrapped positions, temporary positions, charges, bead data,
observer buffers, and per-atom measurement state.

Tests:

- ordinary run -> unwrapped output -> batch run under compute-sanitizer;
- batch run -> attempted ordinary run gives a deterministic rejection;
- batch run -> attempted second batch gives a deterministic rejection;
- no optional GPU vector remains sized to the pre-expansion atom count.

### 1.3 Configure every batch potential

Extend NEP batch configuration to supported model types, including dipole
NEP, and resize all arrays that use the atom stride:

- global and local neighbor lists;
- `NN/NL` radial and angular buffers;
- `Fp`, `sum_fxyz`, and partial-force arrays;
- CPU mirror buffers and neighbor capacity diagnostics.

Do not configure a potential by mutating only `N1/N2`. Each potential must
return a capability result for the complete `Replica_Layout`.

Tests:

- scalar plus dipole NEP, 1 vs 2 replicas, identical per-replica dipoles;
- distinct replica coordinates prove there are no cross-replica neighbors;
- compute-sanitizer reports zero memory errors;
- unsupported model types fail before force computation.

### 1.4 Implement or reject periodic images from the real cutoff

Expose the maximum active cutoff through the potential interface. For every
periodic direction, either prove `box_thickness >= 2 * cutoff` or use a
replica-aware expanded-box/cell-list path that includes all required images.
The check must account for triclinic thickness and only the actual periodic
directions.

Do not retain a hard-coded 5 Angstrom assumption or a warning-only failure.

Tests:

- periodic batch and independent single-replica force, energy, and virial
  equivalence above the minimum-image threshold;
- exact boundary cases just below, at, and above `2 * cutoff`;
- a small periodic analytic system that requires self-images;
- orthogonal and triclinic boxes.

Phase 1 exit criteria:

- compute-sanitizer passes all supported batch combinations;
- per-replica scalar energy, force, virial, and dipole match independent runs;
- unsupported lifetime and periodic cases fail before mutation.

## Phase 2: Correct Normal Modes for Molecules and Periodic Cells

Goal: make the projected mode space a function of boundary conditions rather
than the name `Molecular_Hessian`.

### 2.1 Boundary-aware rigid basis

Recommended contract:

- any PBC: remove only three mass-weighted global translations;
- fully nonperiodic linear molecule: remove five rigid modes;
- fully nonperiodic nonlinear molecule: remove six rigid modes.

Pass `Box` or an explicit boundary policy to the rigid-basis builder. Record
the selected basis kind and actual rigid count in the audit output.

### 2.2 Defer the external-mode exclusion default

Replace `exclude_lowest_=6` with an unset sentinel. Resolve the default after
`Box` is available: 3 for periodic cells, 5/6 for a classified isolated
molecule. An explicit user value continues to override the default and is
recorded in output metadata.

Clarify that the periodic automatic Hessian is the complete Gamma
representation of the chosen supercell, including primitive-cell wavevectors
folded to supercell Gamma.

Tests:

- analytic periodic spring model with exactly three translational zero modes;
- periodic N-atom active space is `3N-3` before frequency filtering;
- the 216-atom Si example reports 645 non-rigid directions;
- isolated linear and nonlinear molecules report five and six rigid modes;
- automatic and external-eigenvector periodic paths agree when configured
  equivalently;
- sampled `Var(Q_k)` and `Var(P_k)` match the harmonic Wigner formula.

Phase 2 exit criteria: all advertised periodic examples use the correct mode
count and pass analytical frequency/covariance checks.

## Phase 3: Sampling, Weights, and Modal Outputs

### 3.1 Fix phase and modal storage

Changes:

1. Assign non-Wigner samples by index into the already resized mode vector;
   remove the second `emplace_back` population.
2. Implement literal `phase zero`, or rename and document a distinct fixed
   phase option. Never record a different phase from the one sampled.
3. Add output assertions that the sum of reported modal energies matches the
   sampled total within tolerance.

Tests cover canonical, microcanonical, mode-energy, semiclassical, saddle,
and Wigner outputs, including round-trip reproduction of Cartesian Q/P.

### 3.2 Implement total modal energy for ZPE monitoring

Make the option explicit in `Dump_QCT` state. Store original mode indices,
frequencies, eigenvectors, masses, and reference positions. At each output:

```text
Q_k = sum_i e_ki sqrt(m_i) (r_i - r_i,ref)
P_k = sum_i e_ki sqrt(m_i) v_i
E_k = 0.5 P_k^2 + 0.5 omega_k^2 Q_k^2
```

For periodic systems, define a continuous displacement/unwrapping policy
before enabling the feature. Do not project wrapped coordinate jumps.

Tests:

- pure harmonic oscillator has constant `E_k` and oscillating kinetic energy;
- `phase zero` starts with the expected potential/kinetic split;
- omitted `zpe` option creates no file and performs no projection;
- output mode IDs preserve the original normal-mode indices;
- anharmonic test labels drift as harmonic modal-energy drift, not a ZPE
  conservation theorem.

### 3.3 Define the reweighting policy

Replace the boolean default with `auto|yes|no`:

- `auto`: disable at T=0, enable the documented finite-temperature
  approximation at T>0;
- explicit `yes` at T=0: reject;
- `no`: exact unit log weights.

Keep weights in log space, reject non-finite potential differences, and write
the method (`none_t0`, `none_user`, or `boltzmann_potential`) into the summary.
Do not describe the finite-temperature potential correction as an exact
anharmonic Wigner density.

Tests:

- T=0 policy matrix for `auto|yes|no`;
- `Delta V=0` gives log weight zero;
- known finite `Delta V` gives `-beta * Delta V`;
- NaN/Inf energy fails;
- extreme finite log weights remain representable without writing Inf.

Phase 3 exit criteria: all initial-condition and modal audit files are
internally energy-consistent and weight metadata fully describe the estimator.

## Phase 4: Correlation and Observable Contracts

### 4.1 Add an exact initial-measurement hook

Add a `Property::process_initial()` phase after ensemble sampling and initial
force/virial evaluation but before the first velocity-Verlet step. QCT output
must contain one frame per replica with `Step=0, Time=0`; later frames use the
completed MD step and physical time.

The initial trajectory frame must match `qct_initial.xyz` within output
precision. Give each run an identifier so local time zero is unambiguous in a
multi-run file.

### 4.2 Correct correlation statistics and grid validation

Changes:

1. Remove the extra sample-count division in the ratio-estimator variance;
   document the exact finite-sample convention.
2. Use the per-lag set of valid replicas and its per-lag weight sum.
3. Include all frames; handle one- and two-point inputs without invalid
   indexing.
4. Require monotonic, duplicate-free, compatible time grids by default.
5. Require `Step=0` in strict LSC mode. Support old shifted trajectories only
   through an explicit compatibility flag recorded in output metadata.
6. Normalize log weights by subtracting the maximum finite log weight; reject
   an empty or zero-effective-weight ensemble.

Analytic tests:

- equal-weight mean and standard error for fixed arrays;
- two unequal weights with a hand-computed ratio estimate;
- constant traces of unequal length remain constant when missing replicas are
  excluded per lag;
- duplicated, decreasing, or mismatched time grids fail;
- log weights differing above 700 retain the correct ratio.

### 4.3 Name the scientific estimator accurately

Current default output should be named `wigner_lsc` or
`wigner_sampled_classical`, not unconditionally Kubo-transformed.

If a Kubo option is added, make `correlation_kind` explicit. A defensible first
implementation can limit the frequency-domain symmetrized-to-Kubo conversion
to supported equilibrium Hermitian autocorrelations, use physical angular
frequency, implement the stable zero-frequency limit, and reject unsupported
cross-correlations or T=0 cases.

Tests for a harmonic q-q correlation:

- Wigner/symmetrized amplitude matches
  `hbar/(2*m*omega)*coth(beta*hbar*omega/2)`;
- Kubo amplitude matches `1/(beta*m*omega^2)`;
- their spectral ratio matches `tanh(x)/x`;
- the high-temperature limit converges.

### 4.4 Define operator input schemas before registration

For every operator, declare required frame fields and units. Raise an error if
a field is missing; never return zero as a fallback. If per-atom potential,
force, or virial is needed, add it to a versioned trajectory/output schema and
implement a step/replica keyed loader.

Use native HAC for supported heat-current calculations. A separate strict
LSC HAC estimator, if implemented, must compute each replica's `J_r(0)J_r(t)`
before weighted reduction and use one-replica volume. It must never square the
sum of replica currents.

For dipole/IR:

- align by `(run, replica, step)`, not list position;
- fail on missing required dipoles;
- define mean removal, transform convention, frequency prefactor, quantum
  correction, normalization, and output units;
- average raw correlations and variances consistently across axes.

Phase 4 exit criteria: every reported uncertainty and observable has an
analytic regression and a documented estimator/units contract.

## Phase 5: Process-Level Multi-GPU Execution and Strict Merge

### 5.1 Use a device worker pool

For measurement workflows, generate one task per replica with `replicas 1`.
Resolve the visible device tokens once, preserving UUIDs or explicit physical
indices from `CUDA_VISIBLE_DEVICES`. Schedule multiple tasks sequentially per
GPU when replicas exceed devices; do not encode excess work as native batch.

Each task receives a unique seed derived from a documented global seed scheme
and writes a manifest containing replica ID, seed, device token, command,
input hashes, start/end state, and exit code.

### 5.2 Make staging path-safe and non-destructive

Add an explicit `--work-dir`. Resolve the GPUMD executable and external input
dependencies before changing child `cwd`, or copy all declared dependencies
and rewrite paths. The work directory must not be inside the template.

Refuse an existing task or output directory unless the user supplies a clear
`--resume` or `--overwrite` policy. Never recursively delete `gpu_N` in the
caller's current directory by default.

### 5.3 Make merge atomic and complete

Merge by manifest replica ID rather than file order. Require exactly the
expected replicas, compatible schemas, finite log weights, matching time
grids, and unique seeds. Write into a temporary output directory and rename it
only after every artifact validates.

Do not retain stale files from a previous run. Merge or explicitly declare the
absence of every required artifact, including initial phase points and dipole
data when requested.

HAC policy:

- ordinary GPUMD time-origin HAC may be averaged only under its documented
  classical estimator contract;
- strict LSC HAC must merge per-replica initial-origin correlations with
  log-space Wigner weights;
- a process containing multiple replicas is invalid for HAC merge unless the
  source file itself contains validated per-replica curves.

Tests:

- fake GPUMD executable records device, cwd, seed, and arguments for more tasks
  than devices;
- `CUDA_VISIBLE_DEVICES=2,3` and UUID forms are preserved correctly;
- template paths work from arbitrary caller directories;
- missing, duplicate, failed, or stale replicas make merge fail atomically;
- extreme log weights merge to finite hand-computed values;
- one missing HAC file cannot shift another file onto the wrong weight.

Phase 5 exit criteria: process scheduling and merging pass without a GPU using
a fake executable, followed by a real multi-GPU equivalence run.

## Phase 6: Regression Matrix, CI, and Documentation

### 6.1 Required automated layers

CPU/Python CI:

- parser and output-schema tests;
- correlation, weights, FFT/IR (when supported), merge, and benchmark tests;
- fake-executable multi-GPU end-to-end tests;
- malformed/restart/incomplete dataset rejection tests.

Build CI:

- normal CUDA compilation;
- MDI compilation;
- warning and formatting checks for changed QCT/LSC files.

GPU correctness CI or scheduled regression:

- scalar QCT batch vs independent trajectories;
- dipole batch vs independent trajectories;
- compute-sanitizer batch suite;
- periodic force/energy/virial equivalence at cutoff boundaries;
- molecular and periodic Hessian analytical cases;
- Wigner covariance sampling at T=0 and finite T;
- exact Step=0 trajectory/observable output;
- multi-GPU vs single-process replica ensemble equivalence.

Every `.batch` script must accumulate failures and exit nonzero. Missing HAC,
missing expected peaks, non-finite values, or an invalid column schema are
failures, not warnings followed by PASS.

### 6.2 Documentation release gate

Synchronize all syntax, defaults, units, limitations, and estimator names
across:

- `doc/gpumd/input_parameters/ensemble_qct.rst`;
- `doc/gpumd/input_parameters/ensemble_lsc_ivr.rst`;
- `docs/lsc_ivr.md`;
- `tools/qct/README.md`;
- `tools/qct/MULTIGPU_GUIDE.md`.

Remove contradictory PBC replica statements and `key=value` notation, record
the real seed default, distinguish GPUMD natural-unit `HBAR` from eV*fs, and
publish only workflows covered by executable regressions.

Phase 6 exit criteria: a clean checkout can reproduce every documented quick
start, and each quick start maps to an automated or versioned regression with
explicit numerical acceptance criteria.

## Recommended Change Sequence

Keep reviewable changes isolated in this order:

1. Safety guards, MDI signature/build, and benchmark smoke fix.
2. Batch layout, lifetime guards, and optional-buffer audit.
3. Dipole NEP batch support plus compute-sanitizer tests.
4. Periodic cutoff/image policy and batch equivalence tests.
5. Boundary-aware Hessian and Wigner covariance tests.
6. Modal storage, literal phase semantics, and weight policy.
7. Step-zero output, ZPE total modal energy, and strict trajectory schema.
8. Correlation statistics, log weights, and estimator naming.
9. Operator/IR implementation only after schemas and analytical tests exist.
10. Process-per-replica multi-GPU runner and atomic merge.
11. CI matrix, documentation cleanup, and production re-enable commits.

Do not combine memory-safety fixes with scientific formula changes in one
large commit. Separate commits make GPU equivalence regressions and numerical
changes independently reviewable.

## Overall Definition of Done

The QCT/LSC-IVR work is ready for production review only when all of the
following are true:

- QLR-001 through QLR-012 are closed or their feature paths remain explicitly
  disabled;
- normal and MDI targets build from a clean checkout;
- compute-sanitizer reports no errors for every supported batch layout;
- periodic and molecular mode counts/frequencies pass analytical tests;
- Wigner moments, weighted means, and standard errors pass hand-computed or
  analytical references;
- all replica, time, seed, weight, and input provenance are validated before
  analysis or merge;
- process-level multi-GPU results match the equivalent independent-run
  ensemble;
- documentation uses the same estimator names, defaults, units, and supported
  combinations as the executable code;
- no integration script prints PASS after a missing or failed required check.
