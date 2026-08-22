# QCT and LSC-IVR Comprehensive Code Review V3

Date: 2026-08-10

Review baseline:

- Branch: `qct`
- HEAD: `c81d5c3311c2`
- Committed comparison: `origin/master...HEAD`
- Additional scope: the uncommitted working-tree changes present on
  2026-08-10
- Committed change size: 107 files, 12,702 insertions, 58 deletions

The findings below describe this review baseline. Subsequent Phase 0 guards
and build repairs are recorded in the repair plan and the verification section
below; a guarded feature is still not considered scientifically implemented.

This document supersedes the conclusions in
[LSC-IVR Comprehensive Code Review V2](LSC_IVR_CODE_REVIEW_V2.md) and
[Native QCT Review Issues](REVIEW_ISSUES.md) for the current branch state.
Those reviews remain useful as historical records, but several issues
described there as fixed or outside the implemented scope are open in the
current code.

## Executive Assessment

The normal `gpumd` target builds and the lightweight Python suite passes, but
the current QCT/LSC-IVR implementation is not ready for production scientific
use on all advertised paths. The principal release blockers are:

1. GPU memory corruption in batched dipole evaluation and in a multi-run
   unwrapped-position sequence.
2. Incorrect normal-mode projection for periodic systems.
3. Missing replica isolation in measurements and state lifetime across runs.
4. Scientifically invalid ZPE-leakage and uncertainty outputs.
5. A multi-GPU launcher whose execution and merge model does not match its
   documented process-per-replica design.

Until the corresponding issues are closed, do not use these paths for
production results:

- QCT/LSC batch with `dump_dipole`;
- periodic batch execution;
- periodic automatic Hessian/Wigner sampling;
- `lsc_ivr replicas > 1` with standard GPUMD measurements;
- the current `qct_zpe.csv` leakage metric;
- the energy, heat-current, or IR operators added in the working tree;
- `run_multigpu.py` with more replicas than GPUs or nontrivial weights.

## Severity Definitions

- **Critical**: memory corruption, out-of-bounds GPU access, or an execution
  path that can invalidate unrelated state.
- **High**: a silent error in forces, sampling, correlation functions, or
  reported scientific results; or a documented primary workflow that cannot
  work correctly.
- **Medium**: an incorrect secondary output, build regression in an optional
  target, fragile tooling, or a contract/documentation mismatch that can
  plausibly mislead a user.

## Findings

### QLR-001: Batched dipole NEP is not configured for batch execution

Severity: **critical**. State: **committed**.

Locations:

- `src/force/force.cu:505-515`
- `src/force/nep.cu:348-359,371-408,1037-1085`
- `src/measure/dump_dipole.cu:192-200`

Trigger: use `replicas > 1`, load a scalar NEP followed by a dipole NEP, and
enable `dump_dipole`.

`Force::configure_qct_batch()` calls `configure_qct_batch()` only on the first
scalar NEP. It changes only `N1/N2` on the second dipole NEP. Consequently the
second NEP's neighbor, descriptor, and partial-force buffers remain sized for
one replica, and its replica-isolated neighbor kernel is not enabled.
`Dump_Dipole` then calls that potential with the full expanded atom vectors.

Impact:

- the large-box path can write beyond GPU buffers;
- a small-box or non-crashing execution treats overlapping atoms from other
  replicas as physical neighbors;
- per-replica dipoles are therefore invalid even when the run finishes.

Required direction: reject this combination immediately, then add explicit
batch support for the dipole NEP using the same layout, buffer sizing, and
replica-isolated neighbor contract as the scalar NEP.

### QLR-002: Batch expansion leaks state across runs and can overflow unwrapped buffers

Severity: **critical**. State: **committed**.

Locations:

- `src/integrate/ensemble_qct.cu:1285-1343,1520-1523`
- `src/force/force.cu:337-341,516-519`
- `src/integrate/integrate.cu:264-273,327-353`
- `src/model/read_xyz.cu:533-558`

The batch initializer permanently expands `Atom` and `Group` to
`atoms_per_replica * replicas` and permanently enables the NEP batch layout.
Neither the ensemble destructor nor `Integrate::finalize()` or
`Force::finalize()` restores or invalidates that state.

Impact:

- a later ordinary run treats all replicas as one system in one box;
- a later QCT/LSC run can expand the already expanded topology again;
- the force object continues using the old replica partition;
- if a preceding run allocated `Atom::unwrapped_position` and
  `Atom::position_temp`, `allocate_memory_gpu()` does not resize those vectors
  after batch expansion. `Integrate::compute1()` then writes them using the
  expanded atom count, causing a GPU out-of-bounds access.

The immediate safe contract is to reject a second run after native batch
execution and to resize or clear all optional atom buffers during expansion.
The longer-term implementation should move batch-owned state into a scoped
run context instead of mutating persistent simulation state without rollback.

### QLR-003: Periodic Hessian projection removes real lattice modes

Severity: **high**. State: **committed**.

Locations:

- `src/phonon/molecular_hessian.cu:57-129,217-273`
- `src/integrate/ensemble_qct.cu:645-755`
- `src/integrate/ensemble_lsc_ivr.cu:90-107`

`Molecular_Hessian` has no boundary-condition input. It always constructs
three translations and up to three molecular rotations, projects the
dynamical matrix into the complement, and reports all of them as rigid modes.
For a fixed-cell periodic solid, only the three Gamma-point translations are
zero modes; a global rotation is not a periodic-cell symmetry.

Impact: periodic examples such as the supplied Si LSC-IVR case lose three real
collective modes. The remaining frequencies and eigenvectors are also found
in the wrong projected subspace, so the Wigner covariance, initial energy,
and heat-current statistics are biased.

The external GPUMD eigenvector path has the same policy error: its default
`exclude_lowest=6` is applied without access to `Box`, despite documentation
claiming that periodic systems are adjusted automatically.

### QLR-004: LSC-IVR batch bypasses all QCT batch capability checks

Severity: **high**. State: **committed**.

Locations:

- `src/integrate/ensemble_lsc_ivr.cu:80-87`
- `src/main_gpumd/run.cu:359-403`
- `src/measure/hac.cu:100-106,210`

`Ensemble_LSC_IVR` changes the ensemble type to `-14`, while
`Run::validate_qct_batch_configuration()` returns immediately for every type
other than `-13`. LSC inherits the same batch expansion but can therefore be
combined with global velocity correction, HNEMD, MC, external forces, HAC,
and measurements without per-replica reduction semantics.

For example, HAC sums heat current over the expanded system and normalizes it
with the one shared simulation-box volume. This introduces cross-replica terms
and incorrect scaling while producing a plausible-looking file.

Capability checks should be based on `dynamic_cast<Ensemble_QCT*>` plus
`is_batch()`, or preferably an explicit ensemble capability interface, rather
than the integer type value.

### QLR-005: The multi-GPU launcher is not process-per-replica

Severity: **high**. State: **committed**.

Locations:

- `tools/qct/run_multigpu.py:361-406`
- `tools/qct/MULTIGPU_GUIDE.md:48-73`

The launcher starts exactly one GPUMD process per GPU and writes that
process's assigned count into `replicas N`. Thus `128 replicas / 4 GPUs`
becomes four native batches of 32, rather than 128 independent process tasks
scheduled over four devices.

Consequences:

- the advertised large-system workflow returns to the O(N^2) native batch
  neighbor path;
- QCT plus HAC is rejected by the normal batch guard;
- LSC plus HAC currently runs only because of QLR-004 and produces mixed
  replica measurements;
- one HAC file cannot be combined correctly with multiple distinct Wigner
  weights from the process.

For measurement workflows, the launcher must keep `replicas 1` and schedule
multiple independent processes sequentially or concurrently on a device pool.

### QLR-006: Periodic batch proceeds when its neighbor model is invalid

Severity: **high**. State: **working tree**.

Locations:

- `src/integrate/ensemble_qct.cu:1466-1483`
- `src/force/nep.cu:497-551,1439-1445`

The working tree enables periodic batch execution. The code comments correctly
state that a box thinner than twice the cutoff requires periodic self-images
or multiple images that the replica-isolated kernel does not enumerate.
Nevertheless the implementation only prints a warning and continues. The
threshold is hard-coded to 10 Angstrom instead of reading the active NEP
cutoff, and it is calculated over all dimensions rather than only applying a
well-defined boundary policy.

Impact: initial reweighting energies, forces, virials, and complete trajectories
can be silently wrong. The path must remain rejected until it either uses an
expanded-box/cell-list implementation with replica isolation or proves the
minimum-image condition from the actual potential cutoff.

### QLR-007: The ZPE leakage output measures kinetic energy only

Severity: **high**. State: **working tree**.

Locations:

- `src/measure/dump_qct.cu:16-53,84-127,231-257`
- `doc/gpumd/input_parameters/ensemble_qct.rst:253-261`

The implementation projects velocity and reports `0.5 * P_k^2` as
`mode_energy_eV`, `initial_mode_energy_eV`, and `zpe_drift_eV`. It never
projects displacement from the reference structure and therefore omits
`0.5 * omega_k^2 * Q_k^2`.

Even a perfectly harmonic trajectory with exactly conserved modal energy will
show large periodic drift as kinetic and potential energy exchange. The output
also uses a compact active-mode counter rather than the original mode index.

In addition, the presence of stored modes unconditionally enables monitoring;
the parsed `zpe` option only changes the filename. A plain `dump_qct` therefore
creates the expensive file despite documentation describing the option as
opt-in.

### QLR-008: Ratio-estimator standard errors are too small

Severity: **high**. State: **committed and working-tree extension**.

Locations:

- `tools/qct/lsc_ivr.py:537-542,747-767`
- `docs/lsc_ivr.md:200-208`
- `doc/gpumd/input_parameters/ensemble_qct.rst:319-324`

`sum(w_i^2 * residual_i^2) / sum(w_i)^2` is already the plug-in variance of
the weighted mean. The code takes its square root after dividing by the sample
count once more. For equal weights this changes the expected `s/sqrt(N)`
scaling to approximately `s/N`.

A four-sample `[0, 1, 2, 3]` reproduction yields 0.2795 from the code instead
of the 0.5590 plug-in standard error. The convergence diagnostic repeats the
same error, and both user documents state the same extra factor.

### QLR-009: The correlation time-zero and frame-grid contracts are wrong

Severity: **high**. State: **committed**.

Locations:

- `src/main_gpumd/run.cu:256-315`
- `src/measure/dump_qct.cu:130-168`
- `tools/qct/lsc_ivr.py:678-777,1004-1021`

`dump_qct` does not write the sampled state at time zero. Its first frame is
written after a complete velocity-Verlet step and only when the dump interval
is reached. The post-processor uses that first frame as `A(0)` and shifts its
timestamp to zero instead of reading `qct_initial.xyz`.

This computes `A(t_dump) B(t_dump+t)` while retaining a weight calculated for
the original sample. The bias is especially relevant for anharmonic or
reweighted ensembles and grows with the dump interval.

Additional grid defects:

- `max_lag = n_frames - 1` always discards the final frame;
- a two-frame input returns one point and the CLI then reads point two;
- replica time grids and monotonic `Step`/`Time` values are not validated;
- late lags with missing replicas are divided by the lag-zero weight sum.

### QLR-010: Wigner weight loading and HAC merging are not numerically stable

Severity: **high**. State: **committed**.

Locations:

- `tools/qct/lsc_ivr.py:73-115,739-767`
- `tools/qct/run_multigpu.py:194-304`

The main post-processor clamps every `log_weight > 700` to the same value,
destroying relative weights, while non-finite positive weights are converted
to zero. An all-zero set silently produces a zero correlation instead of an
error. Stable normalization requires subtracting the maximum finite log
weight before exponentiation or using log-sum-exp operations directly.

The HAC merger ignores `log_wigner_weight`, takes only the first replica weight
from each process, does not reject non-finite or non-positive total weight, and
compresses the list of existing HAC files without applying the same mask to
the weights. A missing leading HAC file can therefore discard a later valid
dataset; an infinite weight can produce an all-NaN output reported as success.

### QLR-011: Multi-GPU device and workspace handling violate the CLI contract

Severity: **high**. State: **committed**.

Locations:

- `tools/qct/run_multigpu.py:44-60,366-406`
- `tools/qct/MULTIGPU_GUIDE.md:90-99`
- `tests/gpumd/qct_nep89_si/lsc_ivr_multigpu/lsc_ivr_multigpu.batch:37-56`

The tool counts entries from `CUDA_VISIBLE_DEVICES=2,3` and then replaces the
child value with `0` and `1`, selecting the wrong physical devices. Run
directories are created in the caller's current directory, so relative
potential and executable paths change meaning. The provided batch script
enters the template directory before launching; newly created `gpu_N`
directories can consequently be discovered as template contents and copied
recursively.

Existing `gpu_N` directories are removed without an explicit overwrite or
resume policy. The workspace must be explicit, isolated from the template,
and refuse destructive reuse by default.

### QLR-012: Advertised energy and heat-current operators have no data source

Severity: **high**. State: **working tree**.

Locations:

- `tools/qct/lsc_ivr.py:329-445,880-985`
- `doc/gpumd/input_parameters/ensemble_qct.rst:297-314`

The new operators read `_potential_per_atom` and `_forces` from frame metadata,
but no reader, attachment function, or CLI option ever supplies those fields.
`qct_thermo.csv` contains a replica total rather than the per-atom potential
array claimed by the operator comment.

As implemented:

- `potential_energy` always returns zero;
- `total_energy` silently becomes kinetic energy;
- heat current permanently omits potential and virial contributions.

These operators should be removed from the public registry until a versioned
trajectory schema and strict required-field validation exist. Native GPUMD HAC
should remain the supported heat-current path.

### QLR-013: Dipole alignment and IR output do not implement the advertised observable

Severity: **medium**. State: **working tree**.

Locations:

- `tools/qct/lsc_ivr.py:120-170,630-647,800-821,1023-1047`

Dipole frames carry a `step`, but attachment uses list position and ignores
that value. Different dump intervals, missing frames, and restart concatenation
therefore misalign dipoles; missing tail values are silently replaced with
zero.

The isotropic IR path replaces only the normalized correlation in the first
axis result, leaving raw correlation and standard error inconsistent in the
companion CSV. It then uses the generic `abs(FFT(C))^2` spectrum without a
defined real transform, frequency prefactor, quantum correction, mean-removal
policy, or physical units. It may reveal approximate peak locations, but it
is not an IR absorption spectrum with a defensible intensity contract.

### QLR-014: `phase zero` samples `pi/4`

Severity: **medium**. State: **committed**.

Location: `src/integrate/ensemble_qct.cu:1087-1095`.

The non-random branch assigns `0.25 * PI`, while logs and `qct_initial.out`
record `phase zero`. This silently changes the requested Q/P initial condition
and breaks deterministic debugging. If a true zero phase conflicts with the
real-PES correction algorithm, that conflict must be rejected or handled
explicitly instead of changing user input.

### QLR-015: Non-Wigner modal audit data are stored in the wrong vector entries

Severity: **medium**. State: **committed**.

Locations:

- `src/integrate/ensemble_qct.cu:933-941,1067-1108,1413-1422`

`Sampled_Point::modes` is first resized to `num_modes`, but the non-Wigner
branch appends another `num_modes` entries. The real stable-mode samples are in
the second half while `write_initial_outputs()` reads the zero-initialized
first half.

The Cartesian positions and velocities are built from the local sample and
remain correct, but canonical, microcanonical, mode-energy, and semiclassical
`qct_initial.out` rows report zero energy, phase, Q, and P for stable modes.

### QLR-016: The MDI target was not updated for the QCT initialization API

Severity: **medium**. State: **committed**.

Locations:

- `src/integrate/integrate.cuh:32-39`
- `src/main_mdi/run.cu:221,453`

`Integrate::initialize()` now requires `Force&`; both MDI call sites retain
the old signature. Compiling `main_mdi/run.cu` reproduces `Force& cannot be
initialized with int` and insufficient-argument errors. The normal Makefile
does not compile the MDI target, which hides this regression. The MDI source
also has unrelated pre-existing API drift that should be addressed in the
same build-restoration task.

### QLR-017: The batch benchmark crashes after a successful child run

Severity: **medium**. State: **committed**.

Locations:

- `tools/qct/benchmark_batch.py:115-152`
- `tests/gpumd/qct/test_benchmark_batch.py:22-41`

The benchmark uses `subprocess.run()` and then accesses
`CompletedProcess.pid`, which does not exist. A minimal successful executable
therefore reaches an `AttributeError` before a result is written. Even after
that is fixed, memory is queried only after the process exits and the current
`memory >= 0` predicate does not implement the advertised 85 percent limit.
The tests cover only the recommendation function, not process execution.

### QLR-018: Zero-temperature reweighting silently becomes unit weighting

Severity: **medium**. State: **committed**.

Locations:

- `src/integrate/ensemble_qct.cu:1528-1559`
- `tools/qct/lsc_ivr.py:73-115`

At zero temperature, the C++ code sets `beta=0`; every
`log_wigner_weight` is therefore zero regardless of `Delta V`, while setup
still reports anharmonic reweighting as enabled. The intended
`beta -> infinity` limit is not represented by this formula and requires an
explicit scientific policy. Until such a method is implemented, zero-temperature
reweighting should be rejected or clearly disabled.

The C++ path also writes `exp(log_weight)` without finite/range checks, so the
ordinary weight field can contain zero or infinity even though downstream
tools claim stable processing.

### QLR-019: Analysis and merge tools accept incomplete or stale datasets

Severity: **medium**. State: **committed**.

Locations:

- `tools/qct/analyze_qct.py:1119-1182`
- `tools/qct/run_multigpu.py:106-190,425-500`
- `tools/qct/merge_qct_results.py:20-60`

The analyzer groups frames by occurrence without enforcing monotonic,
duplicate-free `Step`/`Time` or equal replica grids. If an initial file lacks
a replica, it silently substitutes the first dumped frame. The multi-GPU merge
skips missing summaries and malformed or truncated frames, does not clean or
atomically replace its output directory, and can leave previous-run summary,
HAC, or ZPE files in place when the current run does not produce them.

Strict analysis must fail on incomplete provenance unless the user explicitly
selects a documented recovery mode.

### QLR-020: Tests and documentation overstate validated behavior

Severity: **medium**. State: **committed and working tree**.

Relevant locations:

- `tests/gpumd/qct/test_lsc_ivr.py`
- `tests/gpumd/qct_nep89_si/lsc_ivr_kappa.batch`
- `tests/gpumd/qct_nep89_si/lsc_ivr_multigpu/lsc_ivr_multigpu.batch`
- `docs/lsc_ivr.md`
- `doc/gpumd/input_parameters/ensemble_lsc_ivr.rst`

The lightweight tests do not exercise the C++ parameter transform, CUDA
sampling, batch isolation, multi-GPU main path, or benchmark execution. The
synthetic harmonic test repeats the same missing amu-A-fs conversion in its
trajectory generator and expected value, so it passes self-consistently
without validating GPUMD's physical units.

Several GPU batch scripts print PASS even after missing HAC or failed
scientific checks, and they are not run by the Python CI workflow. Current
documentation also contains mutually exclusive PBC replica claims, uses
`key=value` notation for a `key value` parser, disagrees on the default seed,
and labels a generic Wigner `A(0)B(t)` estimator as Kubo-transformed without
implementing the required operator/time-origin contract.

## Baseline Verification Performed

- `make -C src -j4 gpumd`: passed, including compilation of QCT, LSC-IVR,
  molecular Hessian, NEP, and dump sources.
- `pytest -q -p no:cacheprovider tests/gpumd/qct`: 29 passed.
- Python byte compilation for all `tools/qct/*.py`: passed.
- Direct `nvcc` compilation of `main_mdi/run.cu`: failed at the QCT-added
  `Integrate::initialize` call sites, in addition to unrelated existing MDI
  API drift.
- Working-tree `git diff --check`: passed.

No CUDA device was available in the local review environment, so no GPU
dynamics, force-equivalence, or compute-sanitizer regression was run. Passing
the current build and Python suite must not be interpreted as evidence against
the CUDA and scientific-correctness findings above.

## Phase 0 Verification

After the review baseline, the following safety and build changes were applied:

- `make -C src -j4 gpumd`: passed after moving batch validation before
  initialization and adding the periodic-batch guard.
- `make -C src -f makefile_mdi -j4`: passed after synchronizing MDI's current
  `Integrate`, `Velocity`, `Minimize`, `Hessian`, and `Cohesive` APIs.
- `pytest -q -p no:cacheprovider tests/gpumd/qct`: 31 passed, including the
  benchmark child-process and unknown-memory regression tests.
- `python3 -m py_compile tools/qct/benchmark_batch.py
  tests/gpumd/qct/test_benchmark_batch.py`: passed.
- `git diff --check`: passed.

There is still no local CUDA device. GPU equivalence, memory-sanitizer, and
runtime rejection tests remain open under the repair plan.

## Release Recommendation

Treat QLR-001 through QLR-012 as release blockers for their affected feature
paths. Apply the safety gates and staged implementation in the
[QCT and LSC-IVR Repair Plan](QCT_LSC_IVR_FIX_PLAN.md), and require new
numerical and GPU regression tests before re-enabling those paths in user
documentation.
