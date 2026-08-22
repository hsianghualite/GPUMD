# QCT and LSC-IVR Comprehensive Code Review V5

Date: 2026-08-19

Review baseline:
- Branch: `qct`
- HEAD: `c81d5c33` (committed) + uncommitted working-tree changes
- Working-tree diff: 33 files, +3758 / -927 lines

Source reviews:
- [V3](QCT_LSC_IVR_CODE_REVIEW_V3.md)
- [V4](QCT_LSC_IVR_CODE_REVIEW_V4.md) — all V4 issues fixed in prior sessions

This document covers **new issues** found in the working-tree changes that
were **not** present in V4. Issues are numbered QLR-027 onward.

---

## Executive Assessment

**Build:** `make -C src -j4 gpumd` passes.

**Python tests:** 65 passed (`pytest -q -p no:cacheprovider tests/gpumd/qct`).

**V5 status:** All 14 V5 issues (QLR-027 through QLR-040) are **Fixed** and verified.
Build passes (`make -C src -j4 gpumd`); Python tests pass (65/65).

**V4 status:** All 26 V4 issues (QLR-001 through QLR-026) are resolved.

**New changes since V4:** The working tree adds significant new C++
functionality beyond the V4 baseline:
- GPU-resident Hessian computation path (`molecular_hessian.cu`, +583 lines)
- Native QCT batch Wigner-weighted HAC (`hac.cu`, +229 lines)
- ZPE leakage monitoring in `dump_qct.cu`
- `process_initial()` virtual + `measure.process_initial()` infrastructure
- PBC rigid-mode defaults (3 translations for periodic, 6 for molecular)
- `hessian_progress` / `hessian_progress_interval` keywords
- cusolver device-resident symmetric eigensolver wrapper
- GPU mode sampling kernel (`qct_mode_accumulate`)
- ETA in step-progress reporting

### Summary of New Issues

| Issue | Severity | Status | File(s) |
|------|----------|--------|---------|
| QLR-027 | **Critical** | Fixed | `molecular_hessian.cu:743` |
| QLR-028 | **Critical** | Fixed | `run_multigpu.py:263,971` |
| QLR-029 | **High** | Fixed | `hac.cu:311` |
| QLR-030 | **High** | Fixed | `molecular_hessian.cu:421` |
| QLR-031 | **High** | Fixed | `molecular_hessian.cu:296` |
| QLR-032 | **High** | Fixed | `run_multigpu.py:238` |
| QLR-033 | **Medium** | Fixed | `run_multigpu.py:748` |
| QLR-034 | **Medium** | Fixed | `analyze_qct.py:1322` |
| QLR-035 | **Medium** | Fixed | `molecular_hessian.cu:370` |
| QLR-036 | **Medium** | Fixed | `analyze_qct.py:1330` |
| QLR-037 | **Low** | Fixed | `run_multigpu.py:43,282` |
| QLR-038 | **Low** | Fixed | `molecular_hessian.cu:492` |
| QLR-039 | **Low** | Fixed | `analyze_qct.py:1264` |
| QLR-040 | **Low** | Fixed | `lsc_ivr.py.bak` |

---

## Critical Issues

### QLR-027: GPU Hessian eigenvector basis ordering mismatch — silently wrong QCT initial conditions

**Severity: Critical**
**File:** `src/phonon/molecular_hessian.cu:743-744` vs `ensemble_qct.cu:1550-1563`

`compute_device()` stores the **raw cuSOLVER** eigenvector matrix (ascending
eigenvalue order, translation modes *last*) into `device_data`:

```cpp
// molecular_hessian.cu:743
result.device_data = std::make_shared<Molecular_Hessian_Device_Data>(
    dynamical_matrix, dimension);  // raw cuSOLVER order
```

But `finish_full_periodic_hessian()` **reorders** the basis — it puts the 3
rigid translation modes first (from `rigid_basis`) and stable modes after,
sorted by `|eigenvalue|` (`molecular_hessian.cu:296-321`). The reordered
basis is what `build_automatic_modes()` reads into `qct_modes.modes[]`
(`ensemble_qct.cu:782-793`).

At sampling time (`ensemble_qct.cu:1556`), the Q/P coefficients are packed
in `qct_modes.modes` order:

```cpp
q_coefficients[mode + dimension * replica] = sampled_points_[replica].modes[mode].Q;
```

But `qct_mode_accumulate` (`molecular_hessian.cu:421-424`) indexes the basis
as `eigenvectors[coordinate + dimension * mode]` — pairing coefficient `mode`
with column `mode` of the **un-reordered** device basis. Mode `k`'s amplitude
is therefore applied to a *different* eigenvector. This silently corrupts every
generated phase point (positions/velocities do not correspond to the sampled
mode energies). The GPU and CPU paths also produce different results for the
same input.

**Fix:** Store the reordered eigenvector basis (or the reorder permutation)
in `device_data` so the GPU sampling kernel pairs Q/P coefficients with the
same eigenvector order used to build `qct_modes.modes`.

### QLR-028: `qct_initial.xyz` merge always fails — missing `Step=` metadata

**Severity: Critical**
**File:** `tools/qct/run_multigpu.py:263` (requirement) + `:971` (call)

The multi-GPU trajectory workflow unconditionally calls
`merge_trajectories(initial_files, output / "qct_initial.xyz", ...)`. However,
`merge_trajectories` requires a `Step=(-?\d+)` header field (line 263), and
`qct_initial.xyz` as emitted by GPUMD (`ensemble_qct.cu:1489-1520`) contains
only `Replica=`, `Seed=`, `temperature=` — **no `Step=` and no `Time=`**.

Confirmed empirically: merging two valid `qct_initial.xyz` files raises
`ValueError: Missing Step metadata`. This breaks the entire trajectory
workflow. The HAC path is covered by tests, but there is no end-to-end test
for the trajectory workflow.

**Fix:** Either (a) add `Step=0` and `Time=0` to the `qct_initial.xyz`
header in `write_initial_outputs()`, or (b) make `merge_trajectories`
tolerate missing `Step=` for single-frame initial files.

---

## High Issues

### QLR-029: `compute_hac` with Wigner T=0 produces division by zero

**Severity: High**
**File:** `src/measure/hac.cu:311`

When `ensemble lsc_ivr 0` (ground-state Wigner, T=0) is combined with
`compute_hac`, the Green-Kubo prefactor becomes:

```cpp
double factor = dt * 0.5 / (K_B * temperature * temperature * box.get_volume());
```

At T=0, `temperature = 0` → `factor = ∞` (division by zero), producing
NaN thermal conductivity. There is no guard in `hac.cu::postprocess()`
or in `validate_qct_batch_configuration()`.

While T=0 Wigner is primarily for molecular dynamics, it is not explicitly
rejected for `compute_hac`, so a user can trigger this silently.

**Fix:** Add a guard: if `temperature <= 0` in HAC postprocess, raise
`PRINT_INPUT_ERROR("compute_hac requires a positive temperature; QCT/LSC-IVR
T=0 is incompatible with Green-Kubo thermal conductivity.")`.

### QLR-030: GPU mode sampling kernel includes rigid modes without explicit guard

**Severity: High**
**File:** `src/phonon/molecular_hessian.cu:421` (loop `mode = 0..dimension-1`)

The `qct_mode_accumulate` kernel sums over all `dimension` modes, including
rigid translation columns. In `ensemble_qct.cu`, rigid modes are marked
`rigid=true` and skipped during active-mode sampling, but
`sample_harmonic_point` still resizes `point.modes` to `num_modes` and only
fills active ones — rigid entries are default-constructed `Sampled_Mode`
(Q=0, P=0). The zero contribution is correct *only* if the device basis's
rigid columns are orthogonal translations. Combined with QLR-027, the rigid
columns in the device basis are the cuSOLVER's last 3 (near-zero-eigenvalue)
columns, not the analytic translation basis.

Even after fixing QLR-027, the kernel should explicitly skip rigid columns
rather than relying on Q==0.

**Fix:** Pass a `num_active_modes` parameter to the kernel or zero out the
rigid-mode coefficients explicitly before the kernel launch.

### QLR-031: Rigid-mode identification uses only sort order, no near-zero threshold

**Severity: High**
**File:** `src/phonon/molecular_hessian.cu:296-301`

`finish_full_periodic_hessian` identifies rigid modes purely by sorting on
`|eigenvalue|` and taking the first 3, then asserts
`number_of_rigid_modes == 3`. There is no guard that the 3 chosen modes are
actually near zero (e.g. `|omega2| < epsilon`). A numerically unstable
(imaginary/negative) vibrational mode with `|omega2|` smaller than a noisy
near-zero translation could be misclassified as rigid, or vice versa.

**Fix:** Add an explicit near-zero threshold check (e.g.
`|eigenvalues[order[mode]]| < 1e-6`) for the 3 chosen rigid modes.

### QLR-032: `merge_trajectories` writes non-atomically

**Severity: High**
**File:** `tools/qct/run_multigpu.py:238`

`merge_trajectories` writes directly to the final path via
`with output_file.open("w", ...) as out:`. Unlike every other merge function
(`merge_summaries` L213, `_merge_csv_artifact` L607, `merge_hac` L425 all
use `.tmp` + `os.replace`), a mid-merge failure (e.g. QLR-028's raise, or a
malformed frame) leaves a truncated/partial `qct_trajectory.xyz` or
`qct_initial.xyz` in the output directory.

**Fix:** Use the same atomic temp+replace pattern as the other merge
functions: write to `output_file.with_suffix(".tmp")`, then `os.replace`.

---

## Medium Issues

### QLR-033: No child-process cleanup on interrupt in `run_multigpu.py`

**Severity: Medium**
**File:** `tools/qct/run_multigpu.py:748-755`

`subprocess.Popen(...)` + `process.wait()` has no timeout and no
`start_new_session`/`finally: process.kill()`. Because replicas run inside a
`ThreadPoolExecutor`, a `KeyboardInterrupt` is delivered only to the main
thread; the executor's `__exit__` calls `shutdown(wait=True)`, which blocks
forever on workers stuck in `process.wait()`. Ctrl-C does not terminate
running GPUMD processes.

**Fix:** Use `start_new_session=True` in `Popen` and add a `finally` block
that terminates the process group.

### QLR-034: `load_zpe_csv` `TypeError` escapes for short/malformed rows

**Severity: Medium**
**File:** `tools/qct/analyze_qct.py:1322-1323, 1338-1339`

`csv.DictReader` fills missing trailing fields with `None` for short rows.
`int(None)` and `float(None)` both raise `TypeError`, which is **not** caught
by the `except (ValueError, KeyError)` handlers. A malformed CSV with fewer
than 8 fields produces an unhandled `TypeError` instead of the intended
clean `ValueError`.

**Fix:** Add `TypeError` to the caught exceptions:
`except (ValueError, KeyError, TypeError)`.

### QLR-035: `qct_hessian_symmetrize` has O(d) per-thread loop

**Severity: Medium (performance)**
**File:** `src/phonon/molecular_hessian.cu:370-375`

The index decomposition `while (offset >= dimension - column - 1)` is
correct but has warp divergence and O(dimension) cost per thread, making
total work O(d²) instead of O(d²/2). For large molecules this is wasteful.

**Fix:** Use a closed-form mapping (integer square root) or a 2D grid.

### QLR-036: `load_zpe_csv` does not validate `mode` range or `step` sign

**Severity: Medium**
**File:** `tools/qct/analyze_qct.py:1330-1333, 1326-1329`

`replica` is checked for `< 0`, but `mode` is not. A negative `mode` passes
silently. `step` is only checked for monotonicity, not `>= 0`. Negative steps
are nonsensical for MD time. `frequency_THz` is validated for finiteness but
not for sign (negative frequencies are physically meaningless for active
modes).

**Fix:** Add `mode >= 0`, `step >= 0`, and `frequency_THz >= 0` checks.

---

## Low Issues

### QLR-037: Dead code in `run_multigpu.py` — `ENSEMBLE_PATTERN` and `load_wigner_weights`

**Severity: Low**
**File:** `tools/qct/run_multigpu.py:43, 282-301`

`ENSEMBLE_PATTERN` (line 43) is defined but never referenced (the active
ensemble matching uses inline `re.match`). `load_wigner_weights` (lines
282-301) is defined but never called within `run_multigpu.py` (only
`load_wigner_log_weights` at L400 is used); it duplicates
`lsc_ivr.py:73`'s `load_wigner_weights`.

**Fix:** Remove both.

### QLR-038: `sample()` over-allocates `positions_device` with dead ternary

**Severity: Low**
**File:** `src/phonon/molecular_hessian.cu:492-495`

```cpp
GPU_Vector<double> positions_device(positions.size() == dim*reps
    ? positions.size() : dim*reps);
```

The ternary always resolves to `dim*reps`. The conditional is dead code.

**Fix:** Use `GPU_Vector<double> positions_device(dim*reps)`.

### QLR-039: `load_zpe_csv` is effectively dead code

**Severity: Low**
**File:** `tools/qct/analyze_qct.py:1264`

`load_zpe_csv` is not called from `main()`, `run_multigpu.py`, or `lsc_ivr.py`.
It is only exercised by tests. It is scaffolding for future ZPE-drift
analysis.

**Fix:** Wire it into the analysis pipeline or document it as scaffolding.

### QLR-040: Stale backup file `lsc_ivr.py.bak`

**Severity: Low**
**File:** `tools/qct/lsc_ivr.py.bak`

A backup file `lsc_ivr.py.bak` (22259 bytes) exists in the working tree but
is not tracked by git. It duplicates an old version of `lsc_ivr.py`.

**Fix:** Remove it.

---

## Fix Log (2026-08-20)

All 14 V5 issues have been fixed and verified. Build passes; 65/65 Python tests pass.

| Issue | Fix Applied |
|-------|------------|
| QLR-027 | `compute_device()` now stores the **reordered** eigenvector basis (after `finish_full_periodic_hessian`) instead of raw cuSOLVER order. `device_data` is created from `result.eigenvectors` (reordered), so the GPU sampling kernel `qct_mode_accumulate` pairs Q/P coefficients with the correct eigenvector columns. |
| QLR-028 | Added `Step=0 Time=0` to the `qct_initial.xyz` header in `write_initial_outputs()` (`ensemble_qct.cu:1504`), so `merge_trajectories` can parse the `Step=` field. |
| QLR-029 | Added `temperature <= 0.0` guard in `HAC::postprocess` (`hac.cu:287`) that rejects `compute_hac` for T=0 Wigner (prevents division by zero in Green-Kubo prefactor). |
| QLR-030 | Kernel `qct_mode_accumulate` now takes a `num_rigid_modes` parameter and skips rigid columns (modes 0..num_rigid_modes-1). `Molecular_Hessian_Device_Data` has a `num_rigid_modes` field (default 0), set via `set_num_rigid_modes()` after construction (`molecular_hessian.cu:785`). |
| QLR-031 | Added `rigid_threshold = 1e-6` check in `finish_full_periodic_hessian` (`molecular_hessian.cu:306`) to classify near-zero eigenvalues as rigid modes, with a diagnostic message. |
| QLR-032 | `merge_trajectories` now uses a temp-file + `os.replace()` atomic pattern (`run_multigpu.py`) to prevent partial-output corruption on failure. |
| QLR-033 | Added `start_new_session=True` to subprocess launch and `KeyboardInterrupt` handler that kills the entire process group (`run_multigpu.py:739`), ensuring clean cleanup on Ctrl+C. |
| QLR-034 | Added `TypeError` to the `except` clauses in `load_zpe_csv` integer and numeric field parsing (`analyze_qct.py:1334,1378`). |
| QLR-035 | Replaced the O(d) while-loop index decomposition in `qct_hessian_symmetrize` with a 2D grid (`blockIdx.{x,y}` → column, row) for O(1) per-thread work (`molecular_hessian.cu:381`). |
| QLR-036 | Added `mode >= 0`, `step >= 0`, and `frequency_THz >= 0` validation in `load_zpe_csv` (`analyze_qct.py:1342-1358`). |
| QLR-037 | Removed dead `ENSEMBLE_PATTERN` and `load_wigner_weights` from `run_multigpu.py`. |
| QLR-038 | Replaced the dead ternary in `sample()` with a direct allocation `GPU_Vector<double> positions_device(dim*reps)` (`molecular_hessian.cu`). |
| QLR-039 | Documented `load_zpe_csv` as scaffolding for future ZPE-drift analysis in its docstring (`analyze_qct.py:1264`). |
| QLR-040 | Deleted `tools/qct/lsc_ivr.py.bak`. |

---


---

## Verified Correct

The following areas were reviewed and found correct:

- **HAC Wigner-weighted reduction** (`hac.cu:49-106`): log-space stabilization,
  finite checks, NaN/inf rejection, and normalized weight computation are
  correct.
- **HAC per-replica isolation** (`hac.cu:291-321`): `gpu_find_hac` uses
  `blockIdx.y` for replica index and `heat_offset`/`hac_offset` per replica,
  preventing cross-replica heat-current terms.
- **HAC `gpu_sum_heat`** (`hac.cu:123-152`): grid is
  `NUM_OF_HEAT_COMPONENTS * number_of_replicas` blocks; atom index
  `replica * atoms_per_replica + n` is bounded by `total_atoms`.
- **`process_initial()` infrastructure** (`property.cuh`, `measure.cu`,
  `dump_qct.cu`): virtual default empty impl is correct; `Dump_QCT`
  overrides it to write Step=0 via `process(-1)`.
- **ZPE monitoring** (`dump_qct.cu:88-141`): initial mode energies computed
  in `preprocess()` before `process_initial()` runs; no duplicate Step=0
  ZPE entry (QLR-021 was a false alarm).
- **PBC rigid basis** (`molecular_hessian.cu:79-100`): for periodic systems,
  returns 3 translation modes only; rotations excluded. For isolated
  molecules, includes 3 rotations via `append_orthonormal`.
- **Wigner weight clamping** (`ensemble_qct.cu:1636-1640`): legacy
  `wigner_weight` column clamped to `[0, exp(700)]`; `log_wigner_weight` is
  authoritative.
- **T=0 Wigner rejection** (`ensemble_qct.cu:459-461`):
  `anharmonic_reweighting yes` at T=0 is rejected.
- **Batch-after-batch rejection** (`run.cu:378-380`):
  `qct_batch_run_completed` flag prevents second batch run.
- **PBC batch rejection** (`run.cu:393-395`): periodic batch is rejected at
  validation layer.
- **HAC merge log-space normalization** (`run_multigpu.py:303-415`): offset
  by max finite log weight, `-inf` mapping for zero linear weights, finite
  check, and `scaled_sum > 0` check are all correct.
- **Per-device scheduling** (`run_multigpu.py:941-949`): `tasks[index::len(devices)]`
  with one worker per device guarantees no GPU sharing.
- **CSV merge atomicity** (`merge_summaries`, `_merge_csv_artifact`): all use
  temp+replace with schema validation and duplicate-ID checks.
- **MDI batch rejection** (`main_mdi/run.cu:219-221`): QCT batch is rejected
  for MDI.
- **cusolver wrapper** (`cusolver_wrapper.cu:147-189`): HIP fallback returns
  -1; CUDA path correctly creates/destroys handle and copies info to host.
- **Property.cuh newline**: file now ends with newline (V4 QLR-023 resolved).

---

## Priority Action Plan (All Complete)

| Priority | Issue | Effort |
|----------|-------|--------|
| **P0** | QLR-027: Fix device_data eigenvector ordering | Medium |
| **P0** | QLR-028: Fix qct_initial.xyz merge (add Step=0) | Small |
| **P1** | QLR-029: Guard compute_hac at T=0 | Small |
| **P1** | QLR-030: Skip rigid modes in GPU sampling kernel | Small |
| **P1** | QLR-031: Add near-zero threshold for rigid modes | Small |
| **P1** | QLR-032: Make merge_trajectories atomic | Small |
| **P2** | QLR-033: Add process cleanup on interrupt | Medium |
| **P2** | QLR-034: Fix TypeError in load_zpe_csv | Trivial |
| **P2** | QLR-035: Optimize qct_hessian_symmetrize | Medium |
| **P2** | QLR-036: Add mode/step/frequency validation | Small |
| **P3** | QLR-037–040: Dead code cleanup | Trivial |
