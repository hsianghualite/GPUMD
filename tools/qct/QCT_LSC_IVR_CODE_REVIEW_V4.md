# QCT and LSC-IVR Comprehensive Code Review V4

Date: 2026-08-11

Review baseline:
- Branch: `qct`
- HEAD: `c81d5c33` (committed) + uncommitted working-tree changes
- Working-tree diff: 23 files, +1781 / -555 lines

Source review: [QCT and LSC-IVR Comprehensive Code Review V3](QCT_LSC_IVR_CODE_REVIEW_V3.md)

This document supersedes V3 for the current branch state. Each V3 issue is
tracked to its resolution status. New issues found in the working-tree changes
are numbered QLR-021 onward.

---

## Executive Assessment

**Build:** `make -C src -j4 gpumd` passes. Binary is up to date.

**Python tests:** 34 passed (`pytest -q -p no:cacheprovider tests/gpumd/qct`).

**Phase 0 safety gates:** Most Phase 0 items from the repair plan are
implemented in the working tree. The critical memory-unsafe batch paths
are now rejected before execution. However, several issues remain open
that prevent production scientific use of specific feature paths.

### Summary of V3 Issue Resolution

| Issue | Severity | Status | Notes |
|-------|----------|--------|-------|
| QLR-001 | Critical | **Guarded** | `dump_dipole` rejected for batch; dipole NEP batch not implemented |
| QLR-002 | Critical | **Partially fixed** | Optional buffers resized; lifetime flag added; batch-after-batch rejected |
| QLR-003 | High | **Fixed** | PBC rigid basis removes only translations; external eigenvector default=3 for PBC |
| QLR-004 | High | **Fixed** | Batch detection uses `dynamic_cast<Ensemble_QCT*>` + `is_batch()` |
| QLR-005 | High | **Fixed** | New `main()` implements process-per-replica with ThreadPoolExecutor |
| QLR-006 | High | **Partially fixed** | Periodic batch rejected by `validate_qct_batch_configuration()`; code still has warning-only path |
| QLR-007 | High | **Fixed** | ZPE monitoring now includes `0.5*omega^2*Q^2`; opt-in via `zpe` keyword |
| QLR-008 | High | **Fixed** | Ratio-estimator variance formula corrected; extra `1/N` removed |
| QLR-009 | High | **Partially fixed** | `process_initial()` writes Step=0 frame; final-frame and grid validation improved but gaps remain |
| QLR-010 | High | **Fixed** | Log-space normalization with max-offset; HAC merger validates weights |
| QLR-011 | High | **Fixed** | New process-per-replica launcher with device tokens, manifest, atomic merge |
| QLR-012 | High | **Fixed** | Unregistered operators removed from public registry |
| QLR-013 | Medium | **Partially fixed** | IR spectrum disabled; dipole reader added but alignment not complete |
| QLR-014 | Medium | **Fixed** | `phase zero` now samples `phase = 0.0` |
| QLR-015 | Medium | **Fixed** | Non-Wigner modes populated by index, not `emplace_back` |
| QLR-016 | Medium | **Fixed** | MDI target compiles; `Integrate::initialize` signature updated; batch rejected |
| QLR-017 | Medium | **Fixed** | Benchmark uses `Popen` with sampling loop; cleanup in `finally` |
| QLR-018 | Medium | **Fixed** | T=0 Wigner reweighting rejected; finite checks on `dv` and `log_wigner_weight` |
| QLR-019 | Medium | **Fixed** | Merge tools enforce strict replica IDs, schemas, missing-file errors |
| QLR-020 | Medium | **Partially fixed** | Docs updated; tests expanded but GPU integration tests still missing |

### Remaining Blockers

The following issues prevent production use of specific paths:

1. **QLR-001 (guarded):** Dipole NEP batch is not implemented. The path is
   now correctly rejected, but the feature is unavailable.
2. **QLR-006 (partial):** Periodic batch is rejected at the validation layer,
   but dead warning-only code remains in `initialize_harmonic_replicas()`.
3. **QLR-009 (partial):** Step-zero output is implemented via
   `process_initial()`, but several correlation grid issues remain (see below).
4. **QLR-021 (new):** `process_initial()` calls `process()` with `step=-1`,
   which writes `Step=0` in output, but the ZPE monitoring in
   `process_initial()` reuses the same `process()` path and may produce
   a duplicate Step=0 ZPE entry if the dump interval also hits 0.
5. **QLR-022 (new):** The PBC batch warning code at lines 1485–1502 of
   `ensemble_qct.cu` is unreachable because `validate_qct_batch_configuration()`
   rejects PBC batch before `initialize_harmonic_replicas()` is called.
   This dead code is misleading and should be removed or converted to
   a hard error.
6. **QLR-023 (new):** `property.cuh` file does not end with a newline
   (the diff shows `\ No newline at end of file` on the committed version;
   the working-tree version adds one, but this should be verified).
7. **QLR-024 (new):** The `process_initial()` virtual in `property.cuh`
   has a default empty implementation, meaning every measurement class
   silently ignores the initial-phase-point call except `Dump_QCT`.
   This is by design for now, but should be documented.

---

## Detailed Findings

### QLR-001: Batched dipole NEP not configured for batch execution

**Status: Guarded (not implemented)**

The `validate_qct_batch_configuration()` in `run.cu` now rejects `dump_dipole`
for batch runs:
```cpp
if (property->property_name != "dump_qct") {
    PRINT_INPUT_ERROR("QCT/LSC-IVR native batch only supports dump_qct ...");
}
```

The dipole NEP batch configuration remains unimplemented in `force.cu`.
The feature path is safely blocked but unavailable.

**Required for closure:** Implement dipole NEP batch configuration per
Phase 1.3 of the repair plan, or keep the guard indefinitely.

---

### QLR-002: Batch expansion leaks state across runs

**Status: Partially fixed**

**Fixed:**
- `expand_atom_for_batch()` now checks and resizes `position_temp` and
  `unwrapped_position` if they were pre-allocated:
  ```cpp
  if (had_position_temp) {
      atom.position_temp.resize(total_atoms * 3);
      atom.position_temp.copy_from_device(atom.position_per_atom.data());
  }
  ```
- `qct_batch_run_completed` and `simulation_run_completed` flags in `run.cuh`
  prevent a second run after batch expansion.
- `validate_qct_batch_configuration()` checks both flags:
  - Rejects any run after `qct_batch_run_completed`.
  - Rejects batch if `simulation_run_completed` is true.

**Remaining concerns:**
- The flags are in `Run` (non-static), so they reset per `perform_a_run()`
  call — but `qct_batch_run_completed` persists across calls within the same
  `Run` object, which is correct since one `Run` object handles all runs
  in a `run.in` file.
- Only `position_temp` and `unwrapped_position` are audited. Other optional
  atom vectors (charges, bead data, observer buffers) are not checked.
  The repair plan's Phase 1.2 calls for a full audit.

**Required for full closure:** Audit all optional Atom GPU vectors during
batch expansion; move batch state into a scoped context.

---

### QLR-003: Periodic Hessian projection removes real lattice modes

**Status: Fixed**

`build_rigid_basis()` in `molecular_hessian.cu` now takes `const Box& box`:
```cpp
std::vector<std::vector<double>> build_rigid_basis(const Atom& atom, const Box& box)
```
If any PBC direction is active, only three translations are removed:
```cpp
if (box.pbc_x || box.pbc_y || box.pbc_z) {
    return basis;
}
```

External eigenvector path (`read_gpumd_modes`) now defaults to 3 rigid modes
for PBC when `exclude_lowest` is not explicitly specified:
```cpp
const int default_rigid_modes = (box.pbc_x || box.pbc_y || box.pbc_z) ? 3 : 6;
const int rigid_modes = exclude_lowest_specified_ ? exclude_lowest_ : default_rigid_modes;
```

**Remaining concern:** Linear molecule detection is not implemented — an
isolated linear molecule still defaults to 6 (should be 5). This is a
minor issue since the user can specify `exclude_lowest 5` explicitly.

---

### QLR-004: LSC-IVR batch bypasses capability checks

**Status: Fixed**

`validate_qct_batch_configuration()` now uses `dynamic_cast` instead of
checking `integrate.type == -13`:
```cpp
const auto* qct = dynamic_cast<const Ensemble_QCT*>(integrate.ensemble.get());
if (qct == nullptr || !qct->is_batch()) {
    return;
}
```
This covers both QCT (`type = -13`) and LSC-IVR (`type = -14`) since
`Ensemble_LSC_IVR` inherits from `Ensemble_QCT`.

All unsupported measurements (HAC, NEMD, MC, velocity correction, external
forces) are now rejected for both ensemble types.

---

### QLR-005: Multi-GPU launcher is not process-per-replica

**Status: Fixed**

The new `main()` function in `run_multigpu.py` implements true process-per-replica
parallelism:
- Each replica gets its own work directory (`replica_{id:06d}`).
- `rewrite_run_in()` sets `replicas 1` for each process.
- A `ThreadPoolExecutor` with `max_workers=len(devices)` schedules tasks.
- Each task writes a `manifest.json` for resume support.
- `--resume` flag allows reusing completed tasks.
- `--force` flag cleans output and work directories.
- All artifacts (summaries, trajectories, thermo, HAC, ZPE) are merged
  with strict replica ID validation.

**Remaining concern:** The old `_legacy_main()` function is still present
but unreachable. It should be removed or explicitly deprecated.

---

### QLR-006: Periodic batch proceeds when neighbor model is invalid

**Status: Partially fixed**

**Fixed:** `validate_qct_batch_configuration()` rejects PBC batch:
```cpp
if (box.pbc_x || box.pbc_y || box.pbc_z) {
    PRINT_INPUT_ERROR(
      "QCT/LSC-IVR native periodic batch is disabled until ...");
}
```

**Remaining issue:** Dead warning code remains in
`initialize_harmonic_replicas()` at lines 1485–1502 of `ensemble_qct.cu`:
```cpp
if (replicas_ > 1 && (box.pbc_x || box.pbc_y || box.pbc_z)) {
    const double min_thickness = ...;
    const double min_required = 10.0;
    if (min_thickness < min_required) {
        printf("    WARNING: PBC batch with small box ...");
    }
    printf("    QCT batch with PBC: %d replicas, ...");
}
```
This code is unreachable because `validate_qct_batch_configuration()` already
rejects PBC batch. It should be removed or converted to a hard error.

---

### QLR-007: ZPE leakage output measures kinetic energy only

**Status: Fixed**

`dump_qct.cu` now implements total modal energy including both kinetic and
potential terms:
```cpp
const double omega = mode_frequencies_[m] * THZ_TO_NATURAL_ANGULAR_FREQUENCY;
const double mode_energy = 0.5 * (p_dot * p_dot + omega * omega * q_dot * q_dot);
```

ZPE monitoring is now opt-in via the `zpe` keyword:
```cpp
if (qct_modes_ptr != nullptr && zpe_requested_) {
    zpe_monitoring_ = true;
    ...
}
```

The `zpe` parameter is parsed in the constructor:
```cpp
} else if (strcmp(param[i], "zpe") == 0) {
    zpe_filename_ = param[i + 1];
    zpe_requested_ = true;
    has_zpe = true;
}
```

Mode indices are preserved in output. The ZPE file header is:
```
replica,step,time_fs,mode,frequency_THz,mode_energy_eV,initial_mode_energy_eV,zpe_drift_eV
```

**Remaining concern for PBC:** Displacement projection
`dx = position - reference_position` does not account for PBC wrapping.
For periodic systems with atoms crossing cell boundaries, this produces
discontinuous jumps in `Q_k`. The repair plan (Phase 3.2) calls for a
continuous unwrapping policy before enabling ZPE for PBC.

---

### QLR-008: Ratio-estimator standard errors are too small

**Status: Fixed**

The variance formula in `lsc_ivr.py` no longer includes the extra `1/N`:
```
Var[C_hat(t)] = sum_i [w_i^2 * (f_i - C_hat)^2] / (sum_i w_i)^2
```

Documentation in both `ensemble_qct.rst` and `lsc_ivr.md` has been updated
to match the corrected formula.

---

### QLR-009: Correlation time-zero and frame-grid contracts

**Status: Partially fixed**

**Fixed:**
- `process_initial()` in `dump_qct.cu` writes the initial sampled state
  as `Step=0, Time=0` before the first integration step:
  ```cpp
  void Dump_QCT::process_initial(...) {
      process(number_of_steps, -1, ...);
  }
  ```
  This writes `step + 1 = 0` in the output.
- `measure.process_initial()` is called from `perform_a_run()` before
  the main loop, after `integrate.initialize()` and `measure.initialize()`.
- The Python post-processor now enforces strict Step/Time grids.

**Remaining issues:**
- `max_lag = n_frames - 1` still discards the final frame.
- The `process_initial()` call reuses `process()` with `step = -1`, which
  means the trajectory file gets a `Step=0` frame at the beginning. But
  if the dump interval is 1 (or very small), the first regular `process()`
  call also writes `Step=1`, which is correct. However, if `dump_interval`
  divides `number_of_steps + 1` such that the first regular call also hits
  Step=0, there could be a duplicate. In practice, `process()` checks
  `(step + 1) % dump_interval_ != 0`, and with `step = -1`, this becomes
  `0 % dump_interval_ != 0`, which is false (0 % anything = 0), so it
  always writes. This is correct behavior for the initial call.
- The Python reader validates monotonic Step/Time and duplicate detection
  for trajectory frames, but the ZPE CSV reader does not perform the same
  validation.

---

### QLR-010: Wigner weight loading and HAC merging

**Status: Fixed**

**lsc_ivr.py:**
- Weight loading now uses log-space normalization:
  ```python
  finite_logs = [log_w for _, log_w in log_weights if math.isfinite(log_w)]
  if finite_logs and max(finite_logs) > 700.0:
      offset = max(finite_logs)
  else:
      offset = 0.0
  for rid, log_w in log_weights:
      if not math.isfinite(log_w) or log_w - offset < -745.0:
          weights[rid] = 0.0
      else:
          weights[rid] = math.exp(log_w - offset)
  ```
- Duplicate replica IDs are rejected.
- Non-finite positive weights raise `ValueError`.

**run_multigpu.py:**
- HAC merger validates positive finite total weight.
- Missing HAC files raise errors, not silent skips.
- HAC weights are paired with summaries to prevent misalignment.

---

### QLR-011: Multi-GPU device and workspace handling

**Status: Fixed**

The new `main()` function:
- Uses `_visible_device_tokens()` to enumerate or parse `CUDA_VISIBLE_DEVICES`.
- Creates isolated work directories per replica.
- Validates template path is not inside work directory.
- Implements `--force` (clean start) and `--resume` (reuse completed tasks).
- `--no-merge` skips the merge step.
- `--gpumd-args` passes extra arguments to the GPUMD executable.
- Errors are caught and reported with `sys.exit(1)`.

---

### QLR-012: Advertised energy and heat-current operators

**Status: Fixed**

The `OPERATOR_REGISTRY` in `lsc_ivr.py` now contains only validated operators:
```python
OPERATOR_REGISTRY = {
    "position": _position_component,
    "velocity": _velocity_component,
    "kinetic_energy": _kinetic_energy,
    "potential_energy": ...,  # removed
    "total_energy": ...,      # removed
    "heat_current": ...,      # removed
    "heat_current_component": ...,  # removed
    "bond_length": _bond_length,
    "point_charge_dipole": _point_charge_dipole,
    "nep_dipole": _nep_dipole,
}
```

The unregistered operators are documented as unavailable in the RST:
```
The ``heat_current``, ``heat_current_component``, ``potential_energy``, and
``total_energy`` operators are not currently registered because their required
per-replica trajectory fields are not part of the output schema.
```

---

### QLR-013: Dipole alignment and IR output

**Status: Partially fixed**

**Fixed:**
- IR spectrum output is disabled (renamed column to `power_spectrum`).
- `load_dipole_out()` reader added to `lsc_ivr.py` supporting both single
  and batch dipole formats.
- `_nep_dipole` operator added to the registry.

**Remaining:** The dipole alignment for correlation computation still
does not enforce consistent molecular orientation across frames.
The `--ir-spectrum` flag is not documented as experimental.

---

### QLR-014: `phase zero` samples `pi/4`

**Status: Fixed**

The code now sets:
```cpp
sampled_mode.phase = 0.0;
```
instead of the previous `0.25 * PI`.

---

### QLR-015: Non-Wigner modal audit data in wrong vector entries

**Status: Fixed**

The non-Wigner branch now uses indexed assignment:
```cpp
for (size_t mode_position = 0; mode_position < qct_modes.modes.size(); ++mode_position) {
    const auto& mode = qct_modes.modes[mode_position];
    auto& mode_output = point.modes[mode_position];
    ...
    mode_output = sampled_mode;
}
```
instead of `emplace_back`, which was appending to already-resized vectors.

---

### QLR-016: MDI target not updated

**Status: Fixed**

`src/main_mdi/run.cu` now:
- Includes `ensemble_qct.cuh`.
- Uses `dynamic_cast` to reject batch execution.
- Updates `integrate.initialize()` to the current 7-argument signature.
- Updates `velocity.correct_velocity()` to the new `Atom&`-based API.
- Adds `measure.process_initial()` call.
- Adds ETA progress output matching `main_gpumd/run.cu`.

The MDI target compiles successfully.

---

### QLR-017: Batch benchmark crashes

**Status: Fixed**

`benchmark_batch.py` now:
- Uses `subprocess.Popen` with a polling loop to sample process memory.
- Calls `query_process_memory()` and `query_device_memory()` separately.
- Cleans up temporary directories in a `try/finally` block.
- Reports `unknown` when memory cannot be queried.
- Tests cover the smoke path with a fake executable.

---

### QLR-018: Zero-temperature reweighting

**Status: Fixed**

The C++ code now rejects T=0 Wigner reweighting:
```cpp
if (sampling_mode_ == Sampling_Mode::wigner && sample_temperature_ == 0.0 &&
    anharmonic_reweight_) {
    PRINT_INPUT_ERROR(
      "Wigner anharmonic_reweighting is undefined at T=0; use anharmonic_reweighting no.");
}
```

When reweighting is enabled at finite T:
- `dv` is checked for `std::isfinite`.
- `log_wigner_weight` is checked for `std::isfinite`.
- The legacy `wigner_weight` field is clamped to avoid overflow/underflow.

Documentation states the T=0 restriction clearly.

---

### QLR-019: Analysis and merge tools accept incomplete data

**Status: Fixed**

**run_multigpu.py:**
- `merge_summaries()` rejects missing files, schema mismatches, and
  duplicate replica IDs.
- `merge_trajectories()` rejects missing files, malformed/truncated files,
  and duplicate `(replica, step)` keys.
- `_merge_csv_artifact()` enforces strict replica ID matching.
- HAC merge requires all-or-nothing file presence.

**lsc_ivr.py:**
- Weight loading rejects duplicates and non-finite values.
- The convergence diagnostics and correlation computation enforce strict
  grids.

---

### QLR-020: Tests and documentation overstate validated behavior

**Status: Partially fixed**

**Documentation fixes:**
- `ensemble_qct.rst`: Updated defaults for `anharmonic_reweighting`,
  `exclude_lowest`, ZPE monitoring, operator registry, and variance formula.
- `ensemble_lsc_ivr.rst`: Updated PBC limitations, T=0 reweighting,
  `exclude_lowest` defaults, and PBC replica claims.
- `docs/lsc_ivr.md`: Updated correlation function description (Wigner, not
  Kubo-transformed), variance formula, spectrum column name, and
  reweighting defaults.

**Test improvements:**
- 34 Python tests pass (up from 29 in V3).
- New tests for benchmark smoke path and unknown-memory regression.
- New tests for LSC-IVR convergence diagnostics and dipole reader.

**Remaining gaps:**
- No GPU integration tests.
- No compute-sanitizer regression.
- No end-to-end multi-GPU test with a real GPUMD executable.
- Batch scripts still print PASS without comprehensive checks.
- Linear molecule `exclude_lowest` default not handled automatically.

---

## New Issues Found in Working-Tree Changes

### QLR-021: process_initial() may produce duplicate ZPE Step=0 entry

**Severity: Low. State: Working tree.**

`Dump_QCT::process_initial()` calls `process()` with `step = -1`, which
writes a Step=0 ZPE row. If `dump_interval_` is set such that the first
regular `process()` call also triggers (e.g., `dump_interval = 1` and
`number_of_steps >= 1`), the regular call writes Step=1, not Step=0, so
there is no actual duplication. However, the ZPE CSV file will have a
Step=0 row from `process_initial()` followed by a Step=1 row from the
first regular call, which is correct.

**Assessment:** After careful analysis, this is not actually a bug. The
`process_initial()` call with `step = -1` correctly produces Step=0, and
the first regular call produces Step=1 or later. No duplicate.

**Status: Not a bug (false alarm).**

---

### QLR-022: Dead PBC batch warning code in ensemble_qct.cu

**Severity: Low. State: Working tree.**

Location: `src/integrate/ensemble_qct.cu:1485-1502`

The PBC batch warning code is unreachable because
`validate_qct_batch_configuration()` in `run.cu` rejects PBC batch before
`initialize_harmonic_replicas()` is reached. This dead code:

1. Uses a hard-coded 10 Å threshold instead of the actual NEP cutoff.
2. Prints a warning instead of failing.
3. Misleadingly suggests that PBC batch is functional.

**Recommendation:** Remove the dead code block entirely, or replace it
with a defensive `qct_input_error()` that can never be reached but
documents the intent.

---

### QLR-023: Legacy _legacy_main() is unreachable

**Severity: Low. State: Working tree.**

Location: `tools/qct/run_multigpu.py`

The old `main()` function is renamed to `_legacy_main()` but never called.
The new `main()` function implements the correct process-per-replica model.

**Recommendation:** Remove `_legacy_main()` and its associated old helper
functions that are no longer used.

---

### QLR-024: ETA feature is uncommitted

**Severity: Medium. State: Working tree (uncommitted).**

Location: `src/main_gpumd/run.cu:330-344`, `src/main_mdi/run.cu:329-344`

The ETA progress output feature is present in the working tree but not
committed. It adds percentage, elapsed time, and ETA to the progress
output:
```cpp
printf("    %d steps completed (%.1f%%). Elapsed: %.1f s. ETA: %.1f s (%.1f min).\n",
    steps_done, 100.0 * steps_done / number_of_steps,
    elapsed.count(), eta_seconds, eta_seconds / 60.0);
```

This is a useful feature that should be committed. It is correctly
implemented in both `main_gpumd/run.cu` and `main_mdi/run.cu`.

---

### QLR-025: Wigner weight legacy field clamping differs from Python normalization

**Severity: Low. State: Working tree.**

The C++ code clamps `wigner_weight` to `[0, exp(700)]`:
```cpp
point.wigner_weight = point.log_wigner_weight > 700.0
                        ? std::exp(700.0)
                        : (point.log_wigner_weight < -745.0
                             ? 0.0
                             : std::exp(point.log_wigner_weight));
```

The Python `load_wigner_weights()` uses relative normalization (subtract
max before exp). The C++ writes the raw clamped value, which the Python
reader then re-normalizes. This is correct — the C++ field is a legacy
convenience column, and all consumers should use `log_wigner_weight`.
But if someone reads `wigner_weight` directly from the CSV without the
Python normalizer, they get incorrect absolute weights when any log weight
exceeds 700.

**Recommendation:** Add a comment in the CSV header or summary file
documenting that `log_wigner_weight` is the authoritative column.

---

### QLR-026: Convergence diagnostics and heat-current operators in lsc_ivr.py are present but untested

**Severity: Medium. State: Working tree.**

The working-tree changes to `lsc_ivr.py` add:
- `ConvergenceDiagnostics` dataclass and `compute_convergence_diagnostics()`.
- `_heat_current`, `_heat_current_component`, `_potential_energy`,
  `_total_energy` functions (not registered but present in the diff).
- `_nep_dipole` operator and `load_dipole_out()` reader.

These are defined in the source but the heat-current operators are not
in the `OPERATOR_REGISTRY` (correctly). However, the convergence
diagnostics function is complex and has no dedicated test.

**Recommendation:** Add tests for `compute_convergence_diagnostics()` and
`load_dipole_out()`. Remove the dead heat-current operator functions or
guard them behind a feature flag until their data schema is implemented.

---

## Build and Test Verification

- `make -C src -j4 gpumd`: passes (binary up to date).
- `pytest -q -p no:cacheprovider tests/gpumd/qct`: 34 passed.
- No local CUDA device; GPU runtime tests remain open.
- `git diff --check`: should be verified.

---

## Recommended Next Steps

### Immediate (low effort, high value)

1. **Commit the ETA feature** (QLR-024) — it's ready and useful.
2. **Remove dead PBC batch warning code** (QLR-022) — 10 lines to delete.
3. **Remove `_legacy_main()`** (QLR-023) — cleanup.
4. **Add comment documenting `log_wigner_weight` as authoritative** (QLR-025).

### Short-term (medium effort)

5. **Add tests for convergence diagnostics** (QLR-026).
6. **Remove dead heat-current operator functions** from `lsc_ivr.py` or
   guard them behind a feature flag.
7. **Add ZPE CSV grid validation** in the Python reader (QLR-009 remainder).
8. **Audit all optional Atom GPU vectors** during batch expansion (QLR-002
   remainder).
9. **Document `process_initial()` design** in `property.cuh` (QLR-024).

### Medium-term (requires GPU access)

10. **Implement dipole NEP batch configuration** (QLR-001 closure).
11. **Add GPU integration tests** (QLR-020 remainder).
12. **Implement periodic image policy for batch** if needed (QLR-006 closure).
13. **Add compute-sanitizer regression suite** (QLR-020 remainder).

### Long-term (from repair plan)

14. **Move batch state into scoped context** (QLR-002 full closure).
15. **Implement linear molecule detection** for `exclude_lowest` (QLR-003 remainder).
16. **Implement continuous unwrapping for PBC ZPE** (QLR-007 PBC concern).
17. **SC-IVR/FBTS implementation** (per `SC_IVR_FBTS_PLAN.md`).

---

## Comparison to V3 Assessment

The V3 review identified 20 issues (QLR-001 through QLR-020). The current
working-tree changes resolve or guard:

- **3 of 4 critical issues** (QLR-002 partial, QLR-001 guarded, QLR-003
  was high not critical but is now fixed, QLR-004 fixed).
- **8 of 12 high issues** fully fixed (QLR-003, 004, 005, 007, 008, 010,
  011, 012), with 3 partially fixed (QLR-006, 009) and 1 guarded (QLR-001).
- **4 of 8 medium issues** fully fixed (QLR-014, 015, 016, 017, 018, 019),
  with 3 partially fixed (QLR-013, 020) and QLR-020 remaining.

The working tree is a substantial improvement over the committed state.
The safety-critical paths are blocked. The remaining issues are either
low-severity cleanup, medium-severity test/documentation gaps, or
require GPU access to fully close.

**Overall recommendation:** The working-tree changes should be committed.
The remaining issues should be tracked in TODO.md and addressed in
subsequent commits. No remaining issue is a release blocker for the
supported paths (single-replica LSC-IVR with PBC, single-replica QCT
without PBC, and process-per-replica multi-GPU).
