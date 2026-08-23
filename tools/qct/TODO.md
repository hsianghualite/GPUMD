# QCT / LSC-IVR TODO

Status is based on `qct` branch HEAD `c81d5c33` + uncommitted working-tree
changes (33 files, +3,834 / -941 lines). Code review V5 complete: all 40
issues (QLR-001–QLR-040) fixed. Build passes; 65/65 Python tests pass.

## Current Scope

The `qct` branch provides:

- **QCT**: `phase_point`, `canonical`, `microcanonical`, `mode_energy`,
  `semiclassical (v,J)` sampling; automatic finite-difference Hessians
  (CPU and GPU paths); external `qct_modes.in` / `eigenvector.out`;
  first-order saddle launches; native single-GPU replica batching with
  scalar NEP.
- **LSC-IVR**: dedicated `ensemble lsc_ivr` with PBC support; Wigner
  thermal sampling; anharmonic reweighting (`log_wigner_weight`);
  `dump_qct` trajectory + ZPE monitoring; `compute_hac` with
  Wigner-weighted HAC reduction; multi-GPU process-level parallelism
  (方案A) with `run_multigpu.py`; `lsc_ivr.py` post-processor
  (7 operators, FFT spectrum).
- **Code review**: 5 rounds (V1→V5), 40 issues found and fixed.
- **Tests**: 65 Python tests; tested on OH (2 atoms), ethanol (9 atoms),
  and Si/NEP89 (216–1728 atoms) for thermal conductivity.

---

## Completed Work (Reference)

### LSC-IVR core

- [x] Wigner thermal distribution sampling (`wigner` mode)
- [x] Anharmonic reweighting with `wigner_weight` / `log_wigner_weight`
- [x] `lsc_ivr.py` post-processor: weighted correlation, FFT, 7 operators
- [x] Saddle-point guard for Wigner (skip reaction-momentum injection)
- [x] `2π` frequency conversion bug fix
- [x] LSC-IVR user documentation (`docs/lsc_ivr.md`)
- [x] Tested on OH radical (2 atoms) and ethanol (9 atoms)
- [x] Dedicated `ensemble lsc_ivr` with PBC support for condensed phase
- [x] GPU-resident Hessian computation (`molecular_hessian.cu`, cuSOLVER)
- [x] Native QCT batch Wigner-weighted HAC (`hac.cu`)
- [x] ZPE leakage monitoring in `dump_qct.cu`
- [x] SC-IVR/FBTS implementation plan (`SC_IVR_FBTS_PLAN.md`)
- [x] Code review V1–V5 (40 issues, all fixed)
- [x] LSC-IVR thermal conductivity guide (`LSC_IVR_KAPPA_GUIDE.md`)

### Multi-GPU and HAC merge

- [x] `run_multigpu.py --shared-eigenvector PATH` with hash validation
- [x] `run_multigpu.py --workflow hac` (HAC-only job support)
- [x] Process-per-replica scheduling with per-GPU serial task queue
- [x] WSL CUDA child environment normalization
- [x] Legacy multi-GPU launcher removed
- [x] Log-space Wigner-weighted HAC merge with `hac_uncertainty.csv`
- [x] Strict replica completion requirement (no warn-and-skip)
- [x] Atomic file writes for all merged outputs
- [x] `hac_merge_manifest.json` with provenance and diagnostics
- [x] Production acceptance: 5/5 replicas, N_eff ≥ 3, max weight ≤ 0.5

### Documentation

- [x] Output schemas documented (`docs/lsc_ivr.md`)
- [x] Limitations section in QCT input docs
- [x] LSC-IVR workflow documentation
- [x] Command examples synchronized with existing files
- [x] `compare_batch.py` and `benchmark_batch.py` restored

---

## 1. LSC-IVR Thermal Conductivity: Validation & Comparison

*From "can run" to "can produce scientific conclusions."*

### 1.1 Convergence benchmarks

- [ ] Establish classical MD convergence baseline for Si/NEP89: determine
  required supercell size (4×4×4 vs 8×8×8 vs 12×12×12), trajectory length
  (ns scale), and sample count, as the reference for LSC-IVR comparison.
- [ ] Systematic LSC-IVR vs classical MD comparison at ≥ 3 temperatures
  (100 K, 300 K, 800 K): quantify the magnitude and direction of the quantum
  correction.
- [ ] LSC-IVR vs RPMD thermal conductivity comparison on the same system
  and temperature; verify whether the two methods agree within statistical
  error.
- [ ] LSC-IVR vs QTB thermal conductivity comparison: evaluate whether QTB
  effective quantum correction differs significantly from LSC-IVR Wigner
  correction.
- [ ] Analyze Wigner weight distribution and effective sample size N_eff
  vs system size and temperature: document when additional trajectories are
  needed for convergence.

### 1.2 Temperature dependence

- [ ] Low-temperature validation (T < Debye temperature): verify that
  LSC-IVR gives a more physically reasonable κ(T) than classical MD.
- [ ] Multi-temperature sweep for periodic systems: implement a
  `temperature_sweep` mode in `run_multigpu.py` that generates multiple
  job directories (one per temperature), each with the same periodic
  structure, NEP potential, and HAC parameters but different Wigner
  sampling temperatures. This works for both molecular and condensed-phase
  (PBC) systems via 方案A (`replicas 1` per process). Implementation is
  pure Python: template a `run.in` with a temperature placeholder, create
  `T_100K/`, `T_300K/`, `T_800K/` subdirectories, and submit them as
  independent multi-GPU jobs. No C++ modification needed.
- [ ] Validate on Si/NEP89 periodic system: run κ(T) at 100, 200, 300,
  500, 800 K and compare LSC-IVR vs classical MD temperature dependence.

### 1.3 Material generality

- [ ] Validate on at least one system other than Si: e.g., Ge, GaAs, or a
  strong-anharmonicity / phonon-bandgap material.

### 1.4 Physical approximation analysis

- [ ] Quantify harmonic-approximation error: compare the harmonic potential
  used in Wigner sampling vs the true NEP potential; record the fraction of
  rejected samples and weight distribution when `anharmonic_reweighting yes`.
- [ ] Evaluate anharmonic reweighting impact on κ: compare
  `anharmonic_reweighting yes` vs `no` results; determine whether the harmonic
  approximation is acceptable for target systems.
- [ ] Analyze ZPE leakage in condensed-phase LSC-IVR trajectories: monitor
  mode-energy drift; assess impact on thermal conductivity.
- [ ] Document the applicable regime of LSC-IVR thermal conductivity: specify
  the temperature range and system types (weak/moderate/strong anharmonicity,
  with/without phonon bandgap) where LSC-IVR is reliable.

---

## 2. LSC-IVR New Sampling Features

### 2.1 NVT pre-equilibration + LSC-IVR two-stage workflow (no code change needed)

GPUMD supports multiple consecutive `run` blocks in a single `run.in`, each
with its own `ensemble`. The `Ensemble_QCT::initialize_before_run()` has an
`initialized_` guard that prevents re-initialization on the second `run`.
Therefore, the two-stage NVT → LSC-IVR workflow is achievable with existing
GPUMD functionality:

```text
# Stage 1: NVT equilibration to relax the structure at target temperature
ensemble nvt_lan 300 100
dump_thermo 100
run 50000          # 25 ps at 0.5 fs/step

# Stage 2: LSC-IVR Wigner sampling + NVE production
ensemble lsc_ivr 300 seed 12345 replicas 1 \
    hessian_displacement 0.001 anharmonic_reweighting no
compute_hac 20 500 10
dump_thermo 100
run 1000000        # 500 ps production
```

The NVT stage thermalizes the system from the initial structure (read from
`model.xyz` or a previous `dump_restart` output) toward the target
temperature. The subsequent `ensemble lsc_ivr` block performs Wigner
sampling on the NVT-relaxed structure and switches to NVE + HAC for
production.

- [ ] Validate that `initialized_` guard works correctly when the ensemble
  type changes between `run` blocks (NVT → LSC-IVR): the LSC-IVR ensemble
  must re-initialize because it is a *new* ensemble object, not the same
  one. Confirm that Hessian computation and Wigner sampling are triggered
  on the second `run` block, using the NVT-relaxed positions as input.
- [ ] Test that `dump_restart` from the NVT stage can be used as `model.xyz`
  for the LSC-IVR stage (useful when the two stages are split across jobs).
- [ ] Document the two-stage workflow in `LSC_IVR_KAPPA_GUIDE.md` and
  `docs/lsc_ivr.md` with a complete `run.in` example.
- [ ] Analyze the effect of NVT pre-equilibration duration on HAC: compare
  HAC curves with 0, 10, 25, 50 ps NVT pre-equilibration.

### 2.2 T=0 ground-state Wigner sampling

- [ ] Ensure T=0 Wigner sampling works for non-HAC paths (DOS, spectra,
  correlations via `lsc_ivr.py`): the HAC T=0 guard (QLR-029) must not
  block `dump_qct`-only runs.
- [ ] Add tests and documentation for T=0 ground-state Wigner sampling.

### 2.3 Morse-potential initial-condition sampling

- [ ] For diatomic or single-bond systems, implement analytic Morse Wigner
  distribution as an alternative to harmonic Wigner, eliminating the need
  for anharmonic reweighting for strongly anharmonic vibrations.
- [ ] Target: H₂, OH at high vibrational states where harmonic approximation
  is poor.

### 2.4 Microcanonical Wigner sampling

- [ ] Add fixed-total-energy Wigner sampling: constrain Σ_k E_k = E while
  drawing per-mode energies from the Wigner distribution.
- [ ] Target: QCT scattering experiments where reactants have a defined
  total energy.

---

## 3. LSC-IVR New Propagation Features

### 3.1 Backward trajectory propagation

- [ ] Add backward NVE propagation (reverse momenta → propagate → reverse
  momenta) for symmetric time-correlation functions
  `C(t) = <A(-t/2) B(t/2)>`.
- [ ] Value: improved statistical efficiency; prerequisite for FBTS.

### 3.2 Adaptive timestep

- [ ] Auto-recommend timestep from the highest Hessian frequency; monitor
  energy drift during run; optionally auto-halve timestep on drift threshold.
- [ ] Rationale: high-frequency modes (OH ~115 THz) require ~0.1 fs, but
  low-frequency modes can tolerate much larger steps.

### 3.3 Trajectory termination and failure policies

- [ ] Add configurable termination on: atom loss (non-periodic), NaN forces,
  invalid coordinates, incomplete output, energy drift exceeding threshold.
- [ ] Report termination reason in manifest.

---

## 4. LSC-IVR New Measurement & Observable Features

### 4.1 LSC-IVR NEMD thermal conductivity

- [ ] Enable `compute_hnemd` after LSC-IVR initialization: apply external
  force to generate heat flux, measure κ directly.
- [ ] Handle Wigner weight averaging for NEMD: each replica's heat flux
  contribution is weighted by its Wigner factor.
- [ ] Value: NEMD converges faster for large systems; avoids long Green-Kubo
  correlation times.

### 4.2 Wigner-weighted DOS / VDOS

- [ ] Make `compute_dos` support Wigner-weighted ensemble averaging: each
  replica's velocity autocorrelation is weighted by `wigner_weight`.
- [ ] Value: quantum-corrected vibrational density of states is a core
  LSC-IVR output; currently requires offline post-processing of trajectory
  files.

### 4.3 Wigner-weighted SDC (self-diffusion coefficient)

- [ ] Make `compute_sdc` support Wigner-weighted ensemble averaging.
- [ ] Value: quantum-corrected diffusion coefficients for light atoms
  (H, D) in solids where tunneling may affect diffusion rates.

### 4.4 Wigner-weighted IR spectrum

- [ ] Integrate `dump_dipole` collection into `run_multigpu.py --workflow ir`:
  collect per-replica dipole trajectories, compute Wigner-weighted dipole
  autocorrelation, FFT to IR absorption spectrum.
- [ ] Value: quantum-corrected IR spectrum is the primary LSC-IVR observable
  for molecular spectroscopy.

### 4.5 Multi-operator correlation in `lsc_ivr.py`

- [ ] Support multiple `(A, B)` pairs in a single config; compute all
  correlations in one pass to avoid repeated trajectory file reads.
- [ ] Value: simultaneously compute position-velocity, dipole-dipole,
  bond-length correlations from the same trajectory set.

### 4.6 Mode-resolved quantum correlation functions

- [ ] Add per-mode `(Q_k, P_k)` trajectory output in `dump_qct`; compute
  mode-mode correlations `C_kk(t) = <Q_k(0) Q_k(t)>` in `lsc_ivr.py`.
- [ ] Value: mode-resolved power spectra identify per-mode quantum-corrected
  frequencies and linewidths for direct comparison with IR/Raman experiments.

---

## 5. Multi-GPU & Large-System Optimization

### 5.1 Checkpoint / resume mechanism

- [ ] Implement trajectory-level checkpoint in `run_multigpu.py`: detect
  completed replicas by output completeness and step count; skip finished
  replicas; rerun only incomplete ones.
- [ ] Value: essential for ns-scale production runs where node timeout is
  common.

### 5.2 Performance benchmarking

- [ ] Record wall-clock time and throughput for 方案A at various system sizes
  (216, 512, 1000, 1728 atoms) and GPU counts (1, 2, 4, 8).
- [ ] Evaluate `run_multigpu.py` scheduling efficiency for large trajectory
  counts (>50): task-queue wait time, GPU utilization, I/O bottlenecks.

### 5.3 Binary Hessian format

- [ ] Support binary Hessian/eigenvector format to accelerate loading for
  large systems where the text file is >100 MB.
- [ ] Value: I/O optimization, not a functional change.

---

## 6. QCT Scattering Workflow (Gas-Phase)

*These items are specific to the QCT scattering/molecular-dynamics
application and are lower priority than LSC-IVR thermal conductivity.*

### 6.1 Reproducible baseline

- [ ] Add automated GPU integration tests for phase-point validation, single
  and batch initialization, potential correction, semiclassical angular
  momentum, saddle launches, and deterministic retry behavior.
- [ ] Add a CI-friendly CPU/Python test layer that does not fail during
  pytest collection when optional GPU helper scripts are absent.
- [ ] Add explicit energy-balance tests with nonzero reaction energy and
  nonzero rotational energy in the same phase point.
- [ ] Replace ad hoc remote regression commands with versioned Slurm scripts;
  record GPU model, CUDA version, NEP model, and timestep.

### 6.2 PES and molecular preparation

- [ ] Define and document the supported PES workflow: PES from `potlib`, ASE
  calculator adapter, optimization, frequency calculation, and conversion to
  GPUMD inputs.
- [ ] Add a native Morse pair potential for the two-atom benchmark (specify
  units, cutoff, forces, batch compatibility).
- [ ] Add an end-to-end A + BC → A + B + C dissociation example with
  reference results.
- [ ] Make optimized structure and Hessian/eigenvector provenance explicit in
  every example; avoid silently mixing `model.xyz`, optimized structures,
  and external mode files.

### 6.3 Initial-state sampling

- [ ] Separate harmonic mode sampling from collision-state sampling.
- [ ] Implement and validate thermal, microcanonical, and fixed-state sampling
  for diatomic and polyatomic reactants.
- [ ] Implement diatomic WKB vibrational action sampling; compare with EBK
  `v` sampler.
- [ ] Define phase sampling semantics and tests for random/fixed phase,
  directional momentum, and reaction-coordinate flux sampling.
- [~] Add validation of mass-weighted orthogonality, rigid-mode removal, and
  reaction-mode separation for user-supplied `qct_modes.in` files. (active
  stable-stable and stable-reaction orthogonality validated; rigid-mode
  contamination of user `qct_modes.in` still open)
- [ ] Decide how ZPE leakage is measured and reported during a trajectory;
  initial ZPE inclusion alone is not a ZPE-leakage treatment.

### 6.4 Integrators and propagation

- [ ] Document the current velocity-Verlet/NVE implementation and its valid
  timestep range for the highest sampled frequency.
- [ ] Add timestep-convergence examples for Morse, NEP89 OH, and a saddle
  trajectory.
- [ ] Provide an optional integrator comparison: velocity Verlet vs
  higher-order symplectic methods.
- [ ] Add trajectory termination and failure policies for atom loss, NaN
  forces, invalid coordinates, and incomplete output.

### 6.5 Scattering observables

- [ ] Add collision geometry and impact-parameter sampling with reproducible
  `b` seed and `bmax` search procedure.
- [ ] Compute reaction probability vs collision energy with uncertainty
  estimates.
- [ ] Compute final relative velocity and scattering angle; document
  forward/backward convention.
- [ ] Compute integral cross sections (ICS) from impact-parameter sampling.
- [ ] Compute differential cross sections (DCS) with angular binning,
  normalization, and error bars.
- [ ] Add a multi-energy/multi-impact-parameter job generator and aggregation
  tool.

### 6.6 Final-state analysis

- [ ] Stabilize translation/rotation/vibration decomposition for linear and
  nonlinear fragments, atom loss, and multiple products.
- [ ] Implement histogram binning (HB), Gaussian binning (GB), and
  configurable soft/hard assignment policies.
- [ ] Add state-to-state products and branching statistics.
- [ ] Make ZPE leakage, below-ZPE states, continuous quantum numbers, and
  rejected mode projections separate, auditable outputs.
- [ ] Validate final-state analysis against the A + BC reference workflow.

---

## 7. Uncertainty Quantification & Statistics

### 7.1 Blockwise HAC uncertainty

- [ ] Keep Wigner-initial-condition uncertainty (between replicas) separate
  from finite-trajectory uncertainty (within-replica blocks). Never count
  blocks from one trajectory as independent replicas.
- [ ] Add an explicit blockwise HAC output or raw heat-current contract before
  implementing within-replica block errors. Do not infer blocks by slicing the
  correlation-lag axis of `hac.out`.
- [ ] Validate the workflow with five short independent trajectories before
  launching `5 × 5 ns`. If N_eff < 3, add new seeds instead of treating
  longer blocks as new replicas.

### 7.2 RPMD thermal conductivity comparison

RPMD thermal conductivity and the spring-potential problem are documented
separately in a dedicated analysis document (to be created). This is **not**
part of the LSC-IVR implementation plan; it is a separate research question
about the applicability of RPMD to condensed-phase heat transport. See
`tools/qct/SEMICLASSICAL_ROADMAP.md` Method 2 for the RPMD correlation
framework.

- [ ] Create a standalone document analyzing RPMD spring-potential issues
  in Green-Kubo heat-current autocorrelation, including literature review,
  physical analysis, and (optionally) TRPMD mitigation tests. Do not block
  LSC-IVR development on this.

---

## 8. Automatic Hessian Memory Policy

Dense automatic Hessians scale as `D²`, where `D = 3N`. A `D = 5184` FP64
matrix is ~206 MiB; retaining raw Hessian, mass-weighted matrix, eigenvectors,
solver workspace, force buffers, and host copies simultaneously can exceed GPU
memory.

- [~] Keep only the minimum number of dense device matrices: symmetrize and
mass-weight in place, let the FP64 eigensolver overwrite its input, release
solver workspace immediately, and stream `qct_hessian.out` in blocks. (CUDA
finite-difference/eigensolve path and memory preflight implemented; complete
elimination of host eigenvector copies remains open.)
- [ ] Add an explicit `hessian_memory_policy` with `error` (default), `auto`,
and `gpu` values. The default must report requested and available bytes and
fail rather than silently changing numerical behavior.
- [ ] Implement the `auto` policy as a CPU-resident matrix fallback while
retaining GPU force evaluations. Requires a real host symmetric eigensolver;
the existing GPU Jacobi wrapper is not a valid fallback.
- [ ] Do not claim full CPU execution until a CPU force/PES backend exists.
- [ ] Add memory diagnostics and regression tests for repeated Hessian calls,
multiple concurrent GPUMD processes, QCT `replicas`, NEP neighbor buffers,
and solver workspace failures.
- [ ] For systems where dense storage is intrinsically too large, implement a
matrix-free / block-Lanczos or Davidson path that stores `O(kD)` vectors and
computes only the active low-frequency modes.
- [ ] Implement an exact finite-cutoff sparse Hessian representation without
dropping physical periodic or many-body coupling blocks. Validate sparse
eigenvalues, acoustic modes, Wigner weights, HAC curves, and thermal
conductivity against the dense reference on small periodic cells.
- [ ] Treat numerical sparsification and finite active-mode sampling as
explicit approximations. Add `k = 256, 512, 1024, 2048` convergence tests;
report when thermal-conductivity changes exceed between-replica statistical
error.
- [ ] Keep domain decomposition limited to local reaction/vibration workflows
until global acoustic and long-wavelength heat-transport errors are measured.

---

## 9. Replica Parallelism & Batch Mode

- [ ] Implement resource-aware batch-size estimation from atom count, NEP
neighbor capacity, GPU memory, and measured throughput.
- [ ] Improve batch diagnostics for allocation failures and per-replica
retries.
- [ ] Decide whether batch support should remain limited to ordinary scalar
NEP or be generalized to other pair/neural potentials.
- [ ] Add multi-GPU/MPI replica distribution only after single-GPU batching
is stable and benchmarked; intentionally not part of the current
implementation.

---

## 10. SC-IVR/FBTS Applicability Documentation

*Based on analysis: SC-IVR/FBTS is NOT applicable to condensed-phase
thermal conductivity due to sign problem scaling exponentially with system
size. It is limited to small molecules (N ≤ 20–30) where quantum phase
information (tunneling, interference, resonance) is needed.*

- [ ] Add an applicability section to `SC_IVR_FBTS_PLAN.md`: explicitly
state N ≤ 20–30 limit, applicable observables (vibrational spectra, tunneling
splittings, reaction rates), and inapplicability to condensed-phase thermal
conductivity.
- [ ] Update the decision guide in `SEMICLASSICAL_ROADMAP.md`: route
condensed-phase thermal conductivity to LSC-IVR / RPMD / QTB; restrict
SC-IVR/FBTS to small-molecule gas-phase dynamics.
- [ ] Document a method-applicability matrix for thermal conductivity:

| Method   | Condensed-phase κ | Small-mol spectra | Tunneling/rates |
|----------|:-:|:-:|:-:|
| Classical MD | ✅ baseline | ⚠️ no quantum | ❌ |
| QTB          | ✅ approx. | ⚠️ effective | ❌ |
| LSC-IVR      | ✅ recommended | ✅ | ⚠️ partial |
| RPMD/TRPMD   | ⚠️ spring issue | ✅ | ⚠️ partial |
| SC-IVR/FBTS  | ❌ sign problem | ✅ precise | ✅ |

---

## 11. Documentation & Release Quality

- [ ] Update beginner workflow: publish one complete end-to-end example for
  both LSC-IVR thermal conductivity (Si/NEP89) and QCT scattering.
- [ ] Clean up historical review documents: V1–V5 (5 files); retain V5 as
  authoritative, archive or merge V1–V4.
- [ ] Update all timestamps and baseline commit references (currently
  `b479062c`; actual is `c81d5c33` + uncommitted changes).
- [ ] Integrate QCT/LSC-IVR tools into a `gpumdkit` Python package for
  streamlined installation and CI integration.

---

## P2 Implementation (Completed)

All P2 priority items have been implemented and tested (131 tests pass):

### §3.1 Backward Trajectory Propagation
- [x] `compute_symmetric_correlation()` in `lsc_ivr.py`: computes `C(t) = <A(-t/2) B(t/2)>`
  using trajectory midpoint splitting. Activated via `--symmetric` CLI flag.
- [x] Better statistical properties for symmetric operators.

### §3.2 Adaptive Timestep
- [x] `recommend_timestep()` and `parse_hessian_frequencies()` in `lsc_ivr.py`.
- [x] Recommends MD timestep from highest Hessian frequency with configurable
  steps-per-period (default 20). Activated via `--adaptive-timestep` CLI flag.

### §4.3 Wigner-Weighted SDC Merge
- [x] `merge_sdc()` in `run_multigpu.py`: merges `sdc.out` files with Wigner weights.
- [x] Added `"sdc"` workflow to `_detect_workflow()`, `_required_artifacts()`, and
  main() dispatch.
- [x] Added `sdc.out` to `GENERATED_ARTIFACTS`.

### §4.6 Mode-Resolved Correlations
- [x] `compute_mode_correlations()` and `write_mode_correlations_csv()` in `lsc_ivr.py`.
- [x] Computes per-mode `C_k(t) = <Q_k(0) Q_k(t)>` from eigenvectors and trajectory.
- [x] Activated via `--mode-correlations` and `--mode-masses` CLI flags.

### §7.1 Blockwise HAC Uncertainty
- [x] Block-wise variance estimation added to `merge_hac()` in `run_multigpu.py`.
- [x] Separates Wigner-initial-condition uncertainty (between replicas) from
  finite-trajectory uncertainty (within-replica blocks).
- [x] Extended `hac_uncertainty.csv` with `block_se_*` and `combined_se_*` columns.
- [x] Added `blockwise_uncertainty` section to `hac_merge_manifest.json`.
- [x] Activated via `--block-size` CLI flag.

### §10 SC-IVR Applicability Documentation
- [x] Added applicability section to `SC_IVR_FBTS_PLAN.md`.
- [x] Added method-applicability matrix to `SEMICLASSICAL_ROADMAP.md`.

## Priority Summary

| Priority | Section | Items | Rationale |
|----------|---------|-------|----------|
| **P0** | §1.1–1.2 | Convergence benchmarks, T-dependence | Core scientific output of LSC-IVR |
| **P0** | §2.1 | NVT→LSC-IVR two-stage workflow | No code change; validate multi-run + document |
| **P0** | §4.1 | LSC-IVR NEMD | Faster κ convergence for large systems |
| **P0** | §5.1 | Checkpoint/resume | Essential for ns-scale production |
| **P1** | §1.3–1.4 | Material generality, approximation analysis | Validate beyond Si |
| **P1** | §1.2 | `temperature_sweep` (periodic) | Automate T-dependence studies; Python-only |
| **P1** | §4.2 | Wigner-weighted DOS | Core LSC-IVR observable |
| **P1** | §4.4 | Wigner-weighted IR spectrum | Core molecular spectroscopy |
| **P1** | §4.5 | Multi-operator correlation | Efficiency improvement |
| **P2 ✅** | §3.1–3.2 | Backward propagation, adaptive timestep | Statistical/numerical improvements |
| **P2 ✅** | §4.3 | Wigner-weighted SDC | Quantum diffusion |
| **P2 ✅** | §4.6 | Mode-resolved correlations | Detailed spectral analysis |
| **P2 ✅** | §7.1 | Blockwise HAC uncertainty | Statistical rigor |
| **P2 ✅** | §10 | SC-IVR applicability docs | Prevent misuse |
| **P2** | §2.2–2.4 | T=0, Morse, microcanonical Wigner | Specialized sampling modes (future) |
| **P3** | §5.2–5.3 | Benchmarks, binary Hessian | Optimization |
| **P3** | §6.* | QCT scattering workflow | Gas-phase application, smaller user base |
| **P3** | §8 | Hessian memory policy | Large-system optimization |
| **P3** | §9 | Batch parallelism | Engineering improvement |
| **P3** | §11 | Documentation cleanup | Maintainability |
