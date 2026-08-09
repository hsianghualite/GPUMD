# Semiclassical and Quasi-Classical Dynamics Roadmap

This document records the planned evolution of semiclassical and
quasi-classical dynamics methods in GPUMD, relative to the current `qct`
branch. It is a design reference for future contributors, not a user manual;
it does not describe already-shipped features except where needed for context.

Status reflects the `qct` branch working tree as of 2026-08-05. "Native"
means implemented in the C++ core under `src/`; "tool" means a Python script
under `tools/qct/`; "planned" means not yet implemented.

## Current capabilities

| Method | Native propagation | Native measurement | Tool |
|---|---|---|---|
| Classical QCT (`canonical`, `microcanonical`, `mode_energy`, `semiclassical`) | `ensemble qct` in `src/integrate/ensemble_qct.cu` | `dump_qct` in `src/measure/dump_qct.cu` | `analyze_qct.py`, `compare_batch.py`, `benchmark_batch.py` |
| LSC-IVR / Wigner-Liouville (`wigner`) | `ensemble qct wigner` (reuses QCT NVE engine) | `dump_qct` | `lsc_ivr.py` |
| PIMD / RPMD / TRPMD | `ensemble pimd/rpmd/trpmd` in `src/integrate/ensemble_pimd.cu` | `dump_xyz` + `compute_dos` (per-bead, no centroid reduction) | none specific |
| Harmonic normal modes | `Molecular_Hessian` in `src/phonon/molecular_hessian.cuh` | audit files `qct_stationary.xyz`, `qct_hessian.out`, `qct_eigenvector.out` | `eigenvector_to_qct_modes.py` |
| Dipole on trajectory | `dump_dipole` in `src/measure/dump_dipole.cu` | per-step dipole from a second NEP dipole potential | none specific |
| Velocity / DOS correlations | `compute_dos` in `src/measure/dos.cu` | single-trajectory sliding-window autocorrelation -> DOS | none specific |

The QCT branch and the PIMD/RPMD/TRPMD stack currently develop
independently. They share the force evaluator and the velocity-Verlet core,
but their initial-condition samplers, measurement keywords, and observable
post-processing do not interoperate. The methods below are grouped by how
much of this shared infrastructure they reuse.

## Method 1 — LSC-IVR (linearized semiclassical IVR)

Aliases in the literature: classical Liouville dynamics, Wigner-Liouville
dynamics. The two names refer to the same propagation scheme; LSC-IVR
emphasizes the IVR derivation, Liouville dynamics the phase-space picture.

### What it is

Sample each normal mode from the harmonic Wigner thermal distribution, not
from the classical Boltzmann + ZPE offset used by the current `canonical`
sampler. Propagate classically (NVE velocity-Verlet, already available).
Form the quantum time correlation function as a reweighted average over
independent trajectories:

```
C_AB(t) = < w_i * A(0)_i * B(t)_i > / < w_i >
```

### Relation to existing `canonical` QCT sampling

The current `canonical` branch places each mode on a ring of radius
`sqrt(2 * E_k)` with `E_k = ZPE + classical Boltzmann draw`, then picks a
random phase. LSC-IVR instead draws `(Q_k, P_k)` as independent Gaussians
with variance `(hbar / 2 omega_k) * coth(beta hbar omega_k / 2)`. The
`coth` factor is the quantum correction: it reduces to the classical limit
at high temperature and to the zero-point Gaussian at T = 0.

### Anharmonic reweighting

Because the Wigner sampler uses the harmonic approximation, samples are
reweighted by the ratio of the true Boltzmann weight to the harmonic one:

```
w_i = exp( -beta * [ V(Q0_i) - V_ref - sum_k 0.5 omega_k^2 Q_k^2 ] )
```

`V(Q0_i)` is the true-PES potential at the sampled point, `V_ref` is the
potential at the reference (equilibrium) structure, and the sum is the
harmonic potential relative to the reference. This is the same potential
information already computed by `evaluate_batch_potential_energy` and
`evaluate_potential_energy` in `ensemble_qct.cu`, but used to form a
statistical weight rather than to rescale velocities. The existing
`apply_potential_correction` resampling loop is bypassed for Wigner
samples because every sample is statistically valid; discarding or
rescaling any would bias the distribution.

### Implementation status (complete — 2026-08-05)

- Sampling: a new `wigner` mode in `Sampling_Mode` in
  `src/integrate/ensemble_qct.cuh`, dispatched in `sample_harmonic_point`.
  Reuses the existing mode machinery (`Molecular_Hessian`,
  `read_qct_modes`, `read_gpumd_modes`) and the batch replica framework.
- Reweighting: store `wigner_weight` and `log_wigner_weight` in
  `Sampled_Point`; write both to `qct_initial_summary.csv`.
- Measurement: no new C++ measurement class. `dump_qct` already writes
  per-replica per-step positions, velocities, masses; that is sufficient
  for the v1 operator set.
- Post-processing: a new `tools/qct/lsc_ivr.py` implementing the generic
  `A(0) B(t)` correlation and its FFT, consuming `qct_trajectory.xyz` and
  `qct_initial_summary.csv`. Designed so it can also consume RPMD centroid
  trajectories later (shared post-processing layer).

### Limitations baked into v1

- Stationary minima only; no saddle / flux-side rate constants.
- Operators restricted to functions of positions, velocities, masses, and
  fixed charges (i.e. anything computable from `dump_qct` output).
  Dipole-operator correlation (IR spectrum) requires a per-replica dipole
  dump and is deferred.
- Reweighting variance grows with anharmonicity; the tool reports
  `std_error` so the failure mode is visible. Strongly anharmonic systems
  should use RPMD/TRPMD instead (Method 2).

### When to use

Weakly to moderately anharmonic molecular vibrations, quantum-corrected
DOS, and reaction-rate diagnostics where a single potential energy
surface is adequate. Cheapest entry point into quantum dynamics because it
reuses the QCT single-bead trajectory engine.

## Method 2 — RPMD / TRPMD correlation functions

### What it is

Ring-polymer molecular dynamics already propagates quantum fluctuations
exactly via the bead representation; GPUMD implements the propagation in
`src/integrate/ensemble_pimd.cu`. What is missing is a measurement class
that forms the Kubo time correlation function from centroid (or
single-bead) trajectories, analogously to how `compute_dos` forms the
velocity autocorrelation from a classical trajectory.

### Why this is the upgrade path when LSC-IVR fails

LSC-IVR's Wigner sampling is a harmonic approximation; its reweighting
breaks down for strongly anharmonic systems or when tunneling and
interference matter. RPMD does not make the harmonic approximation: the
bead ring polymer represents the quantum thermal density directly.
TRPMD additionally thermostats the internal modes to suppress the RPMD
resonance artifacts that produce spurious peaks in spectra.

### Implementation scope (planned)

- Propagation: already native (`ensemble rpmd` / `ensemble trpmd`).
- Measurement: a new C++ property class, e.g.
  `src/measure/compute_rpmd_correlation.cu`, modeled on `compute_dos` but
  operating on the centroid or a chosen bead, with sliding time origins.
  Output schema mirrors `dos.out` / `mvac.out`.
- Operators: start with position and velocity autocorrelations (vibration
  spectra, diffusion). Dipole autocorrelation (IR) needs per-bead dipole
  evaluation and is a follow-on.
- Post-processing: the same `lsc_ivr.py` correlation/FFT tool, fed by
  centroid trajectory output instead of `dump_qct`.

### Limitations

- RPMD resonance artifacts; TRPMD smears them at the cost of some
  dynamical rigor. The user must converge bead count.
- Bead count multiplies force evaluations, so this is more expensive than
  LSC-IVR by roughly `num_beads` x.
- No native per-bead or centroid `dump` keyword exists yet; one is needed
  to expose trajectories to the post-processing tool.

### When to use

Strongly anharmonic systems, quantum tunneling, ZPE leakage studies, and
any case where LSC-IVR's `std_error` indicates the reweighting has failed.
Also the reference method against which LSC-IVR results should be
validated.

## Method 3 — SC-IVR / FBTS (semiclassical IVR with forward-backward
trajectory stabilization)

### What it is

Full semiclassical IVR (Herman-Kluk) retains the van Vleck determinant and
Maslov phase along each trajectory. FBTS pairs forward and backward legs
so that the rapidly oscillating phase partially cancels, stabilizing the
average while preserving more quantum coherence than LSC-IVR.

### Relation to LSC-IVR

LSC-IVR is the fully linearized limit of SC-IVR, in which all phase
information is discarded. FBTS sits between the two:

```
classical MD < LSC-IVR < FBTS < full SC-IVR
(no phase)   (no phase)   (partial phase)  (full phase)
```

### Implementation scope (planned, long-term)

- Monodromy (stability) matrix propagation: each trajectory must carry
  the `3N x 3N` Jacobian `d x(t) / d x(0)`, evolved alongside the state.
  This is the dominant new cost and requires `O(N^2)` storage per
  trajectory.
- van Vleck weight and Maslov phase per trajectory, formed from the
  monodromy matrix.
- Forward-backward trajectory pairing and phase stabilization.
- No existing infrastructure in GPUMD for stability-matrix propagation;
  this is the main engineering risk.

### When to use

Systems where quantum interference, tunneling, or resonant energy transfer
matter and where LSC-IVR (even reweighted) and RPMD are insufficient or
impractical. Not a default method; only justified when the cheaper methods
have demonstrably failed.

### Limitations

- Highest cost and complexity of all methods listed here.
- Phase calculations are numerically delicate; convergence with trajectory
  count can be poor.
- Should be validated against RPMD on the same system before being trusted.

## Method 4 — PLDM (partial linearized density matrix)

### What it is

Linearize only the bath degrees of freedom while retaining the full
quantum propagator for a small set of system modes, typically the
electronic states in nonadiabatic dynamics.

### Why it is out of scope for the current architecture

GPUMD's force framework is built around single-surface scalar potentials
(ordinary NEP, TMDP, DFT, pair). PLDM's value is in nonadiabatic dynamics
with multiple coupled potential energy surfaces and nonadiabatic coupling
terms, none of which GPUMD currently represents. Implementing PLDM would
therefore require first adding a multi-surface electronic-structure or
model-Hamiltonian layer, which is a separate and larger effort than any
dynamics method listed here.

### When it would become relevant

Only after GPUMD gains a multi-surface / nonadiabatic capability. For
single-surface vibrational and reaction dynamics, PLDM offers no advantage
over LSC-IVR or RPMD, because there is no system sub-space whose coherence
needs to be preserved.

## Decision guide

```
Is the system on a single potential energy surface?
no  -> PLDM needs a multi-surface layer first (Method 4); not currently viable.
yes -> Do tunneling / interference / resonance matter?
       yes -> SC-IVR / FBTS (Method 3); large effort, use only if cheaper methods fail.
       no  -> Is the system strongly anharmonic?
              yes -> RPMD / TRPMD (Method 2); propagation already native, needs a correlation measurement class.
              no  -> LSC-IVR (Method 1); reuses the QCT engine, cheapest, first to implement.
```

## Shared infrastructure to build once

Several pieces are common to more than one method and should be designed
for reuse rather than reimplemented per method:

1. **Generic A(0)B(t) correlation post-processor** (`tools/qct/lsc_ivr.py`):
   accepts a trajectory with per-replica frames and per-replica weights, so
   it serves LSC-IVR directly and RPMD centroid trajectories with no change
   to the tool.
2. **Wigner / Boltzmann initial-condition sampler**: the harmonic mode
   machinery (`Molecular_Hessian`, `read_qct_modes`,
   `read_gpumd_modes`) is method-agnostic; only the draw rule for
   `(Q_k, P_k)` differs between `canonical` QCT and `wigner` LSC-IVR.
3. **Per-replica dipole dump**: needed by IR-spectrum variants of both
   LSC-IVR and RPMD. Currently `dump_dipole` is incompatible with batch
   runs; a per-replica dipole path is a shared prerequisite.
4. **Potential-energy diagnostic**: `evaluate_batch_potential_energy` and
   `evaluate_potential_energy` already exist; both LSC-IVR reweighting and
   future SC-IVR weight diagnostics depend on them.

## Suggested implementation order

1. **Method 1 (LSC-IVR)** — lowest cost, largest user reach, exercises the
   shared correlation post-processor and the Wigner sampler. Land first.
2. **Method 2 (RPMD/TRPMD correlation)** — propagation already works;
   adding the measurement class and a centroid dump unlocks the strongly
   anharmonic regime and provides the validation reference for Method 1.
3. **Method 3 (SC-IVR / FBTS)** — only if Methods 1 and 2 are shown to be
   insufficient for a target problem; requires stability-matrix
   infrastructure that does not yet exist.
4. **Method 4 (PLDM)** — gated on a separate multi-surface capability
   decision; not part of this roadmap otherwise.

## Large-system strategy: 方案A (process-level parallelism)

### Problem

The QCT batch neighbor list (`find_neighbor_list_qct_batch` in
`src/force/nep.cu`) is O(N²) brute force: for each atom it loops over
all atoms in the same replica. This is fine for small molecules but
prohibitive for condensed-phase systems (> 1000 atoms). Additionally,
the batch path bypasses GPUMD's cell-list neighbor search
(`neighbor.find_neighbor_global`) and the expanded-box machinery.

### Solution: process-level parallelism with `replicas=1`

For large systems, each GPU runs a separate GPUMD process with
`replicas=1`. This forces the standard MD code path
(`compute_large_box` or `compute_small_box`), which uses cell-list
neighbor search with O(N) scaling. Multiple independent trajectories
are obtained by running multiple processes with different seeds.

The `run_multigpu.py` tool orchestrates this:

1. Divides `total-replicas` across available GPUs.
2. Each GPU gets `replicas=1` (or a small number) with a unique seed.
3. Each process runs an independent LSC-IVR NVE trajectory.
4. Results are merged: trajectories, thermo, ZPE, and **HAC**.

### Wigner-weighted HAC merge

For LSC-IVR thermal conductivity (Green-Kubo), the ensemble-averaged
heat current autocorrelation integral must be weighted by each
replica's Wigner factor:

$$\kappa(t) = \frac{\sum_i w_i \, \kappa_i(t)}{\sum_i w_i}$$

The `merge_hac()` function in `run_multigpu.py` implements this. When
`anharmonic_reweighting no` is set (all $w_i = 1$), it reduces to a
simple average, which is the classical limit.

### Comparison with batch mode

| Feature | 方案A (process-level) | Batch mode (replicas>1) |
|---|---|---|
| Neighbor list | Cell-list, O(N) | Brute-force, O(N²) |
| PBC support | Full (standard MD path) | Limited (no expanded box) |
| Max system size | Unlimited (cell-list) | ~100 atoms (MN limit) |
| Multi-GPU | Yes (one process per GPU) | No (single GPU only) |
| HAC merge | Wigner-weighted (automatic) | N/A (single HAC) |
| Best for | Condensed phase, large systems | Small molecules, gas phase |

### What this does NOT require

- No spatial decomposition in C++ (no multi-GPU NEP).
- No MPI (process-level parallelism only).
- No changes to the force evaluator or neighbor list.
- The only C++ requirement is that `replicas=1` + PBC works correctly,
  which it does: the standard MD path handles PBC natively.
