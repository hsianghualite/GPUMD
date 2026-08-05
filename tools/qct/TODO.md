# QCT Development TODO

Last updated: 2026-07-23

This document converts the current code review and the small-molecule
scattering roadmap into an ordered implementation backlog. Detailed evidence
for existing defects is recorded in `tools/qct/REVIEW_ISSUES.md`.

## Target

The long-term target is a native GPUMD QCT workflow that can:

1. Generate physically defined `A + BC` scattering initial conditions.
2. Propagate many independent trajectories efficiently on one GPU.
3. Classify reaction and dissociation outcomes.
4. Calculate reaction probabilities, scattering angles, integral cross
   sections (ICS), and differential cross sections (DCS).
5. Support progressively more complete initial-state and final-state
   quantization methods.

The first production target is one reference dissociation system:

```text
A + BC -> A + B + C
```

Use one fixed NEP/PES, one reaction channel, and the existing reference
workflow and results. POTLIB, ASE, VenusPy, VENUS96C, and ANT are reference
implementations; the first native GPUMD milestone does not require a general
POTLIB interface.

## Scope Decisions

- [x] Use NEP89 for the current examples and native GPU propagation.
- [x] Prioritize multiple replicas on one GPU.
- [x] Keep recrossing as a normal trajectory outcome rather than an automatic
      failure.
- [ ] Freeze the exact reference `A + BC` system, PES version, units, and
      reference dataset.
- [ ] Freeze the scattering-angle definition for three-body dissociation.
- [ ] Decide whether the reported angle is for atom A, atom B, atom C, or a
      specified Jacobi recoil vector.
- [ ] Define the supported GPUMD input syntax before implementing the native
      scattering sampler.
- [ ] Keep multi-GPU execution deferred until the single-GPU workflow is
      physically complete and tested.

## Phase 0: Fix Existing Physical Correctness

These items block trustworthy scattering development.

### P0.1 Unify single and batch initialization

- [ ] Remove the duplicated single-replica harmonic sampler in
      `src/integrate/ensemble_qct.cu`.
- [ ] Route `replicas 1` and `replicas > 1` through the same
      `sample_harmonic_point` implementation.
- [ ] Route both paths through the same real-potential correction and retry
      logic.
- [ ] Define deterministic retry seeds for every replica, including replica 0.
- [ ] Use one `qct_initial.out` schema for single and batch runs.
- [ ] Write `qct_initial_summary.csv` for a single replica as well.
- [ ] Add a zero-phase anharmonic regression test.
- [ ] Add a single-versus-batch replica-zero sampling equivalence test.

Related review issue: QCT-003.

Acceptance criteria:

- A positive target kinetic energy with zero current kinetic energy is never
  silently accepted.
- A negative target kinetic energy is handled identically for one and many
  replicas.
- Accepted initial total energy equals the requested energy within the selected
  tolerance.

### P0.2 Preserve semiclassical rotational angular momentum

- [ ] Decompose initial velocity into vibrational, rotational, reaction, and
      center-of-mass components.
- [ ] Exclude rotational velocity from stable-mode rescaling.
- [ ] Preserve `sqrt(J(J+1)) * hbar` through real-potential correction.
- [ ] Reject or resample a point when vibrational kinetic correction is not
      possible without changing `J`.
- [ ] Output requested and reconstructed angular momentum.
- [ ] Reconstruct angular momentum from every `qct_initial.xyz` replica in the
      integration test.
- [ ] Test `J=0`, `J=1`, and at least one larger `J`.

Related review issue: QCT-001.

Acceptance criteria:

- Cartesian angular momentum agrees with the requested semiclassical value.
- Recorded rotational energy agrees with Cartesian rotational energy.
- The check remains valid when `stable_velocity_scale` differs from one.

### P0.3 Require real phase-point velocities

- [ ] Preserve `has_velocity_in_xyz` after model parsing.
- [ ] Reject `ensemble qct phase_point` when `vel` is absent.
- [ ] Reject non-finite phase-point positions, masses, or velocities.
- [ ] Preserve source `Seed` and `Replica` metadata when replaying a generated
      phase point.
- [ ] Do not emit invented seed or replica metadata when provenance is unknown.
- [ ] Add missing-velocity, valid-replay, and metadata-replay tests.

Related review issues: QCT-002 and QCT-012.

### P0.4 Make reaction-coordinate sign deterministic

- [ ] Orient molecular Hessian eigenvectors before writing
      `qct_eigenvector.out`.
- [ ] Share the same orientation rule with external-eigenvector sampling and
      post-processing.
- [ ] Verify that `reaction_direction positive` gives positive initial
      `P_rxn` in the analyzer.
- [ ] Verify the corresponding negative-direction case.

Related review issue: QCT-004.

## Phase 1: Stabilize Normal Modes and Input Contracts

### P1.1 Validate the normal-mode basis

- [ ] Check pairwise orthogonality of active stable modes.
- [ ] Check orthogonality between stable modes and the reaction mode.
- [ ] Document the tolerance and mass-weighted convention.
- [ ] Reject duplicate, collinear, and incomplete mode bases with clear errors.
- [ ] Add invalid `qct_modes.in` tests.

Related review issue: QCT-005.

### P1.2 Settle microcanonical energy semantics

- [ ] Decide whether `energy E` means stable-mode energy or total sampled
      energy including reaction-coordinate energy.
- [ ] Update both single and batch implementations to the selected definition.
- [ ] Update the manual and output column descriptions.
- [ ] Report stable, reaction, rotational, and total sampled energies
      separately.
- [ ] Add saddle tests that assert every component, not only total energy.

Related review issue: QCT-006.

### P1.3 Make conversion tools saddle-aware

- [ ] Replace index-based rigid-mode exclusion with absolute-frequency
      selection.
- [ ] Support five rigid modes for linear molecules and six for nonlinear
      molecules.
- [ ] Preserve exactly one significant imaginary reaction mode.
- [ ] Reject second- and higher-order saddles unless explicitly supported.
- [ ] Add minimum, linear molecule, and first-order saddle tests.

Related review issue: QCT-007.

### P1.4 Clarify sampling-specific options

- [ ] Reject `zpe no` for semiclassical EBK sampling, or implement documented
      alternative semantics.
- [ ] Reject `temperature`, `energy`, `mode`, `v`, and `J` when they are not
      meaningful for the selected sampling method.
- [ ] Clarify whether `phase zero` is diagnostic-only.
- [ ] Label pre-correction and post-correction modal quantities explicitly.

Related review issues: QCT-015 and QCT-016.

## Phase 2: Reference Workflow and Coordinate Conventions

This phase should be completed before writing the native scattering sampler.

### P2.1 Freeze reference data

- [ ] Select one POTLIB PES and record its version, citation, license, energy
      zero, length unit, energy unit, and atom ordering.
- [ ] Wrap the PES as an ASE Calculator for reference calculations.
- [ ] Validate analytic forces against finite differences.
- [ ] Run and archive ASE optimization and frequency calculations.
- [ ] Reproduce the same calculation with VenusPy.
- [ ] Archive the senior-group workflow output as immutable reference data.
- [ ] Record acceptable numeric tolerances for energy, force, frequency,
      trajectory, and statistical observables.

### P2.2 Define Jacobi coordinates

- [ ] Implement Cartesian-to-Jacobi and Jacobi-to-Cartesian conversion for
      `A + BC`.
- [ ] Define the incoming relative vector and velocity sign convention.
- [ ] Define impact-parameter vector orientation.
- [ ] Define the initial separation criterion.
- [ ] Define the diatomic bond vector, rotational angular momentum, and phase
      conventions.
- [ ] Verify center-of-mass position and velocity are exactly removed.
- [ ] Add round-trip coordinate tests and analytic three-body examples.

### P2.3 Define trajectory outcomes

- [ ] Freeze bond, dissociation, and asymptotic separation thresholds.
- [ ] Include radial-velocity checks so a temporarily stretched bond is not
      classified as dissociated.
- [ ] Define `reacted`, `unreacted`, `dissociated`, `recrossed`, and
      `unfinished` statuses.
- [ ] Define maximum propagation time and early-stop conditions.
- [ ] Reproduce reference labels for every supplied trajectory.

## Phase 3: Native `A + BC` Scattering Initial Conditions

### P3.1 Add a scattering input model

- [ ] Add a dedicated scattering input file or an explicit `ensemble qct`
      subcommand; do not overload stationary-point normal-mode options.
- [ ] Represent atom groups A and BC explicitly.
- [ ] Parse collision energy `Ec`, vibrational state `v`, rotational state `J`,
      impact parameter `b`, initial separation, seed, and replica count.
- [ ] Record every sampled variable in a per-replica initial summary.
- [ ] Reject periodic boundaries for the first implementation.

Candidate conceptual input:

```text
ensemble qct scattering reactant_a 0 reactant_bc 1 2 \
  collision_energy 1.0 v 0 J 0 impact_parameter 2.0 \
  separation 12.0 seed 123 replicas 1024
```

The exact syntax remains a design decision.

### P3.2 Sample fixed `Ec`, `v`, `J`, and `b`

- [ ] Convert `Ec` to relative translational momentum using the A/BC reduced
      mass.
- [ ] Place the incoming trajectory at the requested impact parameter without
      adding center-of-mass momentum.
- [ ] Sample diatomic vibrational phase.
- [ ] Construct rotational angular momentum with the requested magnitude.
- [ ] Sample molecular orientation and rotational phase with the correct
      isotropic measure.
- [ ] Preserve total linear momentum, total angular momentum, and requested
      initial energy.
- [ ] Compare sampled distributions against VenusPy.

### P3.3 Support batched external phase points

- [ ] Allow a multi-frame extxyz file to provide one phase point per replica.
- [ ] Preserve each frame's seed and source ID.
- [ ] Validate identical atom count, symbols, masses, box convention, and PES
      compatibility across frames.
- [ ] Expand the native NEP batch from supplied frames without harmonic
      resampling.
- [ ] Add batch-versus-independent propagation tests for supplied frames.

This capability is required even if initial sampling remains in VenusPy during
early development.

## Phase 4: Propagation and Integrator Validation

### P4.1 Harden velocity Verlet

- [x] Use GPU velocity Verlet for QCT NVE propagation.
- [ ] Add a scattering-specific energy-conservation test across representative
      collision energies.
- [ ] Establish a default time-step convergence protocol.
- [ ] Check forward propagation followed by velocity reversal.
- [ ] Record per-replica maximum energy deviation, energy range, and final
      drift.
- [ ] Verify reaction probability and ICS convergence with time step, not only
      single-trajectory energy drift.

### P4.2 Educational alternative integrators

These are useful for teaching and reference validation; they do not need to be
production GPU integrators initially.

- [ ] Implement Python velocity Verlet against the same PES wrapper.
- [ ] Implement Python RK4 for comparison.
- [ ] Compare trajectory error, energy drift, reversibility, and cost.
- [ ] Demonstrate why a small local trajectory error does not guarantee stable
      long-time Hamiltonian behavior.
- [ ] Decide later whether any alternative belongs in native GPUMD.

## Phase 5: Reaction Probability and Cross Sections

### P5.1 Compute `P(Ec, b)`

- [ ] Group completed trajectories by collision energy and impact parameter.
- [ ] Exclude or separately report unfinished trajectories.
- [ ] Calculate reaction/dissociation probability.
- [ ] Report trajectory counts and binomial confidence intervals.
- [ ] Keep random seeds and failed replicas auditable.
- [ ] Match the reference probability-versus-energy curves.

### P5.2 Determine `bmax`

- [ ] Implement coarse impact-parameter scanning.
- [ ] Refine the boundary where reaction probability becomes negligible.
- [ ] Define a statistical stopping criterion rather than relying on one
      non-reactive batch.
- [ ] Check that extending the scan changes ICS by less than the target
      tolerance.
- [ ] Store `bmax(Ec)` with uncertainty and sampling provenance.

### P5.3 Calculate ICS

- [ ] Support quadrature over fixed impact-parameter bins:

  ```text
  sigma(Ec) = 2 pi integral_0^bmax b P(Ec,b) db
  ```

- [ ] Support Monte Carlo sampling with
      `p(b) = 2b / bmax^2`:

  ```text
  sigma(Ec) = pi bmax^2 N_react / N_total
  ```

- [ ] Reject unweighted use of uniformly sampled `b`.
- [ ] Propagate statistical uncertainty into ICS.
- [ ] Verify fixed-bin and Monte Carlo estimates agree.

### P5.4 Calculate scattering angles and DCS

- [ ] Implement the frozen recoil/Jacobi angle definition.
- [ ] Calculate angles in the center-of-mass frame.
- [ ] Histogram in `cos(theta)` or apply the correct solid-angle Jacobian.
- [ ] Label forward and backward scattering explicitly.
- [ ] Normalize DCS so its solid-angle integral reproduces ICS.
- [ ] Report angular-bin statistical uncertainties.
- [ ] Compare with the reference angular distribution.

## Phase 6: Advanced Initial-State Sampling

### P6.1 Thermal sampling

- [ ] Define which quantities are thermalized: collision energy, vibration,
      rotation, orientation, or all of them.
- [ ] Implement the correct translational flux distribution rather than a raw
      Maxwell speed distribution when modeling collision rates.
- [ ] Implement rotational-state or classical angular-momentum sampling.
- [ ] Validate sampled histograms and moments analytically.

### P6.2 Microcanonical scattering sampling

- [ ] Define total available energy and energy-zero convention.
- [ ] Sample translational, rotational, and vibrational phase space with the
      intended measure.
- [ ] Preserve total energy and angular momentum per trajectory.
- [ ] Compare distributions against VenusPy and VENUS96C.

### P6.3 Diatomic WKB sampling

- [ ] Extract the one-dimensional diatomic potential from the selected PES or
      reference diatomic curve.
- [ ] Include the centrifugal effective potential:

  ```text
  Veff(r) = V(r) + L^2 / (2 mu r^2)
  ```

- [ ] Locate inner and outer turning points robustly.
- [ ] Solve the WKB/EBK action condition for the requested `(v, J)`.
- [ ] Sample orbital phase with the correct time measure along the anharmonic
      orbit.
- [ ] Validate sampled action, energy, turning points, and radial distribution.
- [ ] Compare harmonic, Morse, and numerical-PES results.

## Phase 7: Final-State Analysis

This phase is valuable but must not block the first correct ICS/DCS workflow.

### P7.1 Classical energy decomposition

- [x] Implement basic center-of-mass, fragment translation, rotation, and
      vibrational kinetic decomposition in `analyze_qct.py`.
- [ ] Unwrap periodic fragments or reject periodic final-state analysis.
- [ ] Add full internal energy relative to product reference minima.
- [ ] Verify energy closure for every trajectory.
- [ ] Add analytic diatomic and polyatomic decomposition tests.

### P7.2 ZPE leakage policy

- [x] Report basic `below_zpe` diagnostics for projected product modes.
- [ ] Define hard ZPE rejection.
- [ ] Define soft ZPE weighting.
- [ ] Report raw and ZPE-adjusted observables side by side.
- [ ] Never silently discard leaking trajectories.
- [ ] Compare policies against the selected literature reference.

### P7.3 State assignment

- [x] Produce preliminary standard-bin and Gaussian-bin weights for harmonic
      product modes.
- [ ] Validate histogram binning (HB) against analytic examples.
- [ ] Validate Gaussian binning (GB), including width dependence and
      normalization.
- [ ] Handle degenerate modes and linear products.
- [ ] Define rotational quantum-number assignment.
- [ ] Calculate state-to-state probabilities and cross sections.
- [ ] Compare with the methodology used in the selected H. W. Song papers.

## Phase 8: Output, Analysis, and Runtime Hardening

### P8.1 Output lifecycle

- [ ] Reject identical trajectory and thermo filenames.
- [ ] Define append behavior across multiple `run` commands.
- [ ] Add a global monotonically increasing step to appended output.
- [ ] Include `pbc` and `Lattice` in `qct_stationary.xyz`, or mark it as
      analysis-only.
- [ ] Use explicit names for kinetic temperature and its degrees of freedom.

Related review issues: QCT-014, QCT-017, and QCT-022.

### P8.2 Stream batch analysis

- [ ] Replace whole-file extxyz loading with a frame iterator.
- [ ] Avoid retaining the original text after parsing.
- [ ] Process per-replica bond signatures incrementally.
- [ ] Cache configuration, stationary structures, and eigenvectors once.
- [ ] Add a peak-memory benchmark for long trajectories.
- [ ] Fix the missing-initial-file fallback.
- [ ] Reject non-replica thermo data for batch trajectories.

Related review issues: QCT-008, QCT-010, and QCT-011.

### P8.3 Strengthen validation tools

- [ ] Compare `Step`, `Time`, `Seed`, and source replica metadata in batch
      equivalence tests.
- [ ] Use numeric-aware replica sorting.
- [ ] Define crossing and recrossing counts separately.
- [ ] Validate complete initial-frame and thermo coverage.
- [ ] Add malformed and periodic extxyz tests.

Related review issues: QCT-018, QCT-019, QCT-020, and QCT-023.

### P8.4 Guard process lifecycle

- [ ] Reject unsupported ensemble transitions after native batch expansion, or
      restore the original atom and force layout completely.
- [ ] Move known unsupported-operation validation before Hessian calculation
      and GPU allocation.
- [ ] Test two consecutive QCT runs and a QCT-to-NVE transition.

Related review issues: QCT-009 and QCT-021.

### P8.5 Correct GPU capacity benchmarking

- [ ] Associate process memory with GPU UUID/index.
- [ ] Respect `CUDA_VISIBLE_DEVICES` and scheduler device isolation.
- [ ] Separate process allocation from pre-existing device usage.
- [ ] Report initialization peak and steady propagation memory separately.
- [ ] Add mocked multi-GPU `nvidia-smi` tests.

Related review issue: QCT-013.

## Deferred Work

- [ ] Multi-GPU replica distribution.
- [ ] MPI scheduling and result aggregation.
- [ ] General POTLIB runtime loading inside GPUMD.
- [ ] Thermostatted or barostatted QCT propagation.
- [ ] Polyatomic semiclassical action-angle sampling beyond harmonic normal
      modes.
- [ ] Automated rare-event or adaptive impact-parameter sampling.

These items should remain deferred until the single-GPU `A + BC` reference
workflow passes all physical and statistical acceptance tests.

## Immediate Next Sprint

1. [ ] Fix QCT-001 and add Cartesian angular-momentum validation.
2. [ ] Fix QCT-002 and add missing-velocity rejection.
3. [ ] Refactor single initialization to fix QCT-003.
4. [ ] Fix QCT-004 reaction-vector orientation.
5. [ ] Freeze the reference `A + BC` system and scattering-angle convention.
6. [ ] Design the multi-frame external phase-point batch input.

Do not start ICS/DCS implementation until items 1-5 are complete. Cross
sections built from physically incorrect initial conditions are not useful
validation targets.
