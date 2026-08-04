# Native QCT TODO

Status is based on `qct` commit `c8208013`.

## Current Scope

The branch currently provides a native harmonic QCT initializer and NVE
propagation. It supports `phase_point`, canonical harmonic sampling,
microcanonical sampling, single-mode energy sampling, diatomic semiclassical
(`v`, `J`) sampling, automatic finite-difference Hessians, external
`qct_modes.in` or `eigenvector.out`, first-order saddle launches, and native
single-GPU replica batching with ordinary scalar NEP.

The following work remains before this is a complete small-molecule
scattering/QCT workflow.

## 1. Restore A Reproducible Baseline

- [x] Restore or implement the missing `tools/qct/compare_batch.py` and
  `tools/qct/benchmark_batch.py` referenced by the README and tests.
- [ ] Add automated GPU integration tests for phase-point validation, single
  and batch initialization, potential correction, semiclassical angular
  momentum, saddle launches, and deterministic retry behavior.
- [ ] Add a CI-friendly CPU/Python test layer that does not fail during pytest
  collection when optional GPU helper scripts are absent.
- [ ] Add explicit energy-balance tests with nonzero reaction energy and
  nonzero rotational energy in the same phase point.
- [ ] Replace the current ad hoc remote regression commands with versioned
  Slurm scripts and record the tested GPU model, CUDA version, NEP model, and
  timestep.

## 2. PES And Molecular Preparation

- [ ] Define and document the supported PES workflow: PES from `potlib`, ASE
  calculator adapter, optimization, frequency calculation, and conversion to
  GPUMD inputs.
- [ ] Add a native Morse pair potential or a clearly documented Morse test
  potential for the two-atom benchmark. The implementation must specify units,
  cutoff behavior, forces, and whether the potential is suitable for batch
  QCT.
- [ ] Add an end-to-end A + BC -> A + B + C dissociation example using a
  single reaction channel and reference results from VENUSpy.
- [ ] Make the optimized structure and Hessian/eigenvector provenance explicit
  in every example; avoid silently mixing `model.xyz`, optimized structures,
  and external mode files.

## 3. Initial-State Sampling

- [ ] Separate harmonic mode sampling from collision-state sampling. The
  current canonical mode sampler is not a thermal distribution of molecular
  translation, rotation, and vibration for a scattering experiment.
- [ ] Implement and validate thermal, microcanonical, and fixed-state sampling
  for diatomic and polyatomic reactants.
- [ ] Implement diatomic WKB vibrational action sampling and compare it with
  the current EBK/semi-classical `v` sampler.
- [ ] Define phase sampling semantics and tests for random phase, fixed phase,
  directional momentum, and reaction-coordinate flux sampling.
- [~] Add validation of mass-weighted orthogonality, rigid-mode removal, and
  reaction-mode separation for user-supplied `qct_modes.in` files. (active stable-stable and stable-reaction orthogonality now validated; rigid-mode contamination of user `qct_modes.in` still open)
- [ ] Decide how ZPE leakage is measured and reported during a trajectory;
  initial ZPE inclusion alone is not a ZPE-leakage treatment.

## 4. Integrators And Propagation

- [ ] Document the current velocity-Verlet/NVE implementation and its valid
  timestep range for the highest sampled frequency.
- [ ] Add timestep-convergence examples for Morse, NEP89 OH, and a saddle
  trajectory, including energy drift and recrossing sensitivity.
- [ ] Provide an optional integrator comparison study for velocity Verlet,
  higher-order symplectic methods, and any integrators already available in
  GPUMD that are physically appropriate for QCT.
- [ ] Add trajectory termination and failure policies for atom loss, invalid
  coordinates, NaN forces, and incomplete output.

## 5. Scattering Observables

- [ ] Add collision geometry and impact-parameter sampling, including a
  reproducible `b` seed and a documented `bmax` search procedure.
- [ ] Compute reaction probability versus collision energy, with uncertainty
  estimates and a clear definition of a reactive trajectory.
- [ ] Compute final relative velocity and scattering angle, and document the
  forward/backward convention.
- [ ] Compute integral cross sections (ICS) from impact-parameter sampling.
- [ ] Compute differential cross sections (DCS), including angular binning,
  normalization, and statistical error bars.
- [ ] Add a multi-energy/multi-impact-parameter job generator and an
  aggregation tool that records failed, unfinished, and valid trajectories.

## 6. Final-State Analysis

- [ ] Stabilize the translation/rotation/vibration decomposition for linear
  and nonlinear fragments, atom loss, and multiple products.
- [ ] Implement explicit histogram binning (HB), Gaussian binning (GB), and
  configurable soft/hard assignment policies.
- [ ] Add state-to-state products and branching statistics for the final
  vibrational and rotational quantum numbers.
- [ ] Make ZPE leakage, below-ZPE states, continuous quantum numbers, and
  rejected mode projections separate, auditable outputs.
- [ ] Validate final-state analysis against the A + BC reference workflow and
  published H. W. Song state-to-state conventions where applicable.

## 7. Replica Parallelism

- [ ] Implement resource-aware batch-size estimation from atom count, NEP
  neighbor capacity, GPU memory, and measured throughput.
- [ ] Improve batch diagnostics for allocation failures and per-replica
  retries.
- [ ] Decide whether batch support should remain limited to ordinary scalar NEP
  or be generalized to other pair/neural potentials.
- [ ] Add multi-GPU/MPI replica distribution only after single-GPU batching is
  stable and benchmarked; this is intentionally not part of the current
  implementation.

## 8. Documentation And Release Quality

- [x] Keep command examples synchronized with files that actually exist in
  `tools/qct/` and `tests/gpumd/qct/`.
- [~] Document output schemas for `qct_initial.out`, `qct_initial.xyz`,
  `qct_initial_summary.csv`, `qct_trajectory.xyz`, and `qct_thermo.csv`. (`kinetic_temperature_K` documented as raw diagnostic; full schema doc still partial)
- [ ] Add a short limitations section to the main QCT input documentation,
  especially for scalar-NEP batching, NVE-only propagation, and saddle
  topology checks.
- [ ] Publish one complete beginner workflow: PES -> ASE optimization/frequency
  -> sampling -> QCT propagation -> reaction classification -> scattering
  statistics -> final-state analysis.
