# QCT tools

## Engineering status

The current `qct` branch is still under engineering review and has known
memory-safety, replica-isolation, periodic-mode, correlation-statistics, and
multi-GPU orchestration blockers. Before using QCT or LSC-IVR results for
production science, read:

- [QCT and LSC-IVR Comprehensive Code Review V3](QCT_LSC_IVR_CODE_REVIEW_V3.md)
- [QCT and LSC-IVR Repair Plan](QCT_LSC_IVR_FIX_PLAN.md)
- [LSC-IVR Thermal Conductivity Guide](LSC_IVR_KAPPA_GUIDE.md)

The older [Native QCT Review Issues](REVIEW_ISSUES.md) and
[LSC-IVR Comprehensive Code Review V2](LSC_IVR_CODE_REVIEW_V2.md) are retained
as historical records. Their current-status conclusions are superseded by the
V3 review.

## Trajectory analysis

`analyze_qct.py` requires Python 3.10 or newer and NumPy. For a single-replica
production trajectory, write positions, masses, and velocities into the same
extxyz file:

```text
dump_xyz -1 0 100 trajectory.xyz mass velocity
dump_thermo 10
```

Analyze one replica with:

```bash
python3 tools/qct/analyze_qct.py \
  --trajectory replica_0001/trajectory.xyz \
  --config qct_analysis.json \
  --replica 0001
```

The analyzer uses `qct_initial.xyz` beside the trajectory as the sampled phase
point when available, while `bond_reference` defines the expected optimized
reactant topology. It writes `qct_summary.csv` and, when a product-mode
projection is valid, `qct_modes_final.csv`.

For a native single-GPU batch run, use the batch-aware output instead:

```text
ensemble qct canonical modes qct_modes.in temperature 300 seed 123 replicas 128
dump_qct 100
```

`qct_trajectory.xyz` contains one extxyz frame for each replica at every dump
step. Each frame has `Replica`, `Step`, and `Seed` metadata. The companion
`qct_thermo.csv` stores one energy and raw kinetic-temperature row per replica and step. The `kinetic_temperature_K` column is `2K/(3N k_B)` over all atoms and is a kinetic diagnostic, not a canonical molecular temperature, because it includes center-of-mass and rotational motion.
The batch path accepts `dump_qct` and `compute_hac`. For HAC, GPUMD first
reduces each replica independently, then applies normalized Wigner weights;
`hac_replica.out` and `hac_reweighting.csv` preserve the per-replica audit.

To compare a batch trajectory against independent runs with the same seeds:

```bash
python3 tools/qct/compare_batch.py \
  --batch qct_trajectory.xyz \
  --reference 0=replica0/qct_trajectory.xyz \
  --reference 1=replica1/qct_trajectory.xyz
```

To scan the useful batch size on one GPU, use a template directory with an
`ensemble qct ... replicas N` line:

```bash
python3 tools/qct/benchmark_batch.py \
  --template tests/gpumd/qct_nep89_oh/batch \
  --gpumd src/gpumd \
  --replicas 1 2 4 8 16 32 \
  --steps 10000 \
  --no-output \
  --output qct_batch_benchmark.csv
```

The benchmark preserves relative potential paths in isolated temporary
directories, records replica-step throughput, and samples visible GPU memory
when `nvidia-smi` exposes the running process. By default it recommends the
smallest successful batch reaching 95% of the measured peak throughput; keep
a longer `run` in the template for stable measurements and extend the scan if
the measured peak is at the largest candidate.

Harmonic initialization also writes `qct_initial_summary.csv` with the
accepted seed, sampled energy, real-potential correction, and stable-mode
velocity scale for every replica. This scale applies only to vibrational
velocity; saddle reaction momentum and semiclassical rotational angular
momentum are preserved. Use it with the first `qct_thermo.csv` row
to audit initial energy balance. `qct_initial.xyz` includes `pbc` and
`Lattice`, so an accepted frame can be used directly as `model.xyz` with
`ensemble qct phase_point`.

QCT propagation is NVE, but energy conservation still depends on the time
step. Test convergence using per-replica total-energy drift, especially for
high-energy trajectories and modes with large frequencies.

For an automatic-Hessian saddle launch, keep the QCT audit files beside the
trajectory and add the saddle settings to the analysis configuration:

```json
{
  "stationary_point": "saddle",
  "stationary_structure": "qct_stationary.xyz",
  "reaction_eigenvector": "qct_eigenvector.out",
  "reaction_coordinate_deadband_sqrt_amu_A": 0.05,
  "reaction_coordinate_hysteresis_sqrt_amu_A": 0.02
}
```

The analyzer projects every frame onto the unique imaginary mode, writes
`qct_reaction_coordinate.csv`, and adds `Q_rxn`, `P_rxn`, final side, crossing
count, and recrossing count to `qct_summary.csv`. The initial dividing-surface
topology is reported but is not rejected as an invalid sampled reactant. Final
reaction channels still require a persistent final bond topology.

An example configuration is:

```json
{
  "atom_index_base": 0,
  "time_step_fs": 0.1,
  "dump_interval": 100,
  "bond_reference": "reactants/optimized.xyz",
  "bond_scale": 1.25,
  "bond_cutoffs_A": {"C-H": 1.35},
  "persistence_frames": 5,
  "energy_drift_tolerance_eV": 0.05,
  "channels": [
    {
      "name": "H_loss",
      "broken_bonds": [[0, 5]],
      "fragments": ["C2H5O", "H"],
      "exact_bond_changes": true
    }
  ],
  "product_reference_energies_eV": {
    "H_loss": -40.0
  },
  "product_modes": [
    {
      "name": "C2H5O",
      "channels": ["H_loss"],
      "atoms": [0, 1, 2, 3, 4, 6, 7, 8],
      "reference": "products/C2H5O/optimized.xyz",
      "eigenvector": "products/C2H5O/eigenvector.out",
      "exclude_lowest": 6,
      "min_frequency": 0.001,
      "gaussian_width": 0.05,
      "max_rmsd_A": 0.3
    }
  ]
}
```

Atom indices use `atom_index_base`, which defaults to zero. A channel must
contain at least one of `formed_bonds`, `broken_bonds`, or `fragments`.
Configured bond changes are subset matches unless `exact_bond_changes` is
true. Unconfigured final graphs are retained under automatically generated
`product:*` or `dissociated:*` channel names.

When `bond_reference` is provided, its optimized structure defines the
expected reactant graph. The sampled `qct_initial.xyz` is checked against that
graph. A trajectory whose harmonic sample already changes connectivity is
marked `invalid_initial_topology` and must not enter reaction statistics.
For saddle analysis, this check is informational because the dividing-surface
structure is not required to have the topology of a separated reactant.

Product mode projection removes center-of-mass translation and rigid rotation,
mass-aligns the final fragment to its optimized reference, and reports harmonic
mode energies, continuous quantum numbers, standard bins, Gaussian-bin
weights, and zero-point-energy leakage flags. Each product needs its own
optimized reference and GPUMD `eigenvector.out`.
The projection is rejected when the mass-weighted aligned RMSD exceeds
`max_rmsd_A` (default 0.3 Angstrom), because harmonic mode energies are not
meaningful outside the neighborhood of the selected product minimum.

## Ensemble aggregation

Each replica should write its summary in its own directory. Merge only after
all jobs have finished:

```bash
python3 tools/qct/merge_qct_results.py \
  replica_*/qct_summary.csv \
  --output qct_ensemble.csv
```

The merger rejects duplicate replica identifiers and reports status counts,
channel counts, and branching fractions over valid (`completed`) trajectories.

## LSC-IVR (Linearized Semiclassical Initial Value Representation)

LSC-IVR is a semiclassical dynamics method that captures quantum
effects—zero-point energy, tunneling, and thermal quantum
fluctuations—within a classical trajectory framework.  GPUMD implements
it as a dedicated `lsc_ivr` ensemble (internally using the QCT engine's `wigner` mode): initial conditions are
drawn from the harmonic Wigner thermal distribution, propagated with
ordinary NVE dynamics, and quantum-corrected time-correlation functions
are computed through anharmonic reweighting.

### Quick start

```
potential    nep.txt
time_step    0.1
ensemble     lsc_ivr 300 seed 12345 replicas 64 \
             hessian_displacement 0.001 anharmonic_reweighting yes
dump_qct     10
run          100000
```

Post-process with:

```bash
python3 tools/qct/lsc_ivr.py \
  --trajectory qct_trajectory.xyz \
  --summary qct_initial_summary.csv \
  --config lsc_ivr.json \
  --output qct_lsc_correlation.csv \
  --fft qct_lsc_spectrum.csv \
  --time-step 0.1 \
  --dump-interval 10
```

Config file defines operators A and B:

```json
{
  "operator_A": {"name": "position", "params": {"atom": 0, "axis": 0}},
  "operator_B": {"name": "position", "params": {"atom": 0, "axis": 0}}
}
```

### Input parameters for `ensemble qct wigner`

| Keyword | Required | Default | Description |
|---------|----------|---------|-------------|
| `temperature` | yes | — | Sampling temperature (K). Use `0` for ground-state Wigner. |
| `seed` | no | random | Random seed for reproducible sampling |
| `replicas` | no | 1 | Number of independent trajectories |
| `hessian_displacement` | no | 0.001 Å | Finite-difference Hessian displacement |
| `anharmonic_reweighting` | no | yes | Compute anharmonic reweighting weights |
| `stationary_point` | no | auto | Must be `minimum` or `auto` (not `saddle`) |
| `min_frequency` | no | 0.001 THz | Minimum frequency for active modes (works with auto Hessian) |
| `eigenvector` | no | — | External eigenvector file |
| `modes` | no | — | External `qct_modes.in` file |

**Constraints**: `zpe no` is rejected (Wigner always includes ZPE).
`saddle` is rejected (use `minimum` or `auto`).  If the auto Hessian
detects a strongly imaginary mode, the structure is classified as a
saddle point and Wigner sampling is aborted.

### Operators

| Name | Parameters | Description |
|------|-----------|-------------|
| `position` | `atom`, `axis` | Cartesian position component |
| `velocity` | `atom`, `axis` | Cartesian velocity component |
| `com_position` | `atom_indices` (optional) | Center-of-mass position magnitude |
| `com_velocity` | `atom_indices` (optional) | Center-of-mass speed |
| `bond_length` | `atom1`, `atom2` | Distance between two atoms |
| `kinetic_energy` | `atom_indices` (optional) | Total kinetic energy (eV) |
| `point_charge_dipole` | `axis`, `charges` | Dipole moment component from point charges |

### Output files

| File | Description |
|------|-------------|
| `qct_initial_summary.csv` | Per-replica summary with `wigner_weight` and `log_wigner_weight` columns |
| `qct_trajectory.xyz` | Multi-replica extxyz trajectory (positions, velocities, masses) |
| `qct_thermo.csv` | Per-replica per-step energy diagnostics |
| `qct_initial.out` | Human-readable initial-condition audit |

### Correlation formula

```
C_AB(t) = Σ_i w_i · A(0)_i · B(t)_i  /  Σ_i w_i
```

Standard error uses importance-sampling (ratio estimator) variance:

```
Var[Ĉ(t)] ≈ Σ_i [ w_i² · (f_i - Ĉ)² ] / (Σ_i w_i)²
```

### `lsc_ivr.py` command-line options

| Option | Description |
|--------|-------------|
| `--trajectory` | Multi-replica QCT trajectory (extxyz) — **required** |
| `--summary` | `qct_initial_summary.csv` with Wigner weights |
| `--config` | JSON config defining operators A and B — **required** |
| `--output` | Output correlation CSV (default: `qct_lsc_correlation.csv`) |
| `--fft` | Write FFT spectral density to this CSV |
| `--max-lag` | Maximum correlation lag in fs |
| `--window` | FFT window: `hann` (default), `hamming`, `bartlett`, `none` |
| `--time-step` | MD time step in fs (if trajectory lacks `Time=`) |
| `--dump-interval` | Dump interval in steps (if trajectory lacks `Time=`) |

### Tested systems

| System | Atoms | Replicas | Steps | Sampling | Peak (THz) |
|--------|-------|----------|-------|----------|------------|
| OH radical | 2 | 32 | 5000 | Wigner + reweight | ~112–115 |
| Ethanol C₂H₆O | 9 | 64 | 10000 | Wigner + reweight | 4.2–112.3 (20 modes) |

### Slurm batch scripts

Reference Slurm batch scripts for HPC:

* `tests/gpumd/qct_nep89_oh/lsc_ivr.batch` — OH radical test
* `tests/gpumd/qct_nep89_ethanol/lsc_ivr.batch` — ethanol test

### Full documentation

For complete documentation including theory, physical constants,
troubleshooting, and examples, see [`docs/lsc_ivr.md`](../../docs/lsc_ivr.md).

## Semiclassical methods roadmap

For the broader semiclassical methods roadmap (LSC-IVR, RPMD correlation,
SC-IVR/FBTS, PLDM), see [`SEMICLASSICAL_ROADMAP.md`](SEMICLASSICAL_ROADMAP.md).

The SC-IVR/FBTS implementation plan is documented in
[`SC_IVR_FBTS_PLAN.md`](SC_IVR_FBTS_PLAN.md).

The current combined QCT/LSC-IVR assessment is the
[comprehensive V3 review](QCT_LSC_IVR_CODE_REVIEW_V3.md); implementation order
and release gates are defined in the [repair plan](QCT_LSC_IVR_FIX_PLAN.md).
