# QCT tools

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
The batch path currently accepts only `dump_qct`; standard measurements do not
yet perform segmented per-replica reductions.

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
