# LSC-IVR: Linearized Semiclassical Initial Value Representation

## Overview

LSC-IVR (also called Wigner–Liouville dynamics) is a semiclassical dynamics
method that captures quantum effects—zero-point energy, tunneling, and thermal
quantum fluctuations—within a classical trajectory framework.  In GPUMD it is
implemented as a sampling mode of the native QCT engine:

* **Sampling**: initial conditions are drawn from the harmonic Wigner thermal
  distribution, not the classical Boltzmann + ZPE offset used by standard QCT.
* **Propagation**: each replica runs independent NVE velocity-Verlet dynamics,
  identical to ordinary QCT.
* **Post-processing**: quantum-corrected time-correlation functions
  `C_AB(t) = <w_i A(0)_i B(t)_i> / <w_i>` are computed by `lsc_ivr.py`.

### When to use LSC-IVR

| Property | Classical MD | QCT (canonical) | LSC-IVR |
|----------|:-----------:|:---------------:|:-------:|
| ZPE in initial state | ❌ | ✅ (exact) | ✅ (exact) |
| Thermal quantum fluctuation | ❌ | partial | ✅ |
| Tunneling | ❌ | ❌ | partial |
| Cost per trajectory | 1× | 1× | 1× |
| Post-processing cost | trivial | trivial | low |
| Coherence / interference | ❌ | ❌ | ❌ |

LSC-IVR is the **cheapest** semiclassical method and should be the first choice
when quantum effects on initial conditions matter.  For strongly anharmonic
systems or when quantum coherence is important, consider RPMD or SC-IVR/FBTS
(see `tools/qct/SEMICLASSICAL_ROADMAP.md`).

## Quick Start

### 1. Prepare the optimized structure

The `model.xyz` must be a **minimum** (optimized geometry), not a saddle
point.  If the structure is a saddle point, the automatic Hessian will detect
an imaginary frequency and LSC-IVR will refuse to run.

```
2
pbc="F F F" Lattice="30 0 0 0 30 0 0 0 30" Properties=species:S:1:pos:R:3:mass:R:1
O 14.50782069 15.00000000 15.00000000 15.99900000
H 15.46217931 15.00000000 15.00000000 1.00800000
```

### 2. Write the input file (`run.in`)

```
potential    nep.txt
time_step    0.1
ensemble     qct wigner temperature 300 seed 12345 replicas 64 \
             hessian_displacement 0.001 anharmonic_reweighting yes
dump_qct     10
run          100000
```

### 3. Run GPUMD

```bash
/path/to/gpumd
```

### 4. Post-process with `lsc_ivr.py`

```bash
python3 tools/qct/lsc_ivr.py \
  --trajectory qct_trajectory.xyz \
  --summary qct_initial_summary.csv \
  --config lsc_ivr_config.json \
  --output qct_lsc_correlation.csv \
  --fft qct_lsc_spectrum.csv \
  --time-step 0.1 \
  --dump-interval 10
```

Configuration file (`lsc_ivr_config.json`):

```json
{
  "operator_A": {"name": "position", "params": {"atom": 0, "axis": 0}},
  "operator_B": {"name": "position", "params": {"atom": 0, "axis": 0}}
}
```

## Input Parameters

### `ensemble qct wigner` Syntax

```
ensemble qct wigner temperature T [key-value pairs...]
```

**Required parameters:**

| Keyword | Type | Description |
|---------|------|-------------|
| `temperature` | float (K) | Sampling temperature. Use `0` for ground-state Wigner (T=0 limit). |

**Optional parameters:**

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `seed` | int | random | Random seed for reproducible sampling |
| `replicas` | int | 1 | Number of independent replicas (trajectories) |
| `hessian_displacement` | float (Å) | 0.001 | Finite-difference displacement for automatic Hessian |
| `anharmonic_reweighting` | yes/no | yes | Compute anharmonic reweighting weights |
| `stationary_point` | auto/minimum/saddle | auto | Stationary point type (Wigner requires minimum or auto) |
| `min_frequency` | float (THz) | 0.001 | Minimum frequency for a mode to be active |
| `exclude_lowest` | int | 0 | Exclude N lowest modes (only with eigenvector file) |
| `eigenvector` | file path | — | Use external eigenvector file instead of auto Hessian |
| `modes` | file path | — | Use external qct_modes.in file |
| `zpe` | yes/no | yes | ZPE flag. **Must be yes for Wigner** (ZPE is intrinsic). |

### Constraints and validation

* `zpe no` with `wigner` is **rejected**—Wigner always includes ZPE.
* `stationary_point saddle` with `wigner` is **rejected**—use `minimum` or
  `auto`.
* `min_frequency` is allowed with auto Hessian mode (useful for filtering
  spurious near-zero imaginary modes).
* If the auto Hessian detects a strongly imaginary mode (< -min_frequency),
  the structure is classified as a saddle point and Wigner sampling is aborted.

## How LSC-IVR Works

### Wigner thermal distribution

For each active normal mode *k* with frequency ω_k, the harmonic Wigner
distribution draws independent Gaussian random variables (Q_k, P_k) with:

```
σ_Q² = (ℏ / 2ω_k) · coth(βℏω_k / 2)
σ_P² = (ℏω_k / 2) · coth(βℏω_k / 2)
```

At high temperature (β → 0), `coth → 1/(βℏω_k)` and the distribution reduces to
the classical Boltzmann result.  At T = 0, `coth → 1` and the ground-state Wigner
distribution is recovered.

**Numerical stability**: the `coth(x)` is computed as `(exp(2x) + 1) / (exp(2x) - 1)`
with a guard for `x > 350` (where `coth → 1`), preventing overflow.

### Mass-weighted coordinates

The normal-mode coordinates Q_k are mass-weighted.  When constructing Cartesian
positions and velocities, each displacement is divided by √m_n to convert from
mass-weighted to Cartesian:

```
Δx_n = Σ_k (e_{k,n} / √m_n) · Q_k
```

where e_{k,n} is the mass-weighted eigenvector component.

### Anharmonic reweighting

Because the Wigner sampler uses the harmonic approximation, samples are
reweighted by the ratio of the true Boltzmann weight to the harmonic one:

```
w_i = exp(-β · ΔV_i)
```

where:

```
ΔV = V_real(Q_i) - V_ref - Σ_k ½ ω_k² Q_k²
```

* `V_real(Q_i)` — true PES potential at the sampled geometry
* `V_ref` — potential at the reference (equilibrium) structure
* `Σ_k ½ ω_k² Q_k²` — harmonic potential relative to reference

Both `wigner_weight` and `log_wigner_weight` are written to
`qct_initial_summary.csv`.  The post-processor prefers `log_wigner_weight` for
numerical stability (see BUG-3 fix).

### Correlation function

The LSC-IVR Kubo-transformed correlation function is:

```
C_AB(t) = Σ_i w_i · A(0)_i · B(t)_i  /  Σ_i w_i
```

The standard error uses the importance-sampling (ratio estimator) variance:

```
Var[Ĉ(t)] ≈ (1/N) · Σ_i [ w_i² · (f_i - Ĉ)² ] / (Σ_i w_i)²
```

where `f_i = A(0)_i · B(t)_i` and `Ĉ` is the estimated mean.  This requires a
two-pass computation: first compute `Ĉ`, then compute the weighted variance of
residuals.

## Output Files

### `qct_initial_summary.csv`

Per-replica initial-condition summary with columns:

| Column | Description |
|--------|-------------|
| `replica` | Replica index (0-based) |
| `seed` | Random seed used for this replica |
| `total_sampled_energy_eV` | Total sampled energy |
| `rotational_energy_eV` | Rotational kinetic energy |
| `reaction_energy_eV` | Reaction-mode energy (0 for Wigner) |
| `potential_correction_eV` | ΔV = V_real - V_harmonic |
| `stable_velocity_scale` | Velocity scale factor (1.0 for Wigner) |
| `wigner_weight` | `exp(log_wigner_weight)` — may overflow/underflow |
| `log_wigner_weight` | Log of Wigner weight (preferred for post-processing) |

### `qct_trajectory.xyz`

Extended XYZ trajectory with per-replica frames.  Each frame has `Replica`,
`Step`, and `Seed` metadata attributes.  Contains positions, velocities, and
masses for all atoms.

### `qct_thermo.csv`

Per-replica per-step energy diagnostics.  The `kinetic_temperature_K` column is
`2K/(3N k_B)` and is a raw kinetic diagnostic, not a canonical molecular
temperature.

### `qct_initial.out`

Human-readable initial-condition audit with per-mode details including
frequency, quantum number, energy, eigenvector, and active/rigid classification.

## Post-Processing: `lsc_ivr.py`

### Command-line options

```
python3 tools/qct/lsc_ivr.py [options]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--trajectory` | ✅ | Multi-replica QCT trajectory (extxyz) |
| `--summary` | recommended | `qct_initial_summary.csv` with Wigner weights |
| `--config` | ✅ | JSON config defining operators A and B |
| `--output` | default | Output correlation CSV (default: `qct_lsc_correlation.csv`) |
| `--fft` | optional | Write FFT spectral density to this CSV |
| `--max-lag` | optional | Maximum correlation lag in fs |
| `--window` | default | FFT window: `hann`, `hamming`, `bartlett`, `none` |
| `--time-step` | conditional | MD time step in fs (if trajectory lacks `Time=`) |
| `--dump-interval` | conditional | Dump interval in steps (if trajectory lacks `Time=`) |

### Operators

Operators are defined in the JSON config:

```json
{
  "operator_A": {"name": "operator_name", "params": {...}},
  "operator_B": {"name": "operator_name", "params": {...}}
}
```

| Name | Parameters | Description |
|------|-----------|-------------|
| `position` | `atom` (0-based), `axis` (0=x, 1=y, 2=z) | Cartesian position component |
| `velocity` | `atom` (0-based), `axis` (0=x, 1=y, 2=z) | Cartesian velocity component |
| `com_position` | `atom_indices` (optional, default: all) | Center-of-mass position magnitude |
| `com_velocity` | `atom_indices` (optional, default: all) | Center-of-mass speed |
| `bond_length` | `atom1`, `atom2` (0-based) | Distance between two atoms |
| `kinetic_energy` | `atom_indices` (optional, default: all) | Total kinetic energy (eV) |
| `point_charge_dipole` | `axis`, `charges` (array) | Dipole moment component from point charges |

### Correlation output CSV

Columns: `time_fs`, `c_ab`, `c_ab_normalized`, `std_error`, `n_samples`

* `c_ab` — raw weighted sum `Σ w_i A(0)_i B(t)_i`
* `c_ab_normalized` — normalized correlation `c_ab / Σ w_i`
* `std_error` — standard error of the mean (importance-sampling variance)
* `n_samples` — number of replicas contributing at each lag

### Spectrum output CSV

Columns: `frequency_THz`, `wavenumber_cm_inv`, `intensity`

The FFT is computed on the normalized correlation function with the chosen
window function applied.

## Examples

### Example 1: OH Radical (2 atoms)

```
potential    nep.txt
time_step    0.1
ensemble     qct wigner temperature 300 seed 12345 replicas 32 \
             hessian_displacement 0.001 anharmonic_reweighting yes
dump_qct     1
run          5000
```

Config for O-atom position autocorrelation:

```json
{
  "operator_A": {"name": "position", "params": {"atom": 0, "axis": 0}},
  "operator_B": {"name": "position", "params": {"atom": 0, "axis": 0}}
}
```

Expected results: spectral peak at ~112–115 THz (≈3737–3834 cm⁻¹), matching
the OH stretch vibration.

### Example 2: Ethanol C₂H₆O (9 atoms)

```
potential    nep.txt
time_step    0.1
ensemble     qct wigner temperature 300 seed 12345 replicas 64 \
             hessian_displacement 0.001 anharmonic_reweighting yes
dump_qct     1
run          10000
```

Config for O-atom position autocorrelation (captures C-O stretch dynamics):

```json
{
  "operator_A": {"name": "position", "params": {"atom": 2, "axis": 0}},
  "operator_B": {"name": "position", "params": {"atom": 2, "axis": 0}}
}
```

Expected results: 20 active modes detected (4.2–112.3 THz).  Small imaginary
mode (-5.24 THz) is correctly skipped by the saddle-point guard.  The
structure must be an optimized minimum—if using a raw geometry, optimize it
first with an external tool (e.g., ASE) to avoid saddle-point classification.

### Example 3: Ground-State Wigner (T=0)

```
ensemble     qct wigner temperature 0 seed 42 replicas 128 \
             anharmonic_reweighting no
```

At T=0, `coth → 1` and each mode is sampled from the ground-state Wigner
distribution.  Anharmonic reweighting is disabled (all weights = 1.0) since
β → ∞ makes the reweighting weight ill-defined for the ground state.

## Running on HPC (Slurm)

Use a batch script instead of `srun` for interactive submission:

```bash
#!/bin/bash
#SBATCH --job-name=lsc-ivr
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=01:00:00

module load cuda/12.4.1
export OMP_NUM_THREADS=1

cd /path/to/run/directory
/path/to/gpumd

python3 /path/to/tools/qct/lsc_ivr.py \
  --trajectory qct_trajectory.xyz \
  --summary qct_initial_summary.csv \
  --config lsc_ivr_config.json \
  --output qct_lsc_correlation.csv \
  --fft qct_lsc_spectrum.csv \
  --time-step 0.1 \
  --dump-interval 10
```

Reference batch scripts are in:
* `tests/gpumd/qct_nep89_oh/lsc_ivr.batch` (OH radical)
* `tests/gpumd/qct_nep89_ethanol/lsc_ivr.batch` (ethanol)

## Physical Constants

GPUMD uses the eV–fs–amu–Ångstrom natural unit system:

| Constant | Value | Unit |
|----------|-------|------|
| ℏ | 6.465412 × 10⁻² | eV·fs |
| k_B | 8.617343 × 10⁻⁵ | eV/K |
| Time conversion | 1.018051 × 10¹ | natural→fs |
| THz → natural angular ω | 2π × 10⁻³ × 10.18 = 0.063963 | — |

Python post-processor uses:
* `HBAR_EV_FS = 0.6582119569`
* `AMU_A2_FS2_TO_EV = 103.6426965268`
* `K_B_EV_K = 8.617333262 × 10⁻⁵`

## Troubleshooting

### "ensemble qct wigner does not support saddle points"

The auto Hessian detected a strongly imaginary frequency.  Re-optimize the
geometry to a true minimum.  If small imaginary frequencies persist due to
numerical noise, increase `min_frequency` (e.g., `min_frequency 1.0`).

### "ensemble qct wigner always includes zero-point energy; remove 'zpe no'"

Remove `zpe no` from the input.  Wigner sampling always includes ZPE.

### Wigner weights are all zero or NaN

Check `log_wigner_weight` in `qct_initial_summary.csv`:
* If values are extremely negative (< -745), the sampled geometry has very high
  potential energy relative to the harmonic approximation.  This may indicate a
  bad structure or a too-small `hessian_displacement`.
* Try `anharmonic_reweighting no` to see if the issue is in the reweighting.

### No spectral peak in FFT output

* Ensure enough trajectory steps (at least 2–3 vibrational periods).
* Check that `--time-step` and `--dump-interval` match the `run.in` settings.
* Try `--window none` to see unwindowed spectrum.
* Verify that the chosen operator couples to the mode of interest (e.g., use
  the atom with the largest eigenvector component for that mode).

### Energy drift in trajectories

Reduce `time_step` in `run.in`.  For OH stretch (~115 THz), a time step of 0.1 fs
gives ~80 steps per period, which is typically sufficient.  For stiffer modes,
use 0.05 fs or smaller.

## References

1. H. Wang, X. Sun, W. H. Miller, *J. Chem. Phys.* **112**, 521 (1999).  
   (LSC-IVR / Wigner-Liouville dynamics)
2. J. A. Poulsen et al., *J. Chem. Phys.* **134**, 074309 (2011).  
   (LSC-IVR for vibrational spectra)
3. R. Hernandez, G. A. Voth, *Chem. Phys. Lett.* **323**, 89 (2000).  
   (Anharmonic reweighting for Wigner sampling)
4. See also: `tools/qct/SEMICLASSICAL_ROADMAP.md` for the broader semiclassical
   methods roadmap in GPUMD.
