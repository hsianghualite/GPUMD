# Multi-GPU QCT/LSC-IVR Guide

## Overview

QCT and LSC-IVR replica calculations are embarrassingly parallel: each
replica is an independent trajectory. GPUMD provides two approaches for
multi-GPU execution:

1. **Process-level parallelism (方案A — recommended for large systems)**:
   `run_multigpu.py` launches one GPUMD process per replica and schedules
   those processes across the visible GPUs. Every process uses `replicas=1`,
   a unique seed, and an isolated working directory. This uses the standard
   MD code path with cell-list neighbor search (O(N) scaling), making it
   suitable for condensed-phase systems with >1000 atoms. Results are merged
   automatically, including Wigner-weighted HAC averaging for thermal
   conductivity.

2. **Single-GPU batch**: The native `replicas N` keyword packs N replicas
   into one GPU memory space. This uses the QCT batch neighbor list
   (`find_neighbor_list_qct_batch`), which is O(N²) brute force with no
   cell list. Efficient for small molecules (< 100 atoms) but does not
   scale to large systems.

## When to use multi-GPU

- You have >1 GPU available
- The system is large (> 100 atoms, especially condensed-phase)
- You want maximum throughput (linear scaling with GPU count)
- The single-GPU batch mode is too slow or runs out of memory

## Usage

```bash
# Run 8 independent LSC-IVR trajectories across 4 GPUs
python tools/qct/run_multigpu.py \
    --template run_dir \
    --gpumd ~/gpumd/src/gpumd \
    --total-replicas 5 \
    --num-gpus 4 \
    --base-seed 12345 \
    --workflow hac \
    --shared-eigenvector /absolute/path/qct_eigenvector.out \
    --output merged_output/
```

The template directory must contain:
- `run.in` — with `ensemble lsc_ivr ...` or `ensemble qct ...`
- `model.xyz` — the structure file
- Any potential files referenced in `run.in`

The tool:
1. Creates `replica_000000/`, `replica_000001/`, ... work directories
2. Rewrites each `run.in` with `replicas 1` and a globally unique seed
3. Schedules tasks across the selected `CUDA_VISIBLE_DEVICES` tokens
4. Launches at most one process per visible GPU at a time
5. Merges results:
   - `qct_trajectory.xyz` — all replica frames, preserving global replica IDs
   - `qct_initial_summary.csv` — all replica Wigner weights
   - `qct_thermo.csv` — per-step thermo data
   - `qct_zpe.csv` — ZPE leakage data (if `dump_qct ... zpe` used)
   - `hac.out` — **Wigner-weighted** HAC for thermal conductivity
   - `hac_uncertainty.csv` — between-replica standard error at each time
   - `hac_merge_manifest.json` — normalized weights, effective sample size,
     endpoint conductivity, hashes, and acceptance diagnostics

Each replica's `manifest.json` records the resolved external input paths and
SHA-256 hashes under `referenced_inputs`. Resume reuses a completed task only
when its rewritten input, executable, model, referenced files, completion
marker, required artifacts, and expected seed still validate. Each selected
GPU owns a serial task queue, so no two launcher processes use the same GPU at
the same time.

For a periodic production calculation, calculate the automatic Hessian once
and pass the resulting `qct_eigenvector.out` through `--shared-eigenvector`.
The runner removes `hessian_displacement`, inserts `exclude_lowest 3`, and
uses the same mode basis with independent seeds. Relative potential and mode
paths are resolved before task directories are created.

## Thermal conductivity (Green-Kubo) with LSC-IVR

For LSC-IVR thermal conductivity, each replica process runs an independent
NVE trajectory with Wigner-sampled initial conditions and `compute_hac`.
The `run_multigpu.py` tool validates identical 11-column schemas, row counts,
and time grids, then merges the `hac.out` files with proper Wigner weighting:

$$\kappa(t) = \frac{\sum_i w_i \, \kappa_i(t)}{\sum_i w_i}$$

where $w_i$ is the Wigner (anharmonic reweighting) factor for replica $i$.
When `anharmonic_reweighting no` is set, all $w_i = 1$ and the merge
reduces to a simple average.

The merger prefers `log_wigner_weight`, normalizes in log space, and rejects
missing or malformed replicas. By default it requires an effective sample
size of at least `min(3, replicas)` and no normalized weight above 0.5 when
more than one replica is requested. Override these research acceptance gates
with `--min-effective-replicas` and `--max-normalized-weight` only when the
reason is documented.

For a native GPUMD batch (`replicas > 1`), the executable now performs the
same weighted reduction internally and writes `hac_replica.out` plus
`hac_reweighting.csv` for audit. `run_multigpu.py` always rewrites tasks to
`replicas 1`, so its process-level merge is the only weighting layer and does
not double-weight the per-process `hac.out` files. A single reweighted
trajectory is not a final normalized estimator by itself.

### Example run.in for LSC-IVR thermal conductivity

```
potential    nep89.txt
time_step    0.5
ensemble     lsc_ivr 300 seed 12345 replicas 1 hessian_displacement 0.001 anharmonic_reweighting no
compute_hac  20 500 10
dump_thermo  100
run          2000000
```

This template is used by `run_multigpu.py` — the seed is rewritten per
GPU process, and `replicas 1` ensures each process runs one trajectory
through the standard MD path (cell-list neighbor search, O(N) scaling).

### WSL CUDA launch

On WSL, run a single GPUMD trajectory through the wrapper below. It removes
stale container GPU mappings and prepends the Windows driver bridge library
directory before GPUMD starts:

```bash
tools/qct/run_wsl_cuda.sh /absolute/path/to/src/gpumd < run.in
```

`run_multigpu.py` applies the same environment normalization to every child
process automatically while retaining its per-GPU `CUDA_VISIBLE_DEVICES`
assignment.

## GPU selection

By default, the tool uses GPUs 0, 1, 2, ... Set `CUDA_VISIBLE_DEVICES`
in the environment to restrict which GPUs are used:

```bash
# Use only GPUs 2 and 3
CUDA_VISIBLE_DEVICES=2,3 python tools/qct/run_multigpu.py \
    --template run_dir --gpumd gpumd --total-replicas 64 --num-gpus 2
```

## Performance notes

- Each GPU runs independently with no communication overhead
- Scaling is linear up to the point where individual GPU throughput
  drops due to small replica counts
- **For large systems (> 1000 atoms)**: use `replicas=1` per GPU process.
  The QCT batch neighbor list (`find_neighbor_list_qct_batch`) is O(N²)
  brute force — no cell list. The standard MD path (`compute_large_box`)
  has a cell list and is O(N). So multi-GPU with `replicas=1` per process
  is the correct approach for condensed-phase systems.
- For small molecules (< 20 atoms), the single-GPU batch mode may be
  faster due to better GPU utilization
- No MPI is needed — all parallelism is process-level

## Architecture: 方案A vs batch mode

| Feature | 方案A (process-level) | Batch mode (replicas>1) |
|---|---|---|
| Neighbor list | Cell-list, O(N) | Brute-force, O(N²) |
| PBC support | Full (standard MD path) | Limited (no expanded box) |
| Max system size | Unlimited (cell-list) | ~100 atoms (MN limit) |
| Multi-GPU | Yes (one process per GPU) | No (single GPU only) |
| HAC merge | Wigner-weighted (automatic) | N/A (single HAC) |
| Best for | Condensed phase, large systems | Small molecules, gas phase |

## Post-processing

After multi-GPU run and merge:

```bash
# For correlation functions (vibrational spectra, etc.)
python tools/qct/lsc_ivr.py \
    --trajectory merged_output/qct_trajectory.xyz \
    --summary merged_output/qct_initial_summary.csv \
    --config lsc_ivr.json \
    --output correlation.csv \
    --fft spectrum.csv \
    --diagnostics diagnostics.csv

# For thermal conductivity, analyze hac.out directly
python -c "
import numpy as np
data = np.loadtxt('merged_output/hac.out')
kx = data[:, 6] + data[:, 7]
ky = data[:, 8] + data[:, 9]
kz = data[:, 10]
print(f'kappa_avg = {(kx[-1]+ky[-1]+kz[-1])/3:.4f} W/m/K')
"
```
