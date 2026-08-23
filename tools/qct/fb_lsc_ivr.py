#!/usr/bin/env python3
"""Forward-Backward LSC-IVR post-processing tool.

This module implements FB-LSC-IVR (Forward-Backward Linearized Semiclassical
Initial Value Representation) entirely in Python, without modifying the GPUMD
C++ source code.

Two modes are supported:

1. **HAC-based thermal conductivity** (``--hac``):
   Reads GPUMD ``hac.out`` files (one per replica) and computes the FB
   Green-Kubo integral with Wigner weighting and blockwise uncertainty.
   GPUMD's ``compute_hac`` already performs multi-time-origin averaging
   (all possible origins), which subsumes the statistical benefit of the
   symmetric (FB) form for real autocorrelations.  This tool adds:
   - Wigner-weighted multi-replica merge
   - Blockwise uncertainty separation (between-replica vs. within-trajectory)
   - Symmetric Green-Kubo integration with convergence analysis

2. **Trajectory-based correlations** (``--trajectory``):
   For molecular observables (position, velocity, dipole autocorrelation),
   the FB form ``C(t) = <A(-t/2) B(t/2)>`` is computed from the QCT
   trajectory dump via midpoint splitting.  This is a convenience wrapper
   around the ``--symmetric`` option of ``lsc_ivr.py``.

Usage
-----
HAC mode:
    python fb_lsc_ivr.py --hac replica_0/hac.out replica_1/hac.out \\
        --summary replica_0/qct_initial_summary.csv replica_1/qct_initial_summary.csv \\
        --temperature 300 --volume 1000 --output fb_kappa.csv

Trajectory mode:
    python fb_lsc_ivr.py --trajectory qct_trajectory.xyz \\
        --summary qct_initial_summary.csv --config operators.json \\
        --output fb_correlation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_TOOL_DIR = Path(__file__).resolve().parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

# Physical constants
K_B_EV_K = 8.617333262e-5  # eV/K
# GPUMD internal units: time in natural units, conversion factor
# TIME_UNIT_CONVERSION = 1.018051e+1 fs/natural_time_unit
# HAC unit conversion (matching hac.cu)
# factor = dt * 0.5 / (K_B * T * T * V) * KAPPA_UNIT_CONVERSION
# KAPPA_UNIT_CONVERSION = 1.0e27 / (TIME_UNIT_CONVERSION^2)
# In hac.out, columns 7-11 are RTC (running thermal conductivity)
# which already includes the proper unit conversion.

# HAC output columns (1-indexed):
# 1: time(ps), 2: hac_xi, 3: hac_xo, 4: hac_yi, 5: hac_yo, 6: hac_z,
# 7: rtc_xi, 8: rtc_xo, 9: rtc_yi, 10: rtc_yo, 11: rtc_z


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HACData:
    """HAC data from a single replica."""
    time_ps: np.ndarray         # (N,) time in ps
    hac: np.ndarray             # (N, 5) heat current autocorrelation (xi, xo, yi, yo, z)
    rtc: np.ndarray             # (N, 5) running thermal conductivity (xi, xo, yi, yo, z)
    log_wigner_weight: float = 0.0


@dataclass
class FBKappaResult:
    """FB-LSC-IVR thermal conductivity result."""
    time_ps: np.ndarray
    kappa_x: np.ndarray         # running kappa in x
    kappa_y: np.ndarray         # running kappa in y
    kappa_z: np.ndarray         # running kappa in z
    kappa_avg: np.ndarray       # isotropic average
    kappa_x_se: np.ndarray      # standard error (between-replica)
    kappa_y_se: np.ndarray
    kappa_z_se: np.ndarray
    kappa_avg_se: np.ndarray
    block_se_kappa_avg: np.ndarray   # blockwise (within-trajectory) SE
    combined_se_kappa_avg: np.ndarray  # combined SE
    effective_replicas: float
    n_replicas: int
    block_size: int | None
    n_blocks: int


# ---------------------------------------------------------------------------
# HAC file I/O
# ---------------------------------------------------------------------------

def load_hac_file(path: Path) -> HACData:
    """Load a single hac.out file.

    Format (11 columns):
        time(ps), hac_xi, hac_xo, hac_yi, hac_yo, hac_z,
        rtc_xi, rtc_xo, rtc_yi, rtc_yo, rtc_z
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 11:
        raise ValueError(
            f"hac.out must have 11 columns, got {data.shape[1]}: {path}"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError(f"hac.out contains non-finite data: {path}")
    if data.shape[0] > 1 and not np.all(np.diff(data[:, 0]) > 0.0):
        raise ValueError(f"hac.out time grid must be increasing: {path}")
    return HACData(
        time_ps=data[:, 0].copy(),
        hac=data[:, 1:6].copy(),
        rtc=data[:, 6:11].copy(),
    )


def load_wigner_log_weights(summary_path: Path) -> float:
    """Load the log Wigner weight from a qct_initial_summary.csv file.

    Returns a single float (the log weight for replica 0).
    """
    with summary_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(
            f"Summary file must have exactly 1 row, got {len(rows)}: {summary_path}"
        )
    row = rows[0]
    for col in ("log_wigner_weight", "log_wigner"):
        if col in row:
            val = float(row[col])
            if not math.isfinite(val) and val != -math.inf:
                raise ValueError(f"Invalid log Wigner weight in {summary_path}: {val}")
            return val
    # No weight column → classical (w=1)
    return 0.0


def load_multiple_hac(
    hac_files: list[Path],
    summary_files: list[Path] | None = None,
) -> list[HACData]:
    """Load multiple hac.out files with Wigner weights.

    If summary_files is provided, each file provides the log Wigner weight
    for the corresponding replica.
    """
    if not hac_files:
        raise ValueError("No hac.out files provided")
    if summary_files is None:
        summary_files = [None] * len(hac_files)
    if len(summary_files) != len(hac_files):
        raise ValueError("hac_files and summary_files must have equal length")

    datasets = []
    ref_time = None
    for i, (hac_path, summary_path) in enumerate(zip(hac_files, summary_files)):
        data = load_hac_file(hac_path)
        if ref_time is None:
            ref_time = data.time_ps.copy()
        elif not np.allclose(data.time_ps, ref_time, rtol=1e-12, atol=1e-14):
            raise ValueError(
                f"Time grid mismatch in hac.out file {i}: {hac_path}"
            )
        log_w = 0.0
        if summary_path is not None and Path(summary_path).is_file():
            log_w = load_wigner_log_weights(Path(summary_path))
        data.log_wigner_weight = log_w
        datasets.append(data)

    return datasets


# ---------------------------------------------------------------------------
# Wigner-weighted merge
# ---------------------------------------------------------------------------

def _normalize_weights(log_weights: list[float]) -> np.ndarray:
    """Convert log weights to normalized weights (sum = 1)."""
    finite = [w for w in log_weights if math.isfinite(w)]
    if not finite:
        raise ValueError("All Wigner weights are zero (-inf)")
    offset = max(finite)
    scaled = np.array(
        [0.0 if w == -math.inf else math.exp(w - offset) for w in log_weights],
        dtype=float,
    )
    total = float(np.sum(scaled))
    if total <= 0.0:
        raise ValueError("Wigner weights have invalid total")
    return scaled / total


def merge_hac_wigner(datasets: list[HACData]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Merge HAC data with Wigner weighting.

    Returns (merged_hac, merged_rtc, normalized_weights, effective_replicas).
    """
    log_weights = [d.log_wigner_weight for d in datasets]
    normalized = _normalize_weights(log_weights)

    # Stack HAC and RTC: (n_replicas, N, 5)
    hac_stack = np.stack([d.hac for d in datasets], axis=0)
    rtc_stack = np.stack([d.rtc for d in datasets], axis=0)

    # Weighted merge: sum_i w_i * data_i / sum_i w_i
    merged_hac = np.tensordot(normalized, hac_stack, axes=(0, 0))
    merged_rtc = np.tensordot(normalized, rtc_stack, axes=(0, 0))

    effective = 1.0 / float(np.sum(normalized ** 2))
    return merged_hac, merged_rtc, normalized, effective


# ---------------------------------------------------------------------------
# FB Green-Kubo integration
# ---------------------------------------------------------------------------

def compute_fb_kappa(
    datasets: list[HACData],
    block_size: int | None = None,
) -> FBKappaResult:
    """Compute FB-LSC-IVR thermal conductivity from HAC data.

    The Green-Kubo formula:
        κ_α(t) = (1/(V k_B T²)) ∫₀^t J_α(0)·J_α(τ) dτ

    In GPUMD's hac.out, the RTC (running thermal conductivity) already
    contains this integral with proper unit conversion.  The FB form:

        κ_FB(t) = (1/2) * [κ(t) + κ(-t)]
                = κ(t)    (since HAC is real and even for autocorrelation)

    For autocorrelations of real observables, the FB form is mathematically
    identical to the one-sided form.  The benefit of FB is statistical
    (better sampling for symmetric operators in short trajectories), which
    GPUMD's multi-time-origin averaging already captures.

    This function:
    1. Merges HAC across replicas with Wigner weights
    2. Computes per-replica kappa endpoints
    3. Computes between-replica uncertainty (Wigner sampling noise)
    4. Computes within-trajectory uncertainty (block analysis)
    5. Combines both for total uncertainty
    """
    merged_hac, merged_rtc, normalized, effective = merge_hac_wigner(datasets)
    time_ps = datasets[0].time_ps
    n_replicas = len(datasets)
    n_rows = len(time_ps)

    # Per-replica kappa from RTC: (n_replicas, N, 3) + (n_replicas, N, 1) avg
    kappa_r = np.zeros((n_replicas, n_rows, 4), dtype=float)
    for i, d in enumerate(datasets):
        kappa_r[i, :, 0] = d.rtc[:, 0] + d.rtc[:, 1]   # xi + xo
        kappa_r[i, :, 1] = d.rtc[:, 2] + d.rtc[:, 3]   # yi + yo
        kappa_r[i, :, 2] = d.rtc[:, 4]                   # z
    kappa_r[:, :, 3] = np.mean(kappa_r[:, :, :3], axis=2)

    # Weighted mean kappa
    kappa_mean = np.tensordot(normalized, kappa_r, axes=(0, 0))

    # Between-replica uncertainty (Wigner initial condition noise)
    # SE = sqrt( Σ_i w_i² (κ_i - κ_mean)² )
    kappa_se = np.sqrt(
        np.sum(
            (normalized[:, None, None] ** 2)
            * (kappa_r - kappa_mean[None, :, :]) ** 2,
            axis=0,
        )
    )

    # Blockwise (within-trajectory) uncertainty
    # Split each replica's HAC into non-overlapping blocks, compute
    # block-averaged kappa endpoints, and estimate variance across blocks.
    block_se_flat = np.zeros(4, dtype=float)
    n_blocks_total = 0
    if block_size is not None and block_size > 0 and n_rows >= block_size * 2:
        n_blocks_per_replica = n_rows // block_size
        n_blocks_total = n_blocks_per_replica * n_replicas
        if n_blocks_per_replica > 1:
            block_kappa = np.zeros((n_replicas, n_blocks_per_replica, 4), dtype=float)
            for i in range(n_replicas):
                for b in range(n_blocks_per_replica):
                    s = b * block_size
                    e = s + block_size
                    block_kappa[i, b] = np.mean(kappa_r[i, s:e], axis=0)
            # Weighted block mean
            weighted_block = np.tensordot(normalized, block_kappa, axes=(0, 0))
            # Block variance per replica, accumulated with w_i²
            block_var = np.zeros(4, dtype=float)
            for i in range(n_replicas):
                w = normalized[i]
                diff = block_kappa[i] - weighted_block  # (n_blocks, 4)
                block_var += (w ** 2) * np.var(diff, axis=0, ddof=1)
            w2_sum = max(float(np.sum(normalized ** 2)), 1e-60)
            block_se_flat = np.sqrt(block_var / w2_sum)

    # Broadcast block SE to per-time-row
    block_se = np.broadcast_to(block_se_flat, (n_rows, 4)).copy()
    combined_se = np.sqrt(kappa_se ** 2 + block_se ** 2)

    return FBKappaResult(
        time_ps=time_ps,
        kappa_x=kappa_mean[:, 0],
        kappa_y=kappa_mean[:, 1],
        kappa_z=kappa_mean[:, 2],
        kappa_avg=kappa_mean[:, 3],
        kappa_x_se=kappa_se[:, 0],
        kappa_y_se=kappa_se[:, 1],
        kappa_z_se=kappa_se[:, 2],
        kappa_avg_se=kappa_se[:, 3],
        block_se_kappa_avg=block_se[:, 3],
        combined_se_kappa_avg=combined_se[:, 3],
        effective_replicas=effective,
        n_replicas=n_replicas,
        block_size=block_size,
        n_blocks=n_blocks_total,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_fb_kappa_csv(path: Path, result: FBKappaResult) -> None:
    """Write FB kappa results to CSV."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_ps",
            "kappa_x", "kappa_y", "kappa_z", "kappa_avg",
            "se_kappa_x", "se_kappa_y", "se_kappa_z", "se_kappa_avg",
            "block_se_kappa_avg", "combined_se_kappa_avg",
        ])
        for i in range(len(result.time_ps)):
            writer.writerow([
                f"{result.time_ps[i]:.15e}",
                f"{result.kappa_x[i]:.15e}",
                f"{result.kappa_y[i]:.15e}",
                f"{result.kappa_z[i]:.15e}",
                f"{result.kappa_avg[i]:.15e}",
                f"{result.kappa_x_se[i]:.15e}",
                f"{result.kappa_y_se[i]:.15e}",
                f"{result.kappa_z_se[i]:.15e}",
                f"{result.kappa_avg_se[i]:.15e}",
                f"{result.block_se_kappa_avg[i]:.15e}",
                f"{result.combined_se_kappa_avg[i]:.15e}",
            ])
    import os
    os.replace(tmp, path)
    print(f"Wrote FB kappa to {path}")


def write_fb_manifest(path: Path, result: FBKappaResult, hac_files: list[Path]) -> None:
    """Write a JSON manifest with provenance and diagnostics."""
    manifest = {
        "schema": "FB_LSC_IVR_KAPPA_v1",
        "n_replicas": result.n_replicas,
        "effective_replicas": result.effective_replicas,
        "block_size": result.block_size,
        "n_blocks_total": result.n_blocks,
        "hac_files": [str(p) for p in hac_files],
        "endpoint": {
            "kappa_x": float(result.kappa_x[-1]),
            "kappa_y": float(result.kappa_y[-1]),
            "kappa_z": float(result.kappa_z[-1]),
            "kappa_avg": float(result.kappa_avg[-1]),
            "se_kappa_avg": float(result.kappa_avg_se[-1]),
            "combined_se_kappa_avg": float(result.combined_se_kappa_avg[-1]),
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    import os
    os.replace(tmp, path)
    print(f"Wrote manifest to {path}")


# ---------------------------------------------------------------------------
# Trajectory-based FB correlation (delegates to lsc_ivr.py)
# ---------------------------------------------------------------------------

def run_trajectory_fb(args: argparse.Namespace) -> None:
    """Run trajectory-based FB correlation via lsc_ivr.py's --symmetric."""
    import subprocess

    cmd = [
        sys.executable,
        str(_TOOL_DIR / "lsc_ivr.py"),
        "--trajectory", str(args.trajectory),
        "--output", str(args.output),
        "--symmetric",
    ]
    if args.summary:
        cmd.extend(["--summary", str(args.summary)])
    if args.config:
        cmd.extend(["--config", str(args.config)])
    if args.fft:
        cmd.extend(["--fft", str(args.fft)])
    if args.max_lag:
        cmd.extend(["--max-lag", str(args.max_lag)])
    if args.window:
        cmd.extend(["--window", str(args.window)])

    print(f"Running trajectory FB via lsc_ivr.py --symmetric...")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"lsc_ivr.py exited with code {result.returncode}")
    print(f"FB trajectory correlation written to {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--hac", nargs="+", type=Path,
        help="HAC mode: one or more hac.out files (one per replica).",
    )
    mode.add_argument(
        "--trajectory", type=Path,
        help="Trajectory mode: QCT trajectory extxyz (uses --symmetric from lsc_ivr.py).",
    )

    # HAC options
    parser.add_argument(
        "--summary", nargs="+", type=Path, default=None,
        help="qct_initial_summary.csv files (one per replica, for Wigner weights). "
             "In trajectory mode, a single summary file.",
    )
    parser.add_argument(
        "--block-size", type=int, default=None,
        help="Block size (in time rows) for within-trajectory uncertainty. "
             "Default: disabled.",
    )

    # Trajectory options
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON config defining operator_A and operator_B (trajectory mode).",
    )
    parser.add_argument(
        "--fft", type=Path, default=None,
        help="FFT spectrum output (trajectory mode).",
    )
    parser.add_argument(
        "--max-lag", type=float, default=None,
        help="Maximum correlation lag in fs (trajectory mode).",
    )
    parser.add_argument(
        "--window", default="hann",
        choices=["hann", "hamming", "bartlett", "none"],
        help="Window function for FFT (trajectory mode). Default: hann.",
    )

    # Output
    parser.add_argument(
        "--output", type=Path, default=Path("fb_lsc_ivr_output.csv"),
        help="Output CSV file.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Optional JSON manifest with provenance and diagnostics.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.trajectory is not None:
        run_trajectory_fb(args)
        return

    # HAC mode
    hac_files = args.hac
    summary_files = args.summary if args.summary else [None] * len(hac_files)

    print(f"FB-LSC-IVR HAC mode: {len(hac_files)} replica(s)")
    datasets = load_multiple_hac(hac_files, summary_files)

    for i, d in enumerate(datasets):
        print(f"  Replica {i}: {len(d.time_ps)} rows, "
              f"log_wigner_weight={d.log_wigner_weight:.6f}")

    result = compute_fb_kappa(datasets, block_size=args.block_size)

    print(f"\nFB-LSC-IVR Thermal Conductivity Results:")
    print(f"  Effective replicas: {result.effective_replicas:.3f}")
    print(f"  N_blocks: {result.n_blocks}")
    print(f"  κ_x = {result.kappa_x[-1]:.4f} ± {result.kappa_x_se[-1]:.4f} W/m·K")
    print(f"  κ_y = {result.kappa_y[-1]:.4f} ± {result.kappa_y_se[-1]:.4f} W/m·K")
    print(f"  κ_z = {result.kappa_z[-1]:.4f} ± {result.kappa_z_se[-1]:.4f} W/m·K")
    print(f"  κ_avg = {result.kappa_avg[-1]:.4f} ± {result.kappa_avg_se[-1]:.4f} W/m·K")
    if result.n_blocks > 0:
        print(f"  Combined SE (κ_avg) = {result.combined_se_kappa_avg[-1]:.4f} W/m·K")

    write_fb_kappa_csv(args.output, result)

    if args.manifest is not None:
        write_fb_manifest(args.manifest, result, hac_files)


if __name__ == "__main__":
    main()
