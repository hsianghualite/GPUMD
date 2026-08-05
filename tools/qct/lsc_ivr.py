#!/usr/bin/env python3
"""LSC-IVR post-processing for QCT trajectories.

Reads multi-replica QCT trajectory (extxyz) and initial-condition summary
to compute quantum-corrected time-correlation functions using the
Linearized Semiclassical Initial Value Representation (LSC-IVR) formula:

    C_AB(t) = Σ_i w_i * A(0)_i * B(t)_i  /  Σ_i w_i

where ``w_i`` is the Wigner anharmonic reweighting weight from
``qct_initial_summary.csv`` (falls back to 1 if absent).

Optionally computes the spectral density via FFT of the correlation function.

Usage
-----
    python lsc_ivr.py --trajectory qct_trajectory.xyz \\
        --summary qct_initial_summary.csv \\
        --config lsc_ivr.json \\
        --output qct_lsc_correlation.csv \\
        --fft qct_lsc_spectrum.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Reuse the extxyz reader from analyze_qct.py
_TOOL_DIR = Path(__file__).resolve().parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))
import analyze_qct as aq  # noqa: E402

# Physical constants (eV-fs-amu-Angstrom system, matching GPUMD)
AMU_A2_FS2_TO_EV = aq.AMU_A2_FS2_TO_EV  # 103.6426965268
HBAR_EV_FS = aq.HBAR_EV_FS              # 0.6582119569
K_B_EV_K = 8.617333262e-5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReplicaTrajectory:
    """Per-replica trajectory frames (one per dump interval)."""
    replica_id: int
    frames: list  # list of aq.Frame


@dataclass
class CorrelationResult:
    time_fs: np.ndarray
    c_ab: np.ndarray
    c_ab_normalized: np.ndarray
    std_error: np.ndarray
    n_samples: np.ndarray


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_wigner_weights(summary_path: Path) -> dict[int, float]:
    """Load per-replica Wigner weights from qct_initial_summary.csv.

    Returns a dict ``{replica_id: weight}``.  If the file does not contain
    a ``wigner_weight`` column, all weights default to 1.0.
    """
    weights: dict[int, float] = {}
    if not summary_path.is_file():
        return weights
    with summary_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return weights
        has_weight = "wigner_weight" in reader.fieldnames
        for row in reader:
            rid = int(row["replica"])
            if has_weight:
                w = float(row["wigner_weight"])
                # Guard against underflow/overflow
                if not math.isfinite(w) or w <= 0.0:
                    w = 0.0  # zero-weight samples contribute nothing
                weights[rid] = w
            else:
                weights[rid] = 1.0
    return weights


# ---------------------------------------------------------------------------
# Trajectory parsing — split multi-replica extxyz into per-replica sequences
# ---------------------------------------------------------------------------

def split_replica_trajectory(frames: list[aq.Frame]) -> dict[int, list[aq.Frame]]:
    """Split a multi-replica extxyz trajectory into per-replica frame lists.

    Each frame header must contain a ``Replica=N`` attribute (written by
    dump_qct).  Frames are grouped by replica id in order of appearance.
    """
    groups: dict[int, list[aq.Frame]] = {}
    for frame in frames:
        rid_str = frame.metadata.get("replica")
        if rid_str is None:
            raise ValueError(
                "Trajectory frames must contain a 'Replica' attribute. "
                "Ensure dump_qct was used with ensemble qct."
            )
        rid = int(rid_str)
        groups.setdefault(rid, []).append(frame)
    return groups


# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------

def _com_position(frame: aq.Frame, params: dict[str, Any] | None = None) -> float:
    """Center-of-mass position magnitude (used for diffusion-type correlations).
    Optional ``atom_indices`` in params selects a subset of atoms."""
    atom_indices = params.get("atom_indices") if params else None
    m = frame.masses
    if atom_indices is not None:
        idx = np.asarray(atom_indices, dtype=int)
        m = m[idx]
        pos = frame.positions[idx]
    else:
        pos = frame.positions
    com = np.sum(m[:, None] * pos, axis=0) / np.sum(m)
    return float(np.linalg.norm(com))


def _com_velocity(frame: aq.Frame, params: dict[str, Any] | None = None) -> float:
    """Center-of-mass speed.
    Optional ``atom_indices`` in params selects a subset of atoms."""
    atom_indices = params.get("atom_indices") if params else None
    m = frame.masses
    if atom_indices is not None:
        idx = np.asarray(atom_indices, dtype=int)
        m = m[idx]
        vel = frame.velocities[idx] if frame.velocities is not None else None
    else:
        vel = frame.velocities
    if vel is None:
        return 0.0
    com_v = np.sum(m[:, None] * vel, axis=0) / np.sum(m)
    return float(np.linalg.norm(com_v))


def _bond_length(frame: aq.Frame, params: dict[str, Any]) -> float:
    """Distance between two atoms specified by ``atom1`` and ``atom2`` (0-based)."""
    i = int(params["atom1"])
    j = int(params["atom2"])
    delta = frame.positions[j] - frame.positions[i]
    # Minimum image if periodic
    if frame.lattice is not None and np.any(frame.pbc):
        frac = np.linalg.solve(frame.lattice.T, delta)
        frac[frame.pbc] -= np.rint(frac[frame.pbc])
        delta = frac @ frame.lattice
    return float(np.linalg.norm(delta))


def _kinetic_energy(frame: aq.Frame, params: dict[str, Any] | None = None) -> float:
    """Total kinetic energy in eV.
    Optional ``atom_indices`` in params selects a subset of atoms."""
    if frame.velocities is None:
        return 0.0
    atom_indices = params.get("atom_indices") if params else None
    m = frame.masses
    v = frame.velocities
    if atom_indices is not None:
        idx = np.asarray(atom_indices, dtype=int)
        m = m[idx]
        v = v[idx]
    return 0.5 * np.sum(m[:, None] * v**2) * AMU_A2_FS2_TO_EV


def _position_component(frame: aq.Frame, params: dict[str, Any]) -> float:
    """Single position component: ``atom`` (0-based), ``axis`` (0=x,1=y,2=z)."""
    i = int(params["atom"])
    d = int(params.get("axis", 0))
    return float(frame.positions[i, d])


def _velocity_component(frame: aq.Frame, params: dict[str, Any]) -> float:
    """Single velocity component: ``atom`` (0-based), ``axis`` (0=x,1=y,2=z)."""
    if frame.velocities is None:
        return 0.0
    i = int(params["atom"])
    d = int(params.get("axis", 0))
    return float(frame.velocities[i, d])


def _point_charge_dipole(frame: aq.Frame, params: dict[str, Any]) -> float:
    """Point-charge dipole along a given axis.

    Requires ``charges`` in frame metadata or a ``charges`` array in params.
    ``axis`` selects 0=x, 1=y, 2=z.
    """
    d = int(params.get("axis", 0))
    charges = params.get("charges")
    if charges is None:
        # Try reading from frame metadata (GPUMD can write charges)
        q_str = frame.metadata.get("charges")
        if q_str:
            charges = np.asarray([float(x) for x in q_str.split()])
        else:
            raise ValueError("point_charge_dipole requires 'charges' in operator params or frame metadata")
    else:
        charges = np.asarray(charges)
    return float(np.sum(charges * frame.positions[:, d]))


OPERATOR_REGISTRY: dict[str, Callable[[aq.Frame, dict[str, Any]], float]] = {
    "position": _position_component,
    "velocity": _velocity_component,
    "com_position": _com_position,
    "com_velocity": _com_velocity,
    "bond_length": _bond_length,
    "kinetic_energy": _kinetic_energy,
    "point_charge_dipole": _point_charge_dipole,
}


def make_operator(spec: dict[str, Any]) -> Callable[[aq.Frame], float]:
    """Build a callable operator from a JSON spec.

    Expected format::

        {"name": "bond_length", "params": {"atom1": 0, "atom2": 1}}
        {"name": "position", "params": {"atom": 0, "axis": 0}}
        {"name": "velocity", "params": {"atom": 0, "axis": 0}}
    """
    name = spec["name"]
    params = spec.get("params", {})
    if name not in OPERATOR_REGISTRY:
        raise ValueError(
            f"Unknown operator '{name}'.  Available: {sorted(OPERATOR_REGISTRY)}"
        )
    func = OPERATOR_REGISTRY[name]

    def op(frame: aq.Frame) -> float:
        return func(frame, params)

    return op


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------

def compute_correlation(
    replica_trajs: dict[int, list[aq.Frame]],
    weights: dict[int, float],
    op_a: Callable[[aq.Frame], float],
    op_b: Callable[[aq.Frame], float],
    max_lag_fs: float | None = None,
) -> CorrelationResult:
    """Compute the LSC-IVR correlation function ``C_AB(t) = <w A(0) B(t)> / <w>``.

    Parameters
    ----------
    replica_trajs : per-replica frame lists, each of equal length.
    weights : per-replica Wigner weight (default 1.0 if missing).
    op_a, op_b : operator callables acting on a single Frame.
    max_lag_fs : maximum correlation time in fs.  If None, uses full trajectory.

    Returns
    -------
    CorrelationResult with time_fs, c_ab, c_ab_normalized, std_error, n_samples.
    """
    if not replica_trajs:
        raise ValueError("No replica trajectories provided.")

    # Determine the number of time points and dt from the first replica
    first_replica = sorted(replica_trajs.keys())[0]
    first_frames = replica_trajs[first_replica]
    n_frames = len(first_frames)
    if n_frames < 2:
        raise ValueError("Need at least 2 frames per replica to compute correlation.")

    # Time axis from frame metadata
    times = np.array([
        aq.frame_time(f, idx, None, None) if f.time_fs is None else f.time_fs
        for idx, f in enumerate(first_frames)
    ])
    if np.all(np.isnan(times)):
        raise ValueError(
            "Cannot determine time axis: frames have no 'time' attribute "
            "and no time_step/dump_interval provided."
        )
    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        raise ValueError(f"Non-positive dt ({dt}) inferred from trajectory.")

    max_lag = n_frames - 1
    if max_lag_fs is not None:
        max_lag = min(max_lag, int(max_lag_fs / dt) + 1)

    # Compute A(0) for each replica
    a0_values: dict[int, float] = {}
    for rid, frames in replica_trajs.items():
        a0_values[rid] = op_a(frames[0])

    # Accumulate weighted correlation
    c_ab = np.zeros(max_lag, dtype=float)
    weight_sum = np.zeros(max_lag, dtype=float)
    n_samples = np.zeros(max_lag, dtype=int)
    # For standard error: accumulate sum of (w_i * A(0)_i * B(t)_i)^2
    c_ab_sq = np.zeros(max_lag, dtype=float)

    for rid, frames in replica_trajs.items():
        w = weights.get(rid, 1.0)
        a0 = a0_values[rid]
        for lag in range(max_lag):
            if lag >= len(frames):
                break
            bt = op_b(frames[lag])
            contrib = w * a0 * bt
            c_ab[lag] += contrib
            weight_sum[lag] += w
            c_ab_sq[lag] += contrib * contrib
            n_samples[lag] += 1

    # Normalize: C_AB(t) = Σ w_i A(0)_i B(t)_i / Σ w_i
    # Use the weight_sum at lag=0 for normalization (Kubo-style normalization)
    norm = weight_sum[0]
    if norm == 0.0:
        norm = 1.0

    c_normalized = c_ab / norm

    # Standard error of the mean: σ/sqrt(N)
    # σ² = [<w² A² B²> - <w A B>²] / N
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = np.where(
            n_samples > 1,
            np.maximum(c_ab_sq / np.maximum(weight_sum, 1e-30) - (c_ab / np.maximum(weight_sum, 1e-30))**2, 0.0),
            0.0,
        )
        std_error = np.sqrt(variance / np.maximum(n_samples, 1))

    time_axis = times[:max_lag] - times[0]

    return CorrelationResult(
        time_fs=time_axis,
        c_ab=c_ab,
        c_ab_normalized=c_normalized,
        std_error=std_error,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# FFT spectral density
# ---------------------------------------------------------------------------

def compute_spectrum(
    corr: CorrelationResult,
    dt_fs: float,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectral density via FFT of the correlation function.

    Returns (frequency_THz, wavenumber_cm_inv, intensity).
    """
    n = len(corr.c_ab_normalized)
    if n < 4:
        raise ValueError("Need at least 4 points for FFT.")

    signal = corr.c_ab_normalized.copy()

    # Apply window function
    if window == "hann":
        w = np.hanning(n)
    elif window == "hamming":
        w = np.hamming(n)
    elif window == "bartlett":
        w = np.bartlett(n)
    elif window == "none" or window is None:
        w = np.ones(n)
    else:
        raise ValueError(f"Unknown window function '{window}'")
    signal_windowed = signal * w

    # FFT
    spectrum = np.fft.rfft(signal_windowed)
    intensity = np.abs(spectrum) ** 2

    # Frequency axis
    freq_cycles_per_fs = np.fft.rfftfreq(n, d=dt_fs)
    freq = freq_cycles_per_fs * 1.0e3  # convert cycles/fs to THz
    wavenumber = freq * 33.35641  # THz -> cm^-1

    return freq, wavenumber, intensity


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_correlation_csv(
    path: Path,
    corr: CorrelationResult,
    operator_a_name: str = "A",
    operator_b_name: str = "B",
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_fs",
            "C_AB_raw",
            "C_AB_normalized",
            "std_error",
            "n_samples",
        ])
        for i in range(len(corr.time_fs)):
            writer.writerow([
                f"{corr.time_fs[i]:.6f}",
                f"{corr.c_ab[i]:.12e}",
                f"{corr.c_ab_normalized[i]:.12e}",
                f"{corr.std_error[i]:.12e}",
                int(corr.n_samples[i]),
            ])
    print(f"Wrote correlation data to {path}")


def write_spectrum_csv(
    path: Path,
    freq: np.ndarray,
    wavenumber: np.ndarray,
    intensity: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_THz", "wavenumber_cm_inv", "intensity"])
        for i in range(len(freq)):
            writer.writerow([
                f"{freq[i]:.6f}",
                f"{wavenumber[i]:.6f}",
                f"{intensity[i]:.12e}",
            ])
    print(f"Wrote spectral density to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trajectory", required=True, type=Path,
        help="Multi-replica QCT trajectory extxyz (dump_qct output).",
    )
    parser.add_argument(
        "--summary", type=Path,
        help="qct_initial_summary.csv with wigner_weight column.",
    )
    parser.add_argument(
        "--config", type=Path,
        help="JSON config file defining operator_A and operator_B.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("qct_lsc_correlation.csv"),
        help="Output correlation CSV file.",
    )
    parser.add_argument(
        "--fft", type=Path, default=None,
        help="If specified, write FFT spectral density to this CSV file.",
    )
    parser.add_argument(
        "--max-lag", type=float, default=None,
        help="Maximum correlation lag time in fs. Default: full trajectory.",
    )
    parser.add_argument(
        "--window", default="hann",
        choices=["hann", "hamming", "bartlett", "none"],
        help="Window function for FFT. Default: hann.",
    )
    parser.add_argument(
        "--time-step", type=float, default=None,
        help="MD time step in fs (used if trajectory has no Time= attribute).",
    )
    parser.add_argument(
        "--dump-interval", type=int, default=None,
        help="Trajectory dump interval in MD steps (used if no Time= attribute).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration
    config: dict[str, Any] = {}
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))

    operator_a_spec = config.get("operator_A", {"name": "position", "params": {"atom": 0, "axis": 0}})
    operator_b_spec = config.get("operator_B", {"name": "position", "params": {"atom": 0, "axis": 0}})
    op_a = make_operator(operator_a_spec)
    op_b = make_operator(operator_b_spec)

    # Load weights
    weights = load_wigner_weights(args.summary) if args.summary else {}

    # Load and split trajectory
    print(f"Reading trajectory: {args.trajectory}")
    frames = aq.read_extxyz(args.trajectory)
    print(f"  {len(frames)} total frames")
    replica_trajs = split_replica_trajectory(frames)
    print(f"  {len(replica_trajs)} replicas")

    # Fill in missing time info if needed
    if args.time_step is not None and args.dump_interval is not None:
        for rid, rframes in replica_trajs.items():
            for idx, f in enumerate(rframes):
                if f.time_fs is None:
                    f.time_fs = (idx + 1) * args.time_step * args.dump_interval

    # Fill in default weights
    for rid in replica_trajs:
        if rid not in weights:
            weights[rid] = 1.0
            print(f"  Warning: replica {rid} not in summary, using weight=1.0")

    total_weight = sum(weights.values())
    print(f"  Total weight: {total_weight:.6f}")
    print(f"  Effective replicas: {sum(1 for w in weights.values() if w > 0)}")

    # Compute correlation
    print(f"\nComputing LSC-IVR correlation...")
    print(f"  Operator A: {operator_a_spec}")
    print(f"  Operator B: {operator_b_spec}")
    corr = compute_correlation(
        replica_trajs, weights, op_a, op_b,
        max_lag_fs=args.max_lag,
    )
    print(f"  {len(corr.time_fs)} time points, dt={corr.time_fs[1]-corr.time_fs[0]:.4f} fs")

    # Write correlation
    write_correlation_csv(args.output, corr)

    # FFT
    if args.fft is not None:
        dt = float(corr.time_fs[1] - corr.time_fs[0])
        freq, wavenumber, intensity = compute_spectrum(corr, dt, window=args.window)
        write_spectrum_csv(args.fft, freq, wavenumber, intensity)

    # Summary
    print(f"\nCorrelation at t=0: {corr.c_ab_normalized[0]:.6e}")
    if len(corr.c_ab_normalized) > 1:
        print(f"Correlation at t={corr.time_fs[-1]:.1f} fs: {corr.c_ab_normalized[-1]:.6e}")


if __name__ == "__main__":
    main()
