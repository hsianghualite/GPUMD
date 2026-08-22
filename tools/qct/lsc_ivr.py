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

    Returns a dict ``{replica_id: weight}``.  Prefers ``log_wigner_weight``
    when available for numerical stability (avoids overflow/underflow that
    can occur in the C++ ``exp()`` call).  Falls back to ``wigner_weight``
    column.  If neither column is present, all weights default to 1.0.
    """
    weights: dict[int, float] = {}
    if not summary_path.is_file():
        return weights
    with summary_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return weights
        has_log_weight = "log_wigner_weight" in reader.fieldnames
        has_weight = "wigner_weight" in reader.fieldnames
        if not has_log_weight and not has_weight:
            # No weight columns: default all to 1.0
            for row in reader:
                weights[int(row["replica"])] = 1.0
            return weights
        rows = list(reader)
        if has_log_weight:
            log_weights: list[tuple[int, float]] = []
            for row in rows:
                rid = int(row["replica"])
                if any(existing_rid == rid for existing_rid, _ in log_weights):
                    raise ValueError(f"Duplicate weight entry for replica {rid}.")
                log_w = float(row["log_wigner_weight"])
                if math.isnan(log_w) or log_w == math.inf:
                    raise ValueError(f"Invalid log_wigner_weight for replica {rid}: {log_w}")
                log_weights.append((rid, log_w))
            finite_logs = [log_w for _, log_w in log_weights if math.isfinite(log_w)]
            if finite_logs and max(finite_logs) > 700.0:
                # Preserve relative weights without overflowing exp(). The
                # common scale cancels in every ratio estimator.
                offset = max(finite_logs)
            else:
                offset = 0.0
            for rid, log_w in log_weights:
                if not math.isfinite(log_w) or log_w - offset < -745.0:
                    weights[rid] = 0.0
                else:
                    weights[rid] = math.exp(log_w - offset)
        else:
            for row in rows:
                rid = int(row["replica"])
                if rid in weights:
                    raise ValueError(f"Duplicate weight entry for replica {rid}.")
                w = float(row["wigner_weight"])
                if math.isnan(w) or w == math.inf or w < 0.0:
                    raise ValueError(f"Invalid wigner_weight for replica {rid}: {w}")
                weights[rid] = w
    return weights


# ---------------------------------------------------------------------------
# Dipole trajectory reader (dump_dipole output)
# ---------------------------------------------------------------------------

@dataclass
class DipoleFrame:
    """One dipole measurement: step, replica (optional), dx, dy, dz."""
    step: int
    replica: int = 0
    dipole: np.ndarray = field(default_factory=lambda: np.zeros(3))


def load_dipole_out(path: Path) -> dict[int, list[DipoleFrame]]:
    """Load dipole.out produced by dump_dipole.

    Supports both single-trajectory and QCT batch formats:

    Single:     step dx dy dz
    Batch:      step replica dx dy dz

    Returns a dict {replica_id: [DipoleFrame, ...]} sorted by step.
    """
    if not path.is_file():
        return {}
    result: dict[int, list[DipoleFrame]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            step = int(parts[0])
            if len(parts) == 4:
                # Single-trajectory: step dx dy dz
                replica = 0
                dip = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            elif len(parts) >= 5:
                # Batch: step replica dx dy dz
                replica = int(parts[1])
                dip = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            else:
                continue
            result.setdefault(replica, []).append(
                DipoleFrame(step=step, replica=replica, dipole=dip)
            )
    # Sort each replica's frames by step
    for rid in result:
        result[rid].sort(key=lambda d: d.step)
    return result


# ---------------------------------------------------------------------------
# Trajectory parsing — split multi-replica extxyz into per-replica sequences
# ---------------------------------------------------------------------------

def split_replica_trajectory(frames: list[aq.Frame]) -> dict[int, list[aq.Frame]]:
    """Split a trajectory into per-replica (or single-trajectory) frame lists.

    For multi-replica QCT trajectories, each frame header must contain a
    ``Replica=N`` attribute (written by dump_qct).  Frames are grouped by
    replica id in order of appearance.

    For single-trajectory input (e.g. RPMD centroid trajectories from
    dump_centroid, or ordinary MD trajectories), frames without a
    ``Replica`` attribute are assigned to replica 0.  This allows
    lsc_ivr.py to compute ordinary Kubo autocorrelation functions from
    a single trajectory without any code changes.
    """
    groups: dict[int, list[aq.Frame]] = {}
    has_replica = False
    has_no_replica = False
    for frame in frames:
        rid_str = frame.metadata.get("replica")
        if rid_str is None:
            rid = 0
            has_no_replica = True
        else:
            rid = int(rid_str)
            has_replica = True
        groups.setdefault(rid, []).append(frame)

    # Mixed replica/non-replica frames indicate a malformed trajectory file.
    if has_replica and has_no_replica:
        raise ValueError(
            "Trajectory contains a mix of frames with and without a 'replica' "
            "attribute. This indicates a malformed or concatenated trajectory. "
            "Please ensure all frames consistently include or omit the replica "
            "attribute."
        )

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


def _nep_dipole(frame: aq.Frame, params: dict[str, Any]) -> float:
    """NEP dipole along a given axis.

    Reads dipole values from a dipole.out file (dump_dipole output).
    The dipole data is attached to the frame metadata as ``_dipole_data``
    by :func:`attach_dipole_data`.  Requires ``axis`` (0=x, 1=y, 2=z) in params.
    """
    d = int(params.get("axis", 0))
    dipole_data = frame.metadata.get("_nep_dipole")
    if dipole_data is None:
        raise ValueError(
            "nep_dipole operator requires --dipole to be specified. "
            "Use dump_dipole in the GPUMD run and pass --dipole dipole.out."
        )
    return float(dipole_data[d])


OPERATOR_REGISTRY: dict[str, Callable[[aq.Frame, dict[str, Any]], float]] = {
    "position": _position_component,
    "velocity": _velocity_component,
    "com_position": _com_position,
    "com_velocity": _com_velocity,
    "bond_length": _bond_length,
    "kinetic_energy": _kinetic_energy,
    "point_charge_dipole": _point_charge_dipole,
    "nep_dipole": _nep_dipole,
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
# Convergence diagnostics (Feature 4)
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceDiagnostics:
    """Summary statistics for LSC-IVR convergence assessment."""
    n_replicas: int
    n_effective: float           # Effective sample size
    weight_sum: float            # Σ w_i
    weight_max: float             # max(w_i)
    weight_min: float             # min(w_i) (excluding zeros)
    weight_mean: float           # mean(w_i)
    weight_std: float            # std(w_i)
    max_to_mean_ratio: float     # max(w_i) / mean(w_i) — large = poor sampling
    c0_values: np.ndarray         # A(0)*B(0) for each replica
    c0_weighted_mean: float      # <w A(0)B(0)> / <w>
    c0_std_error: float           # standard error of C(0)
    convergence_curve: np.ndarray  # C(t)/C(0) vs number of replicas
    convergence_n: np.ndarray     # number of replicas used for each point


def compute_convergence_diagnostics(
    replica_trajs: dict[int, list[aq.Frame]],
    weights: dict[int, float],
    op_a: Callable[[aq.Frame], float],
    op_b: Callable[[aq.Frame], float],
    max_lag: int = 10,
) -> ConvergenceDiagnostics:
    """Compute convergence diagnostics for LSC-IVR sampling.

    Parameters
    ----------
    replica_trajs : per-replica frame lists.
    weights : per-replica Wigner weight.
    op_a, op_b : operator callables.
    max_lag : number of lag points for the convergence curve (default 10).

    Returns
    -------
    ConvergenceDiagnostics dataclass.
    """
    replica_ids = sorted(replica_trajs.keys())
    n_replicas = len(replica_ids)

    w = np.asarray([weights.get(rid, 1.0) for rid in replica_ids], dtype=float)
    if np.any(~np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("Convergence diagnostics received a non-finite or negative weight.")
    if not np.any(w > 0.0):
        raise ValueError("All replica weights are zero; convergence diagnostics are undefined.")
    w_positive = w[w > 0]

    weight_sum = float(np.sum(w))
    n_eff = float(np.sum(w)**2 / np.sum(w**2)) if np.sum(w**2) > 0 else 0.0
    weight_max = float(np.max(w)) if len(w_positive) > 0 else 0.0
    weight_min = float(np.min(w_positive)) if len(w_positive) > 0 else 0.0
    weight_mean = float(np.mean(w_positive)) if len(w_positive) > 0 else 0.0
    weight_std = float(np.std(w_positive)) if len(w_positive) > 0 else 0.0
    max_to_mean = weight_max / weight_mean if weight_mean > 0 else float('inf')

    # Compute A(0)*B(0) for each replica
    c0_values = np.zeros(n_replicas)
    for idx, rid in enumerate(replica_ids):
        frames = replica_trajs[rid]
        if len(frames) > 0:
            c0_values[idx] = op_a(frames[0]) * op_b(frames[0])

    c0_weighted_mean = float(np.sum(w * c0_values) / max(weight_sum, 1e-60))
    # Standard error of the ratio estimator
    if n_replicas > 1:
        residuals = c0_values - c0_weighted_mean
        var_est = np.sum(w**2 * residuals**2) / max(weight_sum**2, 1e-60)
        c0_std_error = float(np.sqrt(var_est))
    else:
        c0_std_error = 0.0

    # Convergence curve: add replicas one by one (sorted by weight descending)
    # and see how C(t=0) stabilizes
    sort_idx = np.argsort(-w)  # descending weight
    n_curve_points = min(max_lag, n_replicas)
    convergence_curve = np.zeros(n_curve_points)
    convergence_n = np.zeros(n_curve_points, dtype=int)
    for i in range(n_curve_points):
        n_use = i + 1
        idx_subset = sort_idx[:n_use]
        w_sub = w[idx_subset]
        c0_sub = c0_values[idx_subset]
        if np.sum(w_sub) > 0:
            convergence_curve[i] = np.sum(w_sub * c0_sub) / np.sum(w_sub)
        else:
            convergence_curve[i] = np.mean(c0_sub)
        convergence_n[i] = n_use

    return ConvergenceDiagnostics(
        n_replicas=n_replicas,
        n_effective=n_eff,
        weight_sum=weight_sum,
        weight_max=weight_max,
        weight_min=weight_min,
        weight_mean=weight_mean,
        weight_std=weight_std,
        max_to_mean_ratio=max_to_mean,
        c0_values=c0_values,
        c0_weighted_mean=c0_weighted_mean,
        c0_std_error=c0_std_error,
        convergence_curve=convergence_curve,
        convergence_n=convergence_n,
    )


def write_diagnostics_csv(
    path: Path,
    diagnostics: ConvergenceDiagnostics,
) -> None:
    """Write convergence diagnostics summary."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["n_replicas", diagnostics.n_replicas])
        writer.writerow(["n_effective", f"{diagnostics.n_effective:.4f}"])
        writer.writerow(["weight_sum", f"{diagnostics.weight_sum:.6e}"])
        writer.writerow(["weight_max", f"{diagnostics.weight_max:.6e}"])
        writer.writerow(["weight_min", f"{diagnostics.weight_min:.6e}"])
        writer.writerow(["weight_mean", f"{diagnostics.weight_mean:.6e}"])
        writer.writerow(["weight_std", f"{diagnostics.weight_std:.6e}"])
        writer.writerow(["max_to_mean_ratio", f"{diagnostics.max_to_mean_ratio:.4f}"])
        writer.writerow(["c0_weighted_mean", f"{diagnostics.c0_weighted_mean:.6e}"])
        writer.writerow(["c0_std_error", f"{diagnostics.c0_std_error:.6e}"])
        writer.writerow([])
        writer.writerow(["n_replicas_used", "C0_estimate"])
        for i in range(len(diagnostics.convergence_n)):
            writer.writerow([
                int(diagnostics.convergence_n[i]),
                f"{diagnostics.convergence_curve[i]:.6e}",
            ])
    print(f"Wrote convergence diagnostics to {path}")


def write_weight_histogram(
    path: Path,
    weights: dict[int, float],
    n_bins: int = 30,
) -> None:
    """Write a histogram of Wigner weights."""
    w = np.array(list(weights.values()))
    w_positive = w[w > 0]
    if len(w_positive) == 0:
        return
    counts, edges = np.histogram(w_positive, bins=n_bins)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_left", "bin_right", "count"])
        for i in range(len(counts)):
            writer.writerow([
                f"{edges[i]:.6e}",
                f"{edges[i+1]:.6e}",
                int(counts[i]),
            ])
    print(f"Wrote weight histogram to {path}")


def attach_dipole_data(
    replica_trajs: dict[int, list[aq.Frame]],
    dipole_data: dict[int, list[DipoleFrame]],
) -> None:
    """Attach per-step dipole data to trajectory frames.

    Each frame's metadata gets ``_nep_dipole`` set to the (dx, dy, dz)
    array from the corresponding dipole.out entry.
    """
    for rid, frames in replica_trajs.items():
        dips = dipole_data.get(rid, [])
        if not dips:
            raise ValueError(f"Missing dipole data for replica {rid}.")
        dipole_by_step: dict[int, np.ndarray] = {}
        for dipole in dips:
            if dipole.step in dipole_by_step:
                raise ValueError(f"Duplicate dipole Step={dipole.step} for replica {rid}.")
            dipole_by_step[dipole.step] = dipole.dipole
        for frame in frames:
            step = _frame_step(frame)
            if step is None or step not in dipole_by_step:
                raise ValueError(f"Missing dipole for replica {rid}, Step={step}.")
            frame.metadata["_nep_dipole"] = dipole_by_step[step]


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------

def _frame_step(frame: aq.Frame) -> int | None:
    """Return the explicit GPUMD step, if present."""
    value = frame.metadata.get("step")
    if value is None:
        return None
    try:
        step = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid Step metadata: {value!r}") from error
    return step


def _trajectory_times(frames: list[aq.Frame], replica_id: int) -> np.ndarray:
    times = np.asarray(
        [float(frame.time_fs) if frame.time_fs is not None else math.nan for frame in frames],
        dtype=float,
    )
    if not np.all(np.isfinite(times)):
        raise ValueError(
            f"Replica {replica_id} is missing finite Time metadata; provide a trajectory with explicit Time values."
        )
    if len(times) > 1:
        differences = np.diff(times)
        if np.any(differences <= 0.0):
            raise ValueError(f"Replica {replica_id} has non-monotonic or duplicate Time values.")
        dt = float(differences[0])
        if not np.allclose(differences, dt, rtol=1.0e-8, atol=1.0e-12):
            raise ValueError(f"Replica {replica_id} does not use an evenly spaced time grid.")
    return times


def _validate_replica_grid(replica_trajs: dict[int, list[aq.Frame]]) -> tuple[np.ndarray, float]:
    if not replica_trajs:
        raise ValueError("No replica trajectories provided.")
    grids: dict[int, np.ndarray] = {}
    for rid, frames in replica_trajs.items():
        if not frames:
            raise ValueError(f"Replica {rid} has no frames.")
        steps = [_frame_step(frame) for frame in frames]
        if any(step is None for step in steps):
            raise ValueError(f"Replica {rid} is missing explicit Step metadata.")
        if steps[0] != 0 or steps != sorted(steps) or len(set(steps)) != len(steps):
            raise ValueError(f"Replica {rid} must start at Step=0 with unique increasing steps.")
        times = _trajectory_times(frames, rid)
        if not math.isclose(times[0], 0.0, abs_tol=1.0e-12):
            raise ValueError(f"Replica {rid} must contain the strict t=0 frame (Time=0).")
        grids[rid] = times
    first_id = sorted(grids)[0]
    first_times = grids[first_id]
    for rid, times in grids.items():
        if len(times) != len(first_times) or not np.allclose(
            times, first_times, rtol=1.0e-8, atol=1.0e-12
        ):
            raise ValueError(f"Replica {rid} has a time grid incompatible with replica {first_id}.")
    dt = float(first_times[1] - first_times[0]) if len(first_times) > 1 else math.nan
    return first_times, dt

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
    times, dt = _validate_replica_grid(replica_trajs)
    n_frames = len(times)
    max_lag = n_frames
    if max_lag_fs is not None:
        if max_lag_fs < 0.0:
            raise ValueError("max_lag_fs must be non-negative.")
        if n_frames > 1:
            max_lag = min(max_lag, int(math.floor(max_lag_fs / dt)) + 1)
        else:
            max_lag = 1
    if max_lag < 1:
        raise ValueError("Correlation must include the t=0 point.")

    replica_ids = sorted(replica_trajs)
    replica_weights: dict[int, float] = {}
    for rid in replica_ids:
        weight = float(weights.get(rid, 1.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Replica {rid} has invalid weight {weight}.")
        replica_weights[rid] = weight
    if sum(replica_weights.values()) <= 0.0:
        raise ValueError("All replica weights are zero; correlation is undefined.")

    # Compute A(0) for each replica
    a0_values: dict[int, float] = {}
    for rid, frames in replica_trajs.items():
        a0_values[rid] = op_a(frames[0])

    # Accumulate weighted correlation using importance-sampling estimator.
    #
    # The LSC-IVR correlation function is:
    #   C_AB(t) = <w A(0) B(t)> / <w>
    #
    # where w_i are Wigner importance weights.  The correct standard error
    # for a ratio estimator <w f> / <w> is obtained from the variance of
    # the *ratio* random variable, not the raw weighted sum.  Specifically:
    #
    #   Var[C_hat] ≈ [ Σ_i w_i² (f_i - C)² ] / (Σ_i w_i)²
    #
    # where f = A(0)*B(t) and C = <w f> / <w>.  This requires a two-pass
    # computation: first compute the mean C_hat, then compute the weighted
    # variance of (f - C_hat) with weights w.
    c_ab = np.zeros(max_lag, dtype=float)
    weight_sum = np.zeros(max_lag, dtype=float)
    n_samples = np.zeros(max_lag, dtype=int)

    # Pass 1: compute weighted mean
    for rid, frames in replica_trajs.items():
        w = replica_weights[rid]
        a0 = a0_values[rid]
        for lag in range(max_lag):
            if lag >= len(frames):
                break
            bt = op_b(frames[lag])
            contrib = w * a0 * bt
            c_ab[lag] += contrib
            weight_sum[lag] += w
            n_samples[lag] += 1

    # Normalize each lag by the weights actually contributing at that lag.
    # This remains correct if a compatibility caller supplies unequal traces.
    if np.any(weight_sum <= 0.0):
        raise ValueError("At least one correlation lag has no positive-weight samples.")
    c_normalized = c_ab / weight_sum

    # Pass 2: compute weighted variance for standard error
    # Var[C_hat(t)] ≈ Σ_i [ w_i² (f_i - C_hat)² ] / (Σ w_i)²
    weighted_var = np.zeros(max_lag, dtype=float)
    for rid, frames in replica_trajs.items():
        w = replica_weights[rid]
        a0 = a0_values[rid]
        for lag in range(max_lag):
            if lag >= len(frames):
                break
            bt = op_b(frames[lag])
            f_i = a0 * bt
            residual = f_i - c_normalized[lag]
            weighted_var[lag] += (w * w) * (residual * residual)

    with np.errstate(invalid="ignore", divide="ignore"):
        variance = np.where(
            n_samples > 1,
            weighted_var / np.maximum(weight_sum * weight_sum, 1e-60),
            0.0,
        )
        std_error = np.sqrt(variance)

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

    Returns (frequency_THz, wavenumber_cm_inv, power_spectrum).
    """
    n = len(corr.c_ab_normalized)
    if n < 4:
        raise ValueError("Need at least 4 points for FFT.")
    if not math.isfinite(dt_fs) or dt_fs <= 0.0:
        raise ValueError("FFT requires a positive finite time step.")
    grid_dt = np.diff(corr.time_fs)
    if len(grid_dt) and not np.allclose(grid_dt, dt_fs, rtol=1.0e-8, atol=1.0e-12):
        raise ValueError("Correlation time grid is not evenly spaced for FFT.")

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
        writer.writerow(["frequency_THz", "wavenumber_cm_inv", "power_spectrum"])
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
        "--dipole", type=Path, default=None,
        help="dipole.out from dump_dipole (for IR spectrum / nep_dipole operator).",
    )
    parser.add_argument(
        "--diagnostics", type=Path, default=None,
        help="If specified, write convergence diagnostics to this CSV file.",
    )
    parser.add_argument(
        "--weight-histogram", type=Path, default=None,
        help="If specified, write a histogram of Wigner weights to this CSV file.",
    )
    parser.add_argument(
        "--ir-spectrum", type=Path, default=None,
        help="Deprecated: IR output is disabled pending a versioned observable contract.",
    )
    parser.add_argument(
        "--dump-interval", type=int, default=None,
        help="Trajectory dump interval in MD steps (used if no Time= attribute).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.ir_spectrum is not None:
        raise ValueError(
            "--ir-spectrum is temporarily disabled until dipole alignment, transform, and units are versioned."
        )

    # Load configuration
    config: dict[str, Any] = {}
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))

    # Support both single-operator (legacy) and multi-operator configs.
    # Legacy: {"operator_A": {...}, "operator_B": {...}}
    # Multi:  {"correlations": [{"name": ..., "operator_A": {...}, "operator_B": {...}}, ...]}
    if "correlations" in config:
        correlations_spec = config["correlations"]
        if not isinstance(correlations_spec, list) or not correlations_spec:
            raise ValueError("'correlations' must be a non-empty list of {name, operator_A, operator_B}")
    else:
        # Legacy single-operator mode: wrap into a list
        correlations_spec = [{
            "name": "default",
            "operator_A": config.get("operator_A", {"name": "position", "params": {"atom": 0, "axis": 0}}),
            "operator_B": config.get("operator_B", {"name": "position", "params": {"atom": 0, "axis": 0}}),
        }]

    # Build operators for all correlations
    correlation_ops = []
    for spec in correlations_spec:
        name = spec.get("name", f"corr_{len(correlation_ops)}")
        op_a_spec = spec.get("operator_A", {"name": "position", "params": {"atom": 0, "axis": 0}})
        op_b_spec = spec.get("operator_B", {"name": "position", "params": {"atom": 0, "axis": 0}})
        op_a = make_operator(op_a_spec)
        op_b = make_operator(op_b_spec)
        correlation_ops.append((name, op_a, op_b, op_a_spec, op_b_spec))

    # Legacy single-operator variables for backward compat with diagnostics
    operator_a_spec = correlation_ops[0][3]
    operator_b_spec = correlation_ops[0][4]
    op_a = correlation_ops[0][1]
    op_b = correlation_ops[0][2]

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
                    f.time_fs = idx * args.time_step * args.dump_interval

    # Fill in default weights
    for rid in replica_trajs:
        if rid not in weights:
            weights[rid] = 1.0
            print(f"  Warning: replica {rid} not in summary, using weight=1.0")

    if any(not math.isfinite(float(w)) or float(w) < 0.0 for w in weights.values()):
        raise ValueError("Summary contains a non-finite or negative replica weight.")
    total_weight = sum(weights.get(rid, 1.0) for rid in replica_trajs)
    if total_weight <= 0.0:
        raise ValueError("All trajectory replica weights are zero.")
    print(f"  Total weight: {total_weight:.6f}")
    print(f"  Effective replicas: {sum(1 for w in weights.values() if w > 0)}")

    # Load dipole data if provided (for IR spectrum / nep_dipole operator)
    dipole_data = {}
    if args.dipole is not None:
        print(f"Reading dipole data: {args.dipole}")
        dipole_data = load_dipole_out(args.dipole)
        print(f"  {sum(len(v) for v in dipole_data.values())} dipole frames, "
              f"{len(dipole_data)} replicas")
        attach_dipole_data(replica_trajs, dipole_data)

    # Convergence diagnostics (Feature 4)
    if args.diagnostics is not None:
        print(f"\nComputing convergence diagnostics...")
        diagnostics = compute_convergence_diagnostics(
            replica_trajs, weights, op_a, op_b)
        write_diagnostics_csv(args.diagnostics, diagnostics)
        print(f"  N_effective = {diagnostics.n_effective:.2f} / {diagnostics.n_replicas}")
        print(f"  Max/mean weight ratio = {diagnostics.max_to_mean_ratio:.4f}")
        if diagnostics.c0_std_error > 0:
            rel_err = abs(diagnostics.c0_std_error / diagnostics.c0_weighted_mean)                 if diagnostics.c0_weighted_mean != 0 else float('inf')
            print(f"  C(0) = {diagnostics.c0_weighted_mean:.6e} ± {diagnostics.c0_std_error:.6e}"
                  f"  ({rel_err:.1%})")

    # Weight histogram (Feature 4)
    if args.weight_histogram is not None:
        write_weight_histogram(args.weight_histogram, weights)

    # Compute correlations (supports multiple operator pairs)
    print(f"\nComputing LSC-IVR correlation(s)...")
    print(f"  {len(correlation_ops)} correlation pair(s) to compute")

    for corr_idx, (corr_name, corr_op_a, corr_op_b, corr_a_spec, corr_b_spec) in enumerate(correlation_ops):
        print(f"\n  [{corr_idx+1}/{len(correlation_ops)}] Correlation: {corr_name}")
        print(f"    Operator A: {corr_a_spec}")
        print(f"    Operator B: {corr_b_spec}")

        corr = compute_correlation(
            replica_trajs, weights, corr_op_a, corr_op_b,
            max_lag_fs=args.max_lag,
        )
        dt_text = "n/a" if len(corr.time_fs) < 2 else f"{corr.time_fs[1]-corr.time_fs[0]:.4f} fs"
        print(f"    {len(corr.time_fs)} time points, dt={dt_text}")

        # Determine output paths
        if len(correlation_ops) == 1:
            corr_output = args.output
            fft_output = args.fft
        else:
            stem = args.output.stem
            suffix = args.output.suffix
            corr_output = args.output.with_name(f"{stem}_{corr_name}{suffix}")
            fft_output = None
            if args.fft is not None:
                fft_stem = args.fft.stem
                fft_suffix = args.fft.suffix
                fft_output = args.fft.with_name(f"{fft_stem}_{corr_name}{fft_suffix}")

        # Write correlation
        write_correlation_csv(corr_output, corr)
        print(f"    Written: {corr_output}")

        # FFT
        if fft_output is not None:
            if len(corr.time_fs) < 2:
                raise ValueError("FFT requires at least two correlation time points.")
            dt = float(corr.time_fs[1] - corr.time_fs[0])
            freq, wavenumber, intensity = compute_spectrum(corr, dt, window=args.window)
            write_spectrum_csv(fft_output, freq, wavenumber, intensity)
            print(f"    FFT: {fft_output}")

        if corr_idx == 0:
            first_corr = corr  # Keep for summary

    # IR spectrum from dipole autocorrelation (Feature 1)
    if args.ir_spectrum is not None:
        if not dipole_data:
            print("\nWarning: --ir-spectrum requires --dipole. Skipping.")
        else:
            print(f"\nComputing IR spectrum from dipole autocorrelation...")
            # Isotropic average: (1/3) * sum over x, y, z of <mu_d(t) * mu_d(0)>
            ir_corrs = []
            for axis in range(3):
                ir_op = make_operator({"name": "nep_dipole", "params": {"axis": axis}})
                ir_corr = compute_correlation(
                    replica_trajs, weights, ir_op, ir_op,
                    max_lag_fs=args.max_lag,
                )
                ir_corrs.append(ir_corr)
            # Average raw numerators and uncertainties consistently across axes.
            avg_corr = ir_corrs[0]
            avg_corr.c_ab = np.mean([c.c_ab for c in ir_corrs], axis=0)
            avg_corr.c_ab_normalized = np.mean(
                [c.c_ab_normalized for c in ir_corrs], axis=0
            )
            avg_corr.std_error = np.sqrt(
                np.mean([c.std_error**2 for c in ir_corrs], axis=0)
            )
            ir_corr_avg = avg_corr
            dt = float(ir_corr_avg.time_fs[1] - ir_corr_avg.time_fs[0])
            freq_ir, wn_ir, intensity_ir = compute_spectrum(ir_corr_avg, dt, window=args.window)
            write_spectrum_csv(args.ir_spectrum, freq_ir, wn_ir, intensity_ir)
            # Also write the dipole autocorrelation
            ir_corr_path = args.ir_spectrum.with_suffix(".correlation.csv")
            write_correlation_csv(ir_corr_path, ir_corr_avg)

    # Summary
    print(f"\nCorrelation at t=0: {first_corr.c_ab_normalized[0]:.6e}")
    if len(first_corr.c_ab_normalized) > 1:
        print(f"Correlation at t={first_corr.time_fs[-1]:.1f} fs: {first_corr.c_ab_normalized[-1]:.6e}")


if __name__ == "__main__":
    main()
