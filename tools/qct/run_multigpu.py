#!/usr/bin/env python3
"""Multi-GPU launcher for QCT/LSC-IVR replica-parallel calculations.

Each replica runs as an independent ``replicas 1`` GPUMD process. A bounded
worker pool schedules those processes across visible GPUs, validates complete
artifacts, and merges results afterward.

Usage
-----
    python run_multigpu.py --template run_dir --gpumd ~/gpumd/src/gpumd \\
        --num-gpus 4 --total-replicas 5 --workflow hac \
        --shared-eigenvector /path/qct_eigenvector.out --output merged_output/

The tool:
1. Detects available GPUs (via nvidia-smi or CUDA_VISIBLE_DEVICES).
2. Creates one isolated directory per global replica.
3. Rewrites each run with ``replicas 1`` and ``base_seed + replica_id``.
4. Optionally reuses one validated eigenvector file for every replica.
5. Strictly merges complete trajectory or Wigner-weighted HAC artifacts.

The template run.in must contain one active QCT/LSC-IVR ensemble line.
"""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPLICA_PATTERN = re.compile(r"\breplicas\s+(\d+)\b")
SEED_PATTERN = re.compile(r"\bseed\s+(\d+)\b")
PATH_KEYWORDS = {"potential"}
GENERATED_ARTIFACTS = {
    "dos.out",
    "dipole.out",
    "gpumd.log",
    "hac.out",
    "hac_replica.out",
    "hac_reweighting.csv",
    "manifest.json",
    "neighbor.out",
    "qct_eigenvector.out",
    "qct_hessian.out",
    "qct_initial.out",
    "qct_initial.xyz",
    "qct_initial_summary.csv",
    "qct_stationary.xyz",
    "qct_thermo.csv",
    "qct_trajectory.xyz",
    "qct_zpe.csv",
    "sdc.out",
    "thermo.out",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _active_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _detect_workflow(run_in: Path) -> str:
    lines = _active_lines(run_in.read_text(encoding="utf-8"))
    if any(re.match(r"^\s*compute_hac\b", line) for line in lines):
        return "hac"
    if any(re.match(r"^\s*compute_hnemd\b", line) for line in lines):
        return "hnemd"
    if any(re.match(r"^\s*compute_sdc\b", line) for line in lines):
        return "sdc"
    if any(re.match(r"^\s*compute_dos\b", line) for line in lines):
        return "dos"
    if any(re.match(r"^\s*dump_dipole\b", line) for line in lines):
        return "ir"
    if any(re.match(r"^\s*dump_qct\b", line) for line in lines):
        return "trajectory"
    raise ValueError(
        "Could not infer workflow: template needs compute_hac, compute_hnemd, "
        "compute_sdc, compute_dos, dump_dipole, or dump_qct"
    )


def _resolve_run_in_paths(text: str, template: Path) -> str:
    resolved_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            resolved_lines.append(line)
            continue
        tokens = shlex.split(line, comments=True)
        if tokens and tokens[0] in PATH_KEYWORDS and len(tokens) >= 2:
            source = Path(tokens[1]).expanduser()
            if not source.is_absolute():
                source = (template / source).resolve()
            if not source.is_file():
                raise ValueError(f"Referenced {tokens[0]} file was not found: {source}")
            if any(character.isspace() for character in str(source)):
                raise ValueError(f"GPUMD input paths cannot contain whitespace: {source}")
            tokens[1] = str(source)
            line = " ".join(tokens)
        elif tokens and tokens[0] == "ensemble" and len(tokens) >= 3:
            changed = False
            for keyword in ("eigenvector", "modes"):
                if keyword not in tokens:
                    continue
                value_index = tokens.index(keyword) + 1
                if value_index >= len(tokens):
                    raise ValueError(f"Missing path after ensemble {keyword}")
                source = Path(tokens[value_index]).expanduser()
                if not source.is_absolute():
                    source = (template / source).resolve()
                if not source.is_file():
                    raise ValueError(f"Referenced ensemble {keyword} file was not found: {source}")
                if any(character.isspace() for character in str(source)):
                    raise ValueError(f"GPUMD input paths cannot contain whitespace: {source}")
                tokens[value_index] = str(source)
                changed = True
            if changed:
                line = " ".join(tokens)
        resolved_lines.append(line)
    return "\n".join(resolved_lines) + "\n"


def rewrite_run_in(
    src_path: Path,
    dst_path: Path,
    replicas: int,
    seed: int,
    shared_eigenvector: Path | None = None,
    exclude_lowest: int = 3,
) -> None:
    """Rewrite run.in with the given replica count and seed."""
    text = src_path.read_text(encoding="utf-8")

    lines = text.splitlines()
    ensemble_lines = [
        index for index, line in enumerate(lines)
        if re.match(r"^\s*ensemble\s+(?:qct|lsc_ivr)\b", line)
        and not line.lstrip().startswith("#")
    ]
    if len(ensemble_lines) != 1:
        raise ValueError("Template must contain exactly one active ensemble qct/lsc_ivr line.")
    index = ensemble_lines[0]
    line = lines[index]
    if REPLICA_PATTERN.search(line):
        line = REPLICA_PATTERN.sub(f"replicas {replicas}", line, count=1)
    else:
        line = line.rstrip() + f" replicas {replicas}"
    if SEED_PATTERN.search(line):
        line = SEED_PATTERN.sub(f"seed {seed}", line, count=1)
    else:
        line = line.rstrip() + f" seed {seed}"
    if shared_eigenvector is not None:
        if re.search(r"\bmodes\s+\S+", line):
            raise ValueError("--shared-eigenvector cannot be combined with ensemble modes")
        line = re.sub(r"\s+eigenvector\s+\S+", "", line)
        line = re.sub(r"\s+hessian_displacement\s+\S+", "", line)
        line = re.sub(r"\s+exclude_lowest\s+\S+", "", line)
        line = line.rstrip() + f" eigenvector {shared_eigenvector} exclude_lowest {exclude_lowest}"
    lines[index] = line
    text = _resolve_run_in_paths("\n".join(lines) + "\n", src_path.parent)

    dst_path.write_text(text, encoding="utf-8")


def merge_summaries(
    input_files: list[Path],
    output_file: Path,
    replica_ids: list[int] | None = None,
) -> int:
    """Merge per-GPU qct_initial_summary.csv files into one.

    Uses manifest/global replica IDs and never renumbers by input order.
    """
    all_rows = []
    fieldnames = None
    if replica_ids is not None and len(replica_ids) != len(input_files):
        raise ValueError("Summary files and replica IDs have different lengths.")
    for file_index, f in enumerate(input_files):
        if not f.is_file():
            raise ValueError(f"Missing required summary file: {f}")
        with f.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"{f} has no CSV header")
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"Summary schema mismatch in {f}")
            rows = list(reader)
            if replica_ids is not None and len(rows) != 1:
                raise ValueError(f"Expected one replica row in {f}, found {len(rows)}")
            for row in rows:
                row["replica"] = str(
                    replica_ids[file_index] if replica_ids is not None else int(row["replica"])
                )
                all_rows.append(row)

    if len({row["replica"] for row in all_rows}) != len(all_rows):
        raise ValueError("Duplicate global replica IDs in summaries.")

    if fieldnames and all_rows:
        temporary = output_file.with_name(output_file.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        os.replace(temporary, output_file)

    return len(all_rows)


def merge_trajectories(
    input_files: list[Path],
    output_file: Path,
    replica_ids: list[int] | None = None,
) -> int:
    """Merge per-GPU qct_trajectory.xyz files into one.

    Uses manifest/global replica IDs rather than file order.
    """
    seen_keys = set()
    total_frames = 0
    if replica_ids is not None and len(replica_ids) != len(input_files):
        raise ValueError("Trajectory files and replica IDs have different lengths.")
    # Write to a temporary file and atomically replace to avoid leaving
    # a truncated output if the merge fails mid-way.
    import os
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with tmp_file.open("w", encoding="utf-8") as out:
        for file_index, f in enumerate(input_files):
            if not f.is_file():
                raise ValueError(f"Missing required trajectory file: {f}")
            global_replica = replica_ids[file_index] if replica_ids is not None else None
            lines = f.read_text(encoding="utf-8").splitlines()
            cursor = 0
            while cursor < len(lines):
                if not lines[cursor].strip():
                    cursor += 1
                    continue
                try:
                    n = int(lines[cursor].strip())
                except ValueError:
                    cursor += 1
                    continue
                if cursor + n + 1 >= len(lines):
                    raise ValueError(f"Malformed or truncated trajectory file: {f}")
                header = lines[cursor + 1]
                replica_match = re.search(r"Replica=(\d+)", header)
                if replica_match is None:
                    raise ValueError(f"Missing Replica metadata in {f}")
                if global_replica is None:
                    global_replica = int(replica_match.group(1))
                header = re.sub(r"Replica=(\d+)", f"Replica={global_replica}", header, count=1)
                step_match = re.search(r"Step=(-?\d+)", header)
                if step_match is None:
                    raise ValueError(f"Missing Step metadata in {f}")
                key = (global_replica, int(step_match.group(1)))
                if key in seen_keys:
                    raise ValueError(f"Duplicate trajectory frame {key}")
                seen_keys.add(key)
                out.write(lines[cursor] + "\n")
                out.write(header + "\n")
                for i in range(n):
                    out.write(lines[cursor + 2 + i] + "\n")
                cursor += n + 2
                total_frames += 1
            if cursor != len(lines):
                raise ValueError(f"Malformed trajectory file: {f}")

    os.replace(tmp_file, output_file)
    return total_frames




def load_wigner_log_weights(summary_file: Path) -> dict[int, float]:
    """Load one or more replica weights in log space without overflow."""
    if not summary_file.is_file():
        raise ValueError(f"Missing required summary file: {summary_file}")
    with summary_file.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or "replica" not in reader.fieldnames:
            raise ValueError(f"Invalid QCT summary schema: {summary_file}")
        rows = list(reader)
        has_log = "log_wigner_weight" in reader.fieldnames
        has_linear = "wigner_weight" in reader.fieldnames
    if not rows or (not has_log and not has_linear):
        raise ValueError(f"Summary has no Wigner weights: {summary_file}")
    weights: dict[int, float] = {}
    for row in rows:
        replica = int(row["replica"])
        if replica in weights:
            raise ValueError(f"Duplicate Wigner weight for replica {replica}")
        if has_log:
            log_weight = float(row["log_wigner_weight"])
            if math.isnan(log_weight) or log_weight == math.inf:
                raise ValueError(f"Invalid log_wigner_weight for replica {replica}: {log_weight}")
        else:
            linear_weight = float(row["wigner_weight"])
            if not math.isfinite(linear_weight) or linear_weight < 0.0:
                raise ValueError(
                    f"Invalid Wigner weight for replica {replica}: {linear_weight}"
                )
            log_weight = -math.inf if linear_weight == 0.0 else math.log(linear_weight)
        weights[replica] = log_weight
    return weights


def merge_hac(
    hac_files: list[Path],
    summary_files: list[Path],
    output_file: Path,
    replica_ids: list[int] | None = None,
    uncertainty_file: Path | None = None,
    manifest_file: Path | None = None,
    min_effective_replicas: float = 0.0,
    max_normalized_weight: float = 1.0,
    block_size: int | None = None,
) -> dict:
    """Merge hac.out files from multiple GPU runs with Wigner weighting.

    For LSC-IVR thermal conductivity (Green-Kubo), each replica's HAC curve
    must be weighted by its Wigner weight w_i:

        C_merged(t) = sum_i [ w_i * C_i(t) ] / sum_i [ w_i ]

    For classical QCT (no Wigner weights), w_i = 1 for all replicas, so this
    reduces to a simple average.

    hac.out format (11 columns):
        time(ps), hac_xi, hac_xo, hac_yi, hac_yo, hac_z,
        rtc_xi, rtc_xo, rtc_yi, rtc_yo, rtc_z

    Inputs must be complete and share exactly the same schema and time grid.
    """
    import numpy as np

    if len(hac_files) != len(summary_files) or not hac_files:
        raise ValueError("HAC files and summaries must form a non-empty complete set")
    if replica_ids is None:
        replica_ids = list(range(len(hac_files)))
    if len(replica_ids) != len(hac_files) or len(set(replica_ids)) != len(replica_ids):
        raise ValueError("HAC replica IDs are missing or duplicated")

    datasets = []
    log_weights = []
    reference_shape = None
    reference_time = None
    for replica_id, hac_path, summary_path in zip(replica_ids, hac_files, summary_files):
        if not hac_path.is_file():
            raise ValueError(f"Missing required HAC file for replica {replica_id}: {hac_path}")
        try:
            data = np.loadtxt(hac_path)
        except Exception as error:
            raise ValueError(f"Could not load HAC file {hac_path}: {error}") from error
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] != 11:
            raise ValueError(f"HAC file must contain a non-empty 11-column matrix: {hac_path}")
        if not np.all(np.isfinite(data)):
            raise ValueError(f"HAC file contains non-finite data: {hac_path}")
        if data.shape[0] > 1 and not np.all(np.diff(data[:, 0]) > 0.0):
            raise ValueError(f"HAC time grid must be strictly increasing: {hac_path}")
        if reference_shape is None:
            reference_shape = data.shape
            reference_time = data[:, 0].copy()
        elif data.shape != reference_shape:
            raise ValueError(
                f"HAC shape mismatch for replica {replica_id}: "
                f"expected {reference_shape}, got {data.shape}"
            )
        elif not np.allclose(data[:, 0], reference_time, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError(f"HAC time-grid mismatch for replica {replica_id}: {hac_path}")
        summary_weights = load_wigner_log_weights(summary_path)
        if len(summary_weights) != 1:
            raise ValueError(f"HAC task summary must contain one replica row: {summary_path}")
        log_weights.append(next(iter(summary_weights.values())))
        datasets.append(data)

    finite_logs = [value for value in log_weights if math.isfinite(value)]
    if not finite_logs:
        raise ValueError("All HAC Wigner weights are zero")
    offset = max(finite_logs)
    scaled = np.asarray(
        [0.0 if value == -math.inf else math.exp(value - offset) for value in log_weights],
        dtype=float,
    )
    scaled_sum = float(np.sum(scaled))
    if not math.isfinite(scaled_sum) or scaled_sum <= 0.0:
        raise ValueError("HAC Wigner weights have an invalid normalized total")
    normalized = scaled / scaled_sum
    stack = np.stack(datasets, axis=0)
    merged = np.empty(reference_shape, dtype=float)
    merged[:, 0] = reference_time
    merged[:, 1:] = np.tensordot(normalized, stack[:, :, 1:], axes=(0, 0))
    residual = stack[:, :, 1:] - merged[None, :, 1:]
    standard_error = np.sqrt(
        np.sum((normalized[:, None, None] ** 2) * residual * residual, axis=0)
    )
    effective_replicas = float(1.0 / np.sum(normalized * normalized))

    temporary_output = output_file.with_name(output_file.name + ".tmp")
    np.savetxt(temporary_output, merged, fmt="%25.15e")
    os.replace(temporary_output, output_file)

    kappa_replicas = np.empty((len(datasets), reference_shape[0], 4), dtype=float)
    kappa_replicas[:, :, 0] = stack[:, :, 6] + stack[:, :, 7]
    kappa_replicas[:, :, 1] = stack[:, :, 8] + stack[:, :, 9]
    kappa_replicas[:, :, 2] = stack[:, :, 10]
    kappa_replicas[:, :, 3] = np.mean(kappa_replicas[:, :, :3], axis=2)
    kappa_mean = np.tensordot(normalized, kappa_replicas, axes=(0, 0))
    kappa_se = np.sqrt(
        np.sum(
            (normalized[:, None, None] ** 2)
            * (kappa_replicas - kappa_mean[None, :, :]) ** 2,
            axis=0,
        )
    )

    # Blockwise (within-replica) uncertainty for finite-trajectory noise.
    # Separates two sources of uncertainty:
    #   1. Wigner-initial-condition (between-replica) — captured by standard_error/kappa_se above
    #   2. Finite-trajectory (within-replica block) — captured by block variance below
    # Total uncertainty = sqrt(SE_replica^2 + SE_block^2)
    #
    # Block analysis: split each replica's HAC into non-overlapping blocks,
    # compute block-averaged kappa endpoints, and estimate the variance across
    # blocks. This gives a single SE per kappa component (not per time row),
    # which is then broadcast to all time rows for the output file.
    blockwise_kappa_se_flat = np.zeros(4, dtype=float)  # (x, y, z, avg)
    blockwise_hac_se_flat = np.zeros(10, dtype=float)
    n_blocks_total = 0
    if block_size is not None and block_size > 0 and reference_shape[0] >= block_size * 2:
        n_blocks_per_replica = reference_shape[0] // block_size
        n_blocks_total = n_blocks_per_replica * len(datasets)
        if n_blocks_per_replica > 1:
            # Block-averaged kappa endpoint for each replica: (n_blocks, 4)
            block_kappa_endpoints = np.zeros((len(datasets), n_blocks_per_replica, 4), dtype=float)
            block_hac_means = np.zeros((len(datasets), n_blocks_per_replica, 10), dtype=float)
            for i in range(len(datasets)):
                for b in range(n_blocks_per_replica):
                    start = b * block_size
                    end = start + block_size
                    # Block-averaged kappa: mean of kappa over the block's time rows
                    block_kappa_endpoints[i, b] = np.mean(kappa_replicas[i, start:end, :], axis=0)
                    block_hac_means[i, b] = np.mean(stack[i, start:end, 1:11], axis=0)
            # Weighted block mean (over replicas): (n_blocks, 4)
            weighted_block_kappa = np.tensordot(normalized, block_kappa_endpoints, axes=(0, 0))
            weighted_block_hac = np.tensordot(normalized, block_hac_means, axes=(0, 0))
            # Accumulate weighted block variance across replicas
            for i in range(len(datasets)):
                w = normalized[i]
                diff_k = block_kappa_endpoints[i] - weighted_block_kappa  # (n_blocks, 4)
                diff_h = block_hac_means[i] - weighted_block_hac          # (n_blocks, 10)
                blockwise_kappa_se_flat += (w ** 2) * np.var(diff_k, axis=0, ddof=1)
                blockwise_hac_se_flat += (w ** 2) * np.var(diff_h, axis=0, ddof=1)
            # Normalize by sum of w_i^2 (effective replicas)
            w2_sum = max(float(np.sum(normalized ** 2)), 1e-60)
            blockwise_kappa_se_flat = np.sqrt(blockwise_kappa_se_flat / w2_sum)
            blockwise_hac_se_flat = np.sqrt(blockwise_hac_se_flat / w2_sum)

    # Broadcast flat SE to per-time-row for the output file
    blockwise_kappa_se = np.broadcast_to(blockwise_kappa_se_flat, (reference_shape[0], 4)).copy()
    blockwise_hac_se = np.broadcast_to(blockwise_hac_se_flat, (reference_shape[0], 10)).copy()

    # Combined uncertainty: sqrt(SE_replica^2 + SE_block^2)
    combined_kappa_se = np.sqrt(kappa_se ** 2 + blockwise_kappa_se ** 2)
    combined_hac_se = np.sqrt(standard_error ** 2 + blockwise_hac_se ** 2)

    if uncertainty_file is not None:
        temporary_uncertainty = uncertainty_file.with_name(uncertainty_file.name + ".tmp")
        fieldnames = [
            "time_ps", "se_hac_xi", "se_hac_xo", "se_hac_yi", "se_hac_yo", "se_hac_z",
            "se_rtc_xi", "se_rtc_xo", "se_rtc_yi", "se_rtc_yo", "se_rtc_z",
            "se_kappa_x", "se_kappa_y", "se_kappa_z", "se_kappa_avg",
            "block_se_kappa_x", "block_se_kappa_y", "block_se_kappa_z", "block_se_kappa_avg",
            "combined_se_kappa_x", "combined_se_kappa_y", "combined_se_kappa_z", "combined_se_kappa_avg",
        ]
        with temporary_uncertainty.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.writer(output_handle)
            writer.writerow(fieldnames)
            for row_index, time_value in enumerate(reference_time):
                writer.writerow(
                    [f"{time_value:.15e}"]
                    + [f"{value:.15e}" for value in standard_error[row_index]]
                    + [f"{value:.15e}" for value in kappa_se[row_index]]
                    + [f"{value:.15e}" for value in blockwise_kappa_se[row_index]]
                    + [f"{value:.15e}" for value in combined_kappa_se[row_index]]
                )
        os.replace(temporary_uncertainty, uncertainty_file)

    replicas = []
    for index, replica_id in enumerate(replica_ids):
        endpoints = kappa_replicas[index, -1]
        task_manifest_path = hac_files[index].parent / "manifest.json"
        task_manifest = {}
        if task_manifest_path.is_file():
            try:
                task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed task manifest: {task_manifest_path}") from error
        replicas.append({
            "replica": replica_id,
            "seed": task_manifest.get("seed"),
            "status": task_manifest.get("status"),
            "input_hash": task_manifest.get("input_hash"),
            "gpumd_sha256": task_manifest.get("gpumd_sha256"),
            "model_sha256": task_manifest.get("model_sha256"),
            "shared_eigenvector_sha256": task_manifest.get("shared_eigenvector_sha256"),
            "hac_file": str(hac_files[index]),
            "hac_sha256": _sha256_file(hac_files[index]),
            "log_wigner_weight": (
                log_weights[index] if math.isfinite(log_weights[index]) else "-inf"
            ),
            "normalized_weight": float(normalized[index]),
            "kappa_x_end": float(endpoints[0]),
            "kappa_y_end": float(endpoints[1]),
            "kappa_z_end": float(endpoints[2]),
            "kappa_avg_end": float(endpoints[3]),
        })
    diagnostics = {
        "schema": "QCT_HAC_MERGE_v1",
        "replica_count": len(replica_ids),
        "effective_replicas": effective_replicas,
        "max_normalized_weight": float(np.max(normalized)),
        "time_rows": int(reference_shape[0]),
        "replicas": replicas,
        "merged_endpoint": {
            "kappa_x": float(kappa_mean[-1, 0]),
            "kappa_y": float(kappa_mean[-1, 1]),
            "kappa_z": float(kappa_mean[-1, 2]),
            "kappa_avg": float(kappa_mean[-1, 3]),
        },
        "blockwise_uncertainty": {
            "block_size": block_size,
            "n_blocks_total": n_blocks_total,
            "block_se_kappa_avg_end": float(blockwise_kappa_se_flat[3]) if n_blocks_total > 0 else None,
            "combined_se_kappa_avg_end": float(np.sqrt(kappa_se[-1, 3] ** 2 + blockwise_kappa_se_flat[3] ** 2)),
        },
    }
    acceptance_errors = []
    if effective_replicas + 1.0e-12 < min_effective_replicas:
        acceptance_errors.append(
            f"effective replicas {effective_replicas:.3f} < {min_effective_replicas:.3f}"
        )
    if float(np.max(normalized)) > max_normalized_weight + 1.0e-12:
        acceptance_errors.append(
            f"maximum normalized weight {float(np.max(normalized)):.3f} > "
            f"{max_normalized_weight:.3f}"
        )
    diagnostics["acceptance_errors"] = acceptance_errors
    if manifest_file is not None:
        _atomic_json(manifest_file, diagnostics)
    print(
        f"  Merged {len(datasets)} HAC files: N_eff={effective_replicas:.3f}, "
        f"max_weight={float(np.max(normalized)):.3f}"
    )
    if acceptance_errors:
        raise ValueError("HAC acceptance failed: " + "; ".join(acceptance_errors))
    return diagnostics


def merge_dos(
    dos_files: list[Path],
    summary_files: list[Path],
    output_file: Path,
    replica_ids: list[int],
) -> None:
    """Merge dos.out files from multiple replicas with Wigner weighting.

    dos.out format: frequency_THz, dos
    Merged: dos_merged = sum_i w_i * dos_i / sum_i w_i
    """
    import numpy as np

    if not dos_files:
        return
    datasets = []
    log_weights = []
    ref_freq = None
    for rid, dos_path, summary_path in zip(replica_ids, dos_files, summary_files):
        if not dos_path.is_file():
            raise ValueError(f"Missing DOS file for replica {rid}: {dos_path}")
        data = np.loadtxt(dos_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if ref_freq is None:
            ref_freq = data[:, 0].copy()
        elif not np.allclose(data[:, 0], ref_freq, rtol=1e-12, atol=1e-14):
            raise ValueError(f"DOS frequency grid mismatch for replica {rid}")
        weights = load_wigner_log_weights(summary_path)
        log_weights.append(next(iter(weights.values())))
        datasets.append(data)

    finite_logs = [v for v in log_weights if math.isfinite(v)]
    offset = max(finite_logs) if finite_logs else 0.0
    scaled = np.asarray(
        [0.0 if v == -math.inf else math.exp(v - offset) for v in log_weights],
        dtype=float,
    )
    normalized = scaled / np.sum(scaled)
    stack = np.stack(datasets, axis=0)
    merged = np.empty_like(datasets[0])
    merged[:, 0] = ref_freq
    merged[:, 1:] = np.tensordot(normalized, stack[:, :, 1:], axes=(0, 0))

    tmp = output_file.with_name(output_file.name + ".tmp")
    np.savetxt(tmp, merged, fmt="%25.15e")
    os.replace(tmp, output_file)
    n_eff = 1.0 / np.sum(normalized ** 2)
    print(f"  Merged {len(datasets)} DOS files: N_eff={n_eff:.3f}")


def merge_dipole(
    dipole_files: list[Path],
    summary_files: list[Path],
    output_file: Path,
    replica_ids: list[int],
) -> None:
    """Merge dipole.out files from multiple replicas with Wigner weighting.

    dipole.out format: step, dx, dy, dz (optionally with replica column)
    Merged output keeps per-replica data but writes a summary file with
    weighted autocorrelation results.
    """
    import numpy as np

    if not dipole_files:
        return
    datasets = []
    log_weights = []
    for rid, dip_path, summary_path in zip(replica_ids, dipole_files, summary_files):
        if not dip_path.is_file():
            raise ValueError(f"Missing dipole file for replica {rid}: {dip_path}")
        data = np.loadtxt(dip_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        datasets.append(data)
        weights = load_wigner_log_weights(summary_path)
        log_weights.append(next(iter(weights.values())))

    finite_logs = [v for v in log_weights if math.isfinite(v)]
    offset = max(finite_logs) if finite_logs else 0.0
    scaled = np.asarray(
        [0.0 if v == -math.inf else math.exp(v - offset) for v in log_weights],
        dtype=float,
    )
    normalized = scaled / np.sum(scaled)

    # Write a merged dipole file with weighted average dipole at each step
    min_rows = min(d.shape[0] for d in datasets)
    merged = np.zeros((min_rows, 4), dtype=float)
    merged[:, 0] = datasets[0][:min_rows, 0]  # step column
    for axis in range(1, min(4, datasets[0].shape[1])):
        for i, (d, w) in enumerate(zip(datasets, normalized)):
            merged[:, axis] += w * d[:min_rows, axis]

    tmp = output_file.with_name(output_file.name + ".tmp")
    np.savetxt(tmp, merged, fmt="%25.15e")
    os.replace(tmp, output_file)
    n_eff = 1.0 / np.sum(normalized ** 2)
    print(f"  Merged {len(datasets)} dipole files: N_eff={n_eff:.3f}")


def merge_hnemd(
    thermo_files: list[Path],
    summary_files: list[Path],
    output_file: Path,
    replica_ids: list[int],
) -> None:
    """Merge NEMD thermo output with Wigner weighting.

    For NEMD, the thermal conductivity is proportional to the average heat
    flux: kappa = -<J> / f_ext. Each replica's contribution is weighted by
    its Wigner factor.

    thermo.out format: step, temperature, Kx, Ky, Kz, Px, Py, Pz, ...
    We merge the temperature and pressure columns with Wigner weights.
    """
    import numpy as np

    if not thermo_files:
        return
    datasets = []
    log_weights = []
    for rid, thermo_path, summary_path in zip(replica_ids, thermo_files, summary_files):
        if not thermo_path.is_file():
            raise ValueError(f"Missing thermo file for replica {rid}: {thermo_path}")
        data = np.loadtxt(thermo_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        datasets.append(data)
        weights = load_wigner_log_weights(summary_path)
        log_weights.append(next(iter(weights.values())))

    finite_logs = [v for v in log_weights if math.isfinite(v)]
    offset = max(finite_logs) if finite_logs else 0.0
    scaled = np.asarray(
        [0.0 if v == -math.inf else math.exp(v - offset) for v in log_weights],
        dtype=float,
    )
    normalized = scaled / np.sum(scaled)

    min_rows = min(d.shape[0] for d in datasets)
    n_cols = datasets[0].shape[1]
    merged = np.zeros((min_rows, n_cols), dtype=float)
    merged[:, 0] = datasets[0][:min_rows, 0]  # step column
    for col in range(1, n_cols):
        for d, w in zip(datasets, normalized):
            merged[:, col] += w * d[:min_rows, col]

    tmp = output_file.with_name(output_file.name + ".tmp")
    np.savetxt(tmp, merged, fmt="%25.15e")
    os.replace(tmp, output_file)
    n_eff = 1.0 / np.sum(normalized ** 2)
    print(f"  Merged {len(datasets)} NEMD thermo files: N_eff={n_eff:.3f}")


def merge_sdc(
    sdc_files: list[Path],
    summary_files: list[Path],
    output_file: Path,
    replica_ids: list[int],
) -> None:
    """Merge sdc.out files from multiple replicas with Wigner weighting.

    sdc.out format (7 columns per group, single group):
        time_ps, msd_x, msd_y, msd_z, sdc_x, sdc_y, sdc_z

    For multi-group runs, columns 1–6 repeat for each group.
    Merged: sdc_merged = sum_i w_i * sdc_i / sum_i w_i
    """
    import numpy as np

    if not sdc_files:
        return
    datasets = []
    log_weights = []
    ref_time = None
    for rid, sdc_path, summary_path in zip(replica_ids, sdc_files, summary_files):
        if not sdc_path.is_file():
            raise ValueError(f"Missing SDC file for replica {rid}: {sdc_path}")
        data = np.loadtxt(sdc_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if ref_time is None:
            ref_time = data[:, 0].copy()
        elif not np.allclose(data[:, 0], ref_time, rtol=1e-12, atol=1e-14):
            raise ValueError(f"SDC time-grid mismatch for replica {rid}")
        weights = load_wigner_log_weights(summary_path)
        log_weights.append(next(iter(weights.values())))
        datasets.append(data)

    finite_logs = [v for v in log_weights if math.isfinite(v)]
    offset = max(finite_logs) if finite_logs else 0.0
    scaled = np.asarray(
        [0.0 if v == -math.inf else math.exp(v - offset) for v in log_weights],
        dtype=float,
    )
    normalized = scaled / np.sum(scaled)
    stack = np.stack(datasets, axis=0)
    merged = np.empty_like(datasets[0])
    merged[:, 0] = ref_time
    merged[:, 1:] = np.tensordot(normalized, stack[:, :, 1:], axes=(0, 0))

    tmp = output_file.with_name(output_file.name + ".tmp")
    np.savetxt(tmp, merged, fmt="%25.15e")
    os.replace(tmp, output_file)
    n_eff = 1.0 / np.sum(normalized ** 2)
    print(f"  Merged {len(datasets)} SDC files: N_eff={n_eff:.3f}")


def _visible_device_tokens(requested: int | None) -> list[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env and env.strip():
        tokens = [token.strip() for token in env.split(",") if token.strip()]
    else:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            tokens = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            tokens = []
    if not tokens:
        raise ValueError("No visible GPU devices were detected.")
    count = len(tokens) if requested is None else requested
    if count <= 0 or count > len(tokens):
        raise ValueError(f"Requested {count} GPUs, but only {len(tokens)} are visible.")
    return tokens[:count]


def _cuda_process_environment(device: str | None = None) -> dict[str, str]:
    """Build a CUDA environment that works with native WSL driver libraries.

    WSL exposes the Windows driver through ``/usr/lib/wsl/lib``. Container
    launchers may also leave ``NVIDIA_VISIBLE_DEVICES`` pointing at an invalid
    namespace, so it must not be inherited by GPUMD. The launcher still sets
    ``CUDA_VISIBLE_DEVICES`` explicitly when scheduling a replica.
    """
    environment = os.environ.copy()
    environment.pop("NVIDIA_VISIBLE_DEVICES", None)
    wsl_library_path = Path("/usr/lib/wsl/lib")
    if wsl_library_path.is_dir():
        current = environment.get("LD_LIBRARY_PATH", "")
        entries = [entry for entry in current.split(":") if entry]
        if str(wsl_library_path) not in entries:
            entries.insert(0, str(wsl_library_path))
        environment["LD_LIBRARY_PATH"] = ":".join(entries)
    if device is None:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = device
    return environment


def _merge_csv_artifact(files: list[Path], output: Path, replica_ids: list[int]) -> None:
    if len(files) != len(replica_ids):
        raise ValueError("CSV files and replica IDs have different lengths.")
    fieldnames = None
    rows = []
    for file_index, path in enumerate(files):
        if not path.is_file():
            raise ValueError(f"Missing required artifact: {path}")
        with path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames is None:
                raise ValueError(f"{path} has no CSV header")
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"CSV schema mismatch in {path}")
            file_rows = list(reader)
            if not file_rows:
                raise ValueError(f"{path} has no data rows")
            if "replica" not in reader.fieldnames:
                raise ValueError(f"{path} is missing replica column")
            for row in file_rows:
                row["replica"] = str(replica_ids[file_index])
                rows.append(row)
    expected = {str(replica_id) for replica_id in replica_ids}
    actual = {row["replica"] for row in rows}
    if actual != expected:
        raise ValueError(
            f"Merged CSV replica IDs mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def _copy_template(template: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for item in template.iterdir():
        if item.name == "run.in" or item.name in GENERATED_ARTIFACTS:
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _required_artifacts(workflow: str) -> tuple[str, ...]:
    if workflow == "hac":
        return ("qct_initial_summary.csv", "hac.out")
    if workflow == "hnemd":
        return ("qct_initial_summary.csv", "thermo.out")
    if workflow == "sdc":
        return ("qct_initial_summary.csv", "sdc.out")
    if workflow == "dos":
        return ("qct_initial_summary.csv", "dos.out")
    if workflow == "ir":
        return ("qct_initial_summary.csv", "dipole.out")
    if workflow == "trajectory":
        return (
            "qct_initial_summary.csv",
            "qct_initial.xyz",
            "qct_trajectory.xyz",
            "qct_thermo.csv",
        )
    raise ValueError(f"Unsupported workflow: {workflow}")


def _validate_task_artifacts(
    run_dir: Path,
    workflow: str,
    expected_seed: int,
    started_at: float,
) -> None:
    for name in _required_artifacts(workflow):
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty required artifact: {path}")
        if path.stat().st_mtime + 1.0 < started_at:
            raise ValueError(f"Required artifact predates this task: {path}")
    log_path = run_dir / "gpumd.log"
    if not log_path.is_file() or "Finished running GPUMD." not in log_path.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise ValueError(f"GPUMD completion marker is missing: {log_path}")
    summary_path = run_dir / "qct_initial_summary.csv"
    with summary_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row in {summary_path}, found {len(rows)}")
    if int(rows[0].get("seed", -1)) != expected_seed:
        raise ValueError(
            f"Summary seed mismatch in {summary_path}: expected {expected_seed}, "
            f"got {rows[0].get('seed')}"
        )


def _referenced_input_hashes(run_in: Path) -> dict[str, str]:
    referenced_inputs: dict[str, str] = {}
    for line in _active_lines(run_in.read_text(encoding="utf-8")):
        tokens = shlex.split(line)
        referenced: list[Path] = []
        if tokens and tokens[0] in PATH_KEYWORDS and len(tokens) >= 2:
            referenced.append(Path(tokens[1]))
        elif tokens and tokens[0] == "ensemble":
            for keyword in ("eigenvector", "modes"):
                if keyword in tokens and tokens.index(keyword) + 1 < len(tokens):
                    referenced.append(Path(tokens[tokens.index(keyword) + 1]))
        for path in referenced:
            if not path.is_file():
                raise ValueError(f"Referenced input file was not found: {path}")
            resolved = path.resolve()
            referenced_inputs[str(resolved)] = _sha256_file(resolved)
    return dict(sorted(referenced_inputs.items()))


def _task_input_hash(run_in: Path, configuration_hash: str) -> str:
    digest = hashlib.sha256()
    digest.update(configuration_hash.encode("ascii"))
    digest.update(run_in.read_bytes())
    model_path = run_in.parent / "model.xyz"
    if model_path.is_file():
        digest.update(_sha256_file(model_path).encode("ascii"))
    for path, file_hash in _referenced_input_hashes(run_in).items():
        digest.update(path.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    return digest.hexdigest()


def _validate_shared_eigenvector(path: Path, template: Path) -> None:
    model_path = template / "model.xyz"
    if not model_path.is_file():
        raise ValueError("--shared-eigenvector validation requires template/model.xyz")
    with model_path.open(encoding="utf-8") as model_file:
        try:
            number_of_atoms = int(model_file.readline().strip())
        except ValueError as error:
            raise ValueError(f"Invalid atom count in {model_path}") from error
    dimension = number_of_atoms * 3
    expected_size = (dimension + dimension * dimension) * 4
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Shared eigenvector size mismatch: expected {expected_size} bytes for "
            f"{number_of_atoms} atoms, got {actual_size}"
        )


def _is_replica_complete(run_dir: Path, workflow: str, seed: int) -> bool:
    """Check if a replica has already completed successfully.

    Returns True if the manifest shows success AND all required artifacts
    are present and non-empty.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if manifest.get("status") != "completed" or manifest.get("return_code") != 0:
        return False
    # Verify seed matches
    if manifest.get("seed") != seed:
        return False
    # Check required artifacts exist and are non-empty
    for name in _required_artifacts(workflow):
        artifact = run_dir / name
        if not artifact.is_file() or artifact.stat().st_size == 0:
            return False
    # Check gpumd.log has completion marker
    log_path = run_dir / "gpumd.log"
    if not log_path.is_file():
        return False
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if "Finished running GPUMD." not in log_text:
        return False
    return True


def _run_replica_task(
    replica_id: int,
    run_dir: Path,
    gpumd: str,
    device: str,
    gpumd_args: list[str],
    seed: int,
    workflow: str,
    metadata: dict,
) -> int:
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "replica": replica_id,
        "seed": seed,
        "device": device,
        "command": [gpumd, *gpumd_args],
        "status": "running",
        "started_at": time.time(),
        "workflow": workflow,
        **metadata,
    }
    _atomic_json(manifest_path, manifest)
    env = _cuda_process_environment(device)
    log_path = run_dir / "gpumd.log"
    started = manifest["started_at"]
    validation_error = None
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [gpumd, *gpumd_args],
                cwd=run_dir,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = process.wait()
            except KeyboardInterrupt:
                # Kill the entire process group so the child GPUMD process
                # is terminated rather than left running.
                import signal
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=10)
                except Exception:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except Exception:
                        pass
                raise
    except OSError as error:
        validation_error = f"Failed to launch GPUMD: {error}"
        return_code = 127
    if return_code == 0:
        try:
            _validate_task_artifacts(run_dir, workflow, seed, started)
        except (OSError, ValueError) as error:
            validation_error = str(error)
            return_code = 2
    manifest.update({
        "status": "completed" if return_code == 0 else "failed",
        "return_code": return_code,
        "elapsed_s": time.time() - started,
        "validation_error": validation_error,
        "completed_at": time.time(),
    })
    _atomic_json(manifest_path, manifest)
    return return_code


def temperature_sweep(
    template: Path,
    gpumd: Path,
    temperatures: list[float],
    output: Path,
    num_gpus: int | None = None,
    base_seed: int = 12345,
    replicas_per_temp: int = 5,
    gpumd_args: str = "",
    shared_eigenvector: Path | None = None,
    exclude_lowest: int = 3,
) -> None:
    """Run LSC-IVR at multiple temperatures, each as an independent multigpu job.

    Creates one subdirectory per temperature (T_100K/, T_300K/, etc.) under
    the output directory. Each subdirectory is a complete multi-GPU job
    with its own merged HAC output.

    The template run.in must contain ``__TEMPERATURE__`` as a placeholder
    for the Wigner sampling temperature.
    """
    template = template.resolve()
    gpumd = gpumd.expanduser().resolve()
    if not template.is_dir() or not (template / "run.in").is_file():
        raise ValueError("Template must be a directory containing run.in")
    run_in_text = (template / "run.in").read_text(encoding="utf-8")
    if "__TEMPERATURE__" not in run_in_text:
        raise ValueError(
            "Template run.in must contain __TEMPERATURE__ placeholder "
            "for the Wigner sampling temperature"
        )
    if not gpumd.is_file():
        raise ValueError(f"GPUMD executable not found: {gpumd}")

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    for temp in temperatures:
        temp_dir = output / f"T_{int(temp)}K"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create per-temperature template
        temp_template = temp_dir / "template"
        temp_template.mkdir(parents=True, exist_ok=True)
        for item in template.iterdir():
            if item.name == "run.in" or item.name in GENERATED_ARTIFACTS:
                continue
            target = temp_template / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

        # Write temperature-specific run.in
        temp_run_in = run_in_text.replace("__TEMPERATURE__", str(int(temp)))
        (temp_template / "run.in").write_text(temp_run_in, encoding="utf-8")

        # Launch multi-GPU job for this temperature
        print(f"\n=== Temperature {int(temp)} K: {replicas_per_temp} replicas ===")
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--template", str(temp_template),
            "--gpumd", str(gpumd),
            "--total-replicas", str(replicas_per_temp),
            "--num-gpus", str(num_gpus) if num_gpus else "0",
            "--base-seed", str(base_seed),
            "--output", str(temp_dir / "merged"),
            "--work-dir", str(temp_dir / "runs"),
            "--force",
        ]
        if gpumd_args:
            cmd += ["--gpumd-args", gpumd_args]
        if shared_eigenvector:
            cmd += ["--shared-eigenvector", str(shared_eigenvector)]
        cmd += ["--exclude-lowest", str(exclude_lowest)]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  WARNING: Temperature {int(temp)} K failed (exit code {result.returncode})")
        else:
            print(f"  Temperature {int(temp)} K completed successfully")

    # Write summary index
    index_path = output / "temperature_sweep_index.json"
    index = {
        "temperatures_K": [int(t) for t in temperatures],
        "replicas_per_temperature": replicas_per_temp,
        "base_seed": base_seed,
        "directories": [f"T_{int(t)}K" for t in temperatures],
    }
    _atomic_json(index_path, index)
    print(f"\nTemperature sweep complete. Index: {index_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--gpumd", required=True, type=Path)
    parser.add_argument("--total-replicas", required=True, type=int)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=Path("multigpu_output"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing output/work directory")
    parser.add_argument("--resume", action="store_true", help="Reuse completed manifest tasks")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--gpumd-args", default="")
    parser.add_argument(
        "--temperature-sweep", type=str, default=None,
        help="Comma-separated temperatures (e.g. 100,300,800). "
             "Template run.in must contain __TEMPERATURE__ placeholder. "
             "Runs independent multi-GPU jobs for each temperature.",
    )
    parser.add_argument(
        "--replicas-per-temperature", type=int, default=5,
        help="Number of replicas per temperature in sweep mode (default: 5).",
    )
    parser.add_argument(
        "--workflow", choices=("auto", "hac", "hnemd", "sdc", "dos", "ir", "trajectory"), default="auto",
        help="Required artifact set; auto detects from run.in keywords",
    )
    parser.add_argument(
        "--shared-eigenvector", type=Path, default=None,
        help="Validated qct_eigenvector.out reused by every replica",
    )
    parser.add_argument(
        "--block-size", type=int, default=None,
        help="Block size (in time rows) for blockwise HAC uncertainty estimation. "
             "Separates within-replica (finite-trajectory) from between-replica "
             "(Wigner-initial-condition) uncertainty. Default: disabled.",
    )
    parser.add_argument("--exclude-lowest", type=int, default=3)
    parser.add_argument("--min-effective-replicas", type=float, default=None)
    parser.add_argument("--max-normalized-weight", type=float, default=None)
    args = parser.parse_args()

    if args.total_replicas <= 0:
        raise ValueError("--total-replicas must be positive")

    if args.temperature_sweep:
        temperatures = [float(t.strip()) for t in args.temperature_sweep.split(",")]
        if not temperatures:
            raise ValueError("--temperature-sweep requires at least one temperature")
        temperature_sweep(
            template=args.template,
            gpumd=args.gpumd,
            temperatures=temperatures,
            output=args.output,
            num_gpus=args.num_gpus,
            base_seed=args.base_seed,
            replicas_per_temp=args.replicas_per_temperature,
            gpumd_args=args.gpumd_args,
            shared_eigenvector=args.shared_eigenvector,
            exclude_lowest=args.exclude_lowest,
        )
        return
    template = args.template.resolve()
    gpumd = args.gpumd.expanduser().resolve()
    if not template.is_dir() or not (template / "run.in").is_file():
        raise ValueError("Template must be a directory containing run.in")
    if not gpumd.is_file():
        raise ValueError(f"GPUMD executable was not found: {gpumd}")
    workflow = _detect_workflow(template / "run.in") if args.workflow == "auto" else args.workflow
    if workflow == "hac" and not any(
        re.match(r"^\s*compute_hac\b", line)
        for line in _active_lines((template / "run.in").read_text(encoding="utf-8"))
    ):
        raise ValueError("--workflow hac requires an active compute_hac command")
    if workflow == "hnemd" and not any(
        re.match(r"^\s*compute_hnemd\b", line)
        for line in _active_lines((template / "run.in").read_text(encoding="utf-8"))
    ):
        raise ValueError("--workflow hnemd requires an active compute_hnemd command")
    if workflow == "sdc" and not any(
        re.match(r"^\s*compute_sdc\b", line)
        for line in _active_lines((template / "run.in").read_text(encoding="utf-8"))
    ):
        raise ValueError("--workflow sdc requires an active compute_sdc command")
    if workflow == "dos" and not any(
        re.match(r"^\s*compute_dos\b", line)
        for line in _active_lines((template / "run.in").read_text(encoding="utf-8"))
    ):
        raise ValueError("--workflow dos requires an active compute_dos command")
    if workflow == "ir" and not any(
        re.match(r"^\s*dump_dipole\b", line)
        for line in _active_lines((template / "run.in").read_text(encoding="utf-8"))
    ):
        raise ValueError("--workflow ir requires an active dump_dipole command")
    if args.exclude_lowest < 0:
        raise ValueError("--exclude-lowest must be non-negative")
    if args.min_effective_replicas is not None and args.min_effective_replicas < 0.0:
        raise ValueError("--min-effective-replicas must be non-negative")
    if args.max_normalized_weight is not None and not (
        0.0 < args.max_normalized_weight <= 1.0
    ):
        raise ValueError("--max-normalized-weight must be in (0, 1]")
    shared_eigenvector = None
    if args.shared_eigenvector is not None:
        shared_eigenvector = args.shared_eigenvector.expanduser().resolve()
        if not shared_eigenvector.is_file():
            raise ValueError(f"Shared eigenvector was not found: {shared_eigenvector}")
        if any(character.isspace() for character in str(shared_eigenvector)):
            raise ValueError("--shared-eigenvector path cannot contain whitespace")
        _validate_shared_eigenvector(shared_eigenvector, template)
    devices = _visible_device_tokens(args.num_gpus)
    output = args.output.resolve()
    work_root = (args.work_dir or (output / "runs")).resolve()
    if os.path.commonpath([str(template), str(work_root)]) == str(template):
        raise ValueError("--work-dir must not be inside the template directory")
    if output.exists() and not args.force and not args.resume:
        raise ValueError(f"Output already exists; use --force or --resume: {output}")
    if work_root.exists() and not args.force and not args.resume:
        raise ValueError(f"Work directory already exists; use --force or --resume: {work_root}")
    if args.force:
        if output.exists():
            shutil.rmtree(output)
        if work_root.exists():
            shutil.rmtree(work_root)
    output.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    gpumd_args = shlex.split(args.gpumd_args)
    gpumd_hash = _sha256_file(gpumd)
    template_hash = _sha256_file(template / "run.in")
    model_hash = _sha256_file(template / "model.xyz") if (template / "model.xyz").is_file() else None
    shared_hash = _sha256_file(shared_eigenvector) if shared_eigenvector is not None else None
    configuration_payload = {
        "workflow": workflow,
        "gpumd_sha256": gpumd_hash,
        "template_run_sha256": template_hash,
        "model_sha256": model_hash,
        "shared_eigenvector": str(shared_eigenvector) if shared_eigenvector is not None else None,
        "shared_eigenvector_sha256": shared_hash,
        "exclude_lowest": args.exclude_lowest,
        "gpumd_args": gpumd_args,
    }
    configuration_hash = hashlib.sha256(
        json.dumps(configuration_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    tasks = []
    for replica_id in range(args.total_replicas):
        run_dir = work_root / f"replica_{replica_id:06d}"
        manifest_path = run_dir / "manifest.json"
        seed = args.base_seed + replica_id
        if run_dir.exists() and args.resume and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
            existing_run = run_dir / "run.in"
            existing_hash = (
                _task_input_hash(existing_run, configuration_hash) if existing_run.is_file() else None
            )
            existing_references = (
                _referenced_input_hashes(existing_run) if existing_run.is_file() else None
            )
            if (
                manifest.get("status") == "completed"
                and manifest.get("return_code") == 0
                and manifest.get("input_hash") == existing_hash
                and manifest.get("referenced_inputs") == existing_references
            ):
                try:
                    _validate_task_artifacts(
                        run_dir, workflow, seed, float(manifest.get("started_at", 0.0))
                    )
                except (OSError, ValueError):
                    pass
                else:
                    tasks.append(
                        (replica_id, run_dir, True, existing_hash, existing_references)
                    )
                    continue
            shutil.rmtree(run_dir)
        elif run_dir.exists():
            raise ValueError(f"Task directory already exists: {run_dir}")
        _copy_template(template, run_dir)
        rewrite_run_in(
            template / "run.in",
            run_dir / "run.in",
            1,
            seed,
            shared_eigenvector,
            args.exclude_lowest,
        )
        input_hash = _task_input_hash(run_dir / "run.in", configuration_hash)
        referenced_inputs = _referenced_input_hashes(run_dir / "run.in")
        tasks.append((replica_id, run_dir, False, input_hash, referenced_inputs))

    def run_task(task, device):
        replica_id, run_dir, complete, input_hash, referenced_inputs = task
        if complete:
            return replica_id, 0
        seed = args.base_seed + replica_id
        metadata = {
            **configuration_payload,
            "configuration_hash": configuration_hash,
            "input_hash": input_hash,
            "referenced_inputs": referenced_inputs,
        }
        return replica_id, _run_replica_task(
            replica_id,
            run_dir,
            str(gpumd),
            device,
            gpumd_args,
            seed,
            workflow,
            metadata,
        )

    tasks_by_device = [tasks[index::len(devices)] for index in range(len(devices))]

    def run_device_queue(device_index):
        device = devices[device_index]
        return [run_task(task, device) for task in tasks_by_device[device_index]]

    # A worker owns one CUDA device for its full queue. This prevents a short
    # task on one GPU from starting a second process on another, still-busy GPU.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        device_results = list(pool.map(run_device_queue, range(len(devices))))
    results = [result for queue_results in device_results for result in queue_results]
    failed = [(rid, code) for rid, code in results if code != 0]
    if failed:
        raise RuntimeError(f"Replica tasks failed: {failed}")

    if args.no_merge:
        print(f"Completed {len(tasks)} replica tasks in {work_root}")
        return
    output.mkdir(parents=True, exist_ok=True)
    replica_ids = [task[0] for task in tasks]
    run_dirs = [task[1] for task in tasks]
    summary_files = [path / "qct_initial_summary.csv" for path in run_dirs]
    merge_summaries(summary_files, output / "qct_initial_summary.csv", replica_ids)
    if workflow == "trajectory":
        trajectory_files = [path / "qct_trajectory.xyz" for path in run_dirs]
        merge_trajectories(trajectory_files, output / "qct_trajectory.xyz", replica_ids)
        _merge_csv_artifact(
            [path / "qct_thermo.csv" for path in run_dirs],
            output / "qct_thermo.csv",
            replica_ids,
        )
        initial_files = [path / "qct_initial.xyz" for path in run_dirs]
        merge_trajectories(initial_files, output / "qct_initial.xyz", replica_ids)
        zpe_files = [path / "qct_zpe.csv" for path in run_dirs]
        if any(path.is_file() for path in zpe_files):
            if not all(path.is_file() for path in zpe_files):
                raise ValueError("ZPE output is incomplete across replica tasks")
            _merge_csv_artifact(zpe_files, output / "qct_zpe.csv", replica_ids)
    hac_files = [path / "hac.out" for path in run_dirs]
    if workflow == "hac" or any(path.is_file() for path in hac_files):
        min_effective = (
            args.min_effective_replicas
            if args.min_effective_replicas is not None
            else float(min(3, args.total_replicas))
        )
        max_weight = (
            args.max_normalized_weight
            if args.max_normalized_weight is not None
            else (1.0 if args.total_replicas == 1 else 0.5)
        )
        merge_hac(
            hac_files,
            summary_files,
            output / "hac.out",
            replica_ids,
            output / "hac_uncertainty.csv",
            output / "hac_merge_manifest.json",
            min_effective,
            max_weight,
            args.block_size,
        )
    if workflow == "sdc":
        sdc_files = [path / "sdc.out" for path in run_dirs]
        merge_sdc(sdc_files, summary_files, output / "sdc.out", replica_ids)
    if workflow == "dos":
        dos_files = [path / "dos.out" for path in run_dirs]
        merge_dos(dos_files, summary_files, output / "dos.out", replica_ids)
    if workflow == "ir":
        dipole_files = [path / "dipole.out" for path in run_dirs]
        merge_dipole(dipole_files, summary_files, output / "dipole.out", replica_ids)
    if workflow == "hnemd":
        thermo_files = [path / "thermo.out" for path in run_dirs]
        merge_hnemd(thermo_files, summary_files, output / "thermo.out", replica_ids)
    print(f"Merged {len(tasks)} process-per-replica tasks into {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
