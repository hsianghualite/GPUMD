#!/usr/bin/env python3
"""Multi-GPU launcher for QCT/LSC-IVR replica-parallel calculations.

Each GPU runs an independent GPUMD process with a subset of replicas.
Results are merged afterward. This is the "embarrassingly parallel" approach:
no inter-GPU communication is needed during dynamics because QCT/LSC-IVR
replicas are independent trajectories.

Usage
-----
    python run_multigpu.py --template run_dir --gpumd ~/gpumd/src/gpumd \\
        --num-gpus 4 --total-replicas 128 --output merged_output/

The tool:
1. Detects available GPUs (via nvidia-smi or CUDA_VISIBLE_DEVICES).
2. Divides total_replicas across GPUs.
3. Creates a subdirectory per GPU, each with a modified run.in that uses
   a different seed and replica count.
4. Launches GPUMD processes in parallel (one per GPU).
5. Optionally merges trajectories and summaries after completion.

The template run.in must contain ``ensemble ... replicas N ...`` or
``ensemble ... lsc_ivr ... seed S``.  The tool rewrites the replica count
and seed for each GPU partition.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPLICA_PATTERN = re.compile(r"\breplicas\s+(\d+)\b")
SEED_PATTERN = re.compile(r"\bseed\s+(\d+)\b")
ENSEMBLE_PATTERN = re.compile(r"(ensemble\s+(?:qct|lsc_ivr)[^\n]+)")


def get_gpu_count() -> int:
    """Detect the number of available GPUs."""
    # Try CUDA_VISIBLE_DEVICES first
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip():
        return len(env.split(","))
    # Fall back to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split("\n"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 1


def divide_replicas(total: int, num_gpus: int) -> list[int]:
    """Divide total replicas as evenly as possible across GPUs."""
    base = total // num_gpus
    remainder = total % num_gpus
    counts = [base + (1 if i < remainder else 0) for i in range(num_gpus)]
    return [c for c in counts if c > 0]


def rewrite_run_in(
    src_path: Path,
    dst_path: Path,
    replicas: int,
    seed: int,
) -> None:
    """Rewrite run.in with the given replica count and seed."""
    text = src_path.read_text(encoding="utf-8")

    # Replace replica count
    if REPLICA_PATTERN.search(text):
        text = REPLICA_PATTERN.sub(f"replicas {replicas}", text)
    else:
        # If no replicas keyword, add it to the ensemble line
        match = ENSEMBLE_PATTERN.search(text)
        if match:
            ensemble_line = match.group(1)
            if "replicas" not in ensemble_line:
                ensemble_line += f" replicas {replicas}"
                text = text.replace(match.group(1), ensemble_line)

    # Replace seed
    if SEED_PATTERN.search(text):
        text = SEED_PATTERN.sub(f"seed {seed}", text)
    else:
        match = ENSEMBLE_PATTERN.search(text)
        if match:
            ensemble_line = match.group(1)
            if "seed" not in ensemble_line:
                ensemble_line += f" seed {seed}"
                text = text.replace(match.group(1), ensemble_line)

    dst_path.write_text(text, encoding="utf-8")


def merge_summaries(
    input_files: list[Path],
    output_file: Path,
) -> int:
    """Merge per-GPU qct_initial_summary.csv files into one.

    Re-numbers replicas to be contiguous. Returns the total number of replicas.
    """
    all_rows = []
    fieldnames = None
    offset = 0
    for f in input_files:
        if not f.is_file():
            continue
        with f.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                continue
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                row["replica"] = str(int(row.get("replica", 0)) + offset)
                all_rows.append(row)
        offset += len(all_rows)  # approximate; actual offset should be max replica+1

    # Fix offset calculation: re-number sequentially
    for i, row in enumerate(all_rows):
        row["replica"] = str(i)

    if fieldnames and all_rows:
        with output_file.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    return len(all_rows)


def merge_trajectories(
    input_files: list[Path],
    output_file: Path,
) -> int:
    """Merge per-GPU qct_trajectory.xyz files into one.

    Re-numbers the Replica= field to be contiguous.
    """
    replica_offset = 0
    total_frames = 0
    with output_file.open("w", encoding="utf-8") as out:
        for f in input_files:
            if not f.is_file():
                continue
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
                    break
                # Rewrite the Replica= field in the header line
                header = lines[cursor + 1]
                # Replace Replica=N with Replica=(N+offset)
                def replace_replica(m):
                    return f"Replica={int(m.group(1)) + replica_offset}"

                header = re.sub(r"Replica=(\d+)", replace_replica, header)
                out.write(lines[cursor] + "\n")
                out.write(header + "\n")
                for i in range(n):
                    out.write(lines[cursor + 2 + i] + "\n")
                cursor += n + 2
                total_frames += 1
            # Count replicas in this file (unique Replica= values)
            with f.open() as fh:
                content = fh.read()
            replicas_in_file = set(int(m) for m in re.findall(r"Replica=(\d+)", content))
            replica_offset += len(replicas_in_file)

    return total_frames



def load_wigner_weights(summary_file: Path) -> dict[int, float]:
    """Load Wigner weights from qct_initial_summary.csv.

    Returns a dict mapping replica index -> wigner_weight.
    """
    weights = {}
    if not summary_file.is_file():
        return weights
    with summary_file.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rep = int(row.get("replica", 0))
            w = float(row.get("wigner_weight", 1.0))
            weights[rep] = w
    return weights


def merge_hac(
    hac_files: list[Path],
    summary_files: list[Path],
    output_file: Path,
) -> bool:
    """Merge hac.out files from multiple GPU runs with Wigner weighting.

    For LSC-IVR thermal conductivity (Green-Kubo), each replica's HAC curve
    must be weighted by its Wigner weight w_i:

        C_merged(t) = sum_i [ w_i * C_i(t) ] / sum_i [ w_i ]

    For classical QCT (no Wigner weights), w_i = 1 for all replicas, so this
    reduces to a simple average.

    hac.out format (11 columns):
        time(ps), hac_xi, hac_xo, hac_yi, hac_yo, hac_z,
        rtc_xi, rtc_xo, rtc_yi, rtc_yo, rtc_z

    Returns True if merge was performed, False if no hac.out files found.
    """
    import numpy as np

    # Collect all valid hac.out files
    valid_files = [f for f in hac_files if f.is_file()]
    if not valid_files:
        return False

    # Load Wigner weights from summaries (per-GPU)
    # Each GPU process may have 1 or more replicas; each produces one hac.out
    # For replicas=1 per process, the weight comes from that single replica
    all_weights = []
    for i, (hf, sf) in enumerate(zip(hac_files, summary_files)):
        if not hf.is_file():
            all_weights.append(None)
            continue
        w = load_wigner_weights(sf)
        if w:
            # Use first (or only) replica's weight
            first_rep = min(w.keys())
            all_weights.append(w[first_rep])
        else:
            all_weights.append(1.0)

    # Load all hac.out data
    datasets = []
    max_rows = 0
    for i, f in enumerate(valid_files):
        try:
            data = np.loadtxt(f)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            datasets.append(data)
            max_rows = max(max_rows, data.shape[0])
        except Exception as e:
            print(f"  WARNING: Could not load {f}: {e}")
            all_weights[i] = None
            datasets.append(None)

    # Filter to valid datasets
    valid_pairs = []
    for i, (data, w) in enumerate(zip(datasets, all_weights)):
        if data is not None and w is not None:
            valid_pairs.append((data, w))

    if not valid_pairs:
        print("  WARNING: No valid hac.out data found for merging.")
        return False

    # Truncate all to the minimum number of rows (they should be the same)
    min_rows = min(data.shape[0] for data, _ in valid_pairs)
    ncols = valid_pairs[0][0].shape[1]

    # Weighted average
    total_weight = sum(w for _, w in valid_pairs)
    merged = np.zeros((min_rows, ncols))
    for data, w in valid_pairs:
        merged += w * data[:min_rows, :]
    merged /= total_weight

    # Write merged hac.out
    np.savetxt(output_file, merged, fmt="%25.15e")
    print(f"  Merged {len(valid_pairs)} hac.out files (total weight = {total_weight:.6e})")
    print(f"  Weighted average kappa at t_end:")
    if ncols == 11:
        kx = merged[-1, 6] + merged[-1, 7]
        ky = merged[-1, 8] + merged[-1, 9]
        kz = merged[-1, 10]
        print(f"    kappa_x = {kx:.4f} W/m/K")
        print(f"    kappa_y = {ky:.4f} W/m/K")
        print(f"    kappa_z = {kz:.4f} W/m/K")
        print(f"    kappa_avg = {(kx + ky + kz) / 3:.4f} W/m/K")

    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--template", required=True, type=Path,
        help="Template run directory containing run.in, model.xyz, etc.",
    )
    parser.add_argument(
        "--gpumd", required=True, type=Path,
        help="Path to the gpumd executable.",
    )
    parser.add_argument(
        "--total-replicas", type=int, required=True,
        help="Total number of replicas across all GPUs.",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None,
        help="Number of GPUs to use (default: auto-detect).",
    )
    parser.add_argument(
        "--base-seed", type=int, default=12345,
        help="Base random seed; each GPU gets base_seed + gpu_index * 10000.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("multigpu_output"),
        help="Output directory for merged results.",
    )
    parser.add_argument(
        "--merge", action="store_true", default=True,
        help="Merge trajectories and summaries after completion.",
    )
    parser.add_argument(
        "--no-merge", dest="merge", action="store_false",
        help="Skip merging step.",
    )
    parser.add_argument(
        "--gpumd-args", default="",
        help="Additional arguments to pass to gpumd (e.g. --debug).",
    )
    args = parser.parse_args()

    # Detect GPUs
    if args.num_gpus is not None:
        num_gpus = args.num_gpus
    else:
        num_gpus = get_gpu_count()
    print(f"Using {num_gpus} GPU(s)")

    if num_gpus == 1:
        print("Only 1 GPU available. Running a single GPUMD process.")
        num_gpus = 1

    # Divide replicas
    replica_counts = divide_replicas(args.total_replicas, num_gpus)
    actual_gpus = len(replica_counts)
    print(f"Replica distribution: {replica_counts} (total: {sum(replica_counts)})")

    # Create run directories
    run_dirs = []
    for gpu_idx in range(actual_gpus):
        run_dir = Path(f"gpu_{gpu_idx}")
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        # Copy template files
        for item in args.template.iterdir():
            if item.name == "run.in":
                continue  # will be rewritten
            if item.is_file():
                shutil.copy2(item, run_dir / item.name)
            elif item.is_dir():
                shutil.copytree(item, run_dir / item.name)
        # Rewrite run.in
        seed = args.base_seed + gpu_idx * 10000
        rewrite_run_in(
            args.template / "run.in",
            run_dir / "run.in",
            replica_counts[gpu_idx],
            seed,
        )
        run_dirs.append(run_dir)
        print(f"  GPU {gpu_idx}: {replica_counts[gpu_idx]} replicas, seed={seed}")

    # Launch GPUMD processes
    processes = []
    for gpu_idx, run_dir in enumerate(run_dirs):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        cmd = [str(args.gpumd)]
        if args.gpumd_args:
            cmd.extend(args.gpumd_args.split())
        log_file = (run_dir / "gpumd.log").open("w")
        print(f"  Launching GPU {gpu_idx}: {' '.join(cmd)} in {run_dir}")
        p = subprocess.Popen(
            cmd, cwd=str(run_dir), env=env,
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        processes.append(p)

    # Wait for all processes
    print(f"\nWaiting for {len(processes)} GPUMD processes...")
    start_time = time.time()
    exit_codes = []
    for i, p in enumerate(processes):
        p.wait()
        elapsed = time.time() - start_time
        print(f"  GPU {i}: exited with code {p.returncode} after {elapsed:.1f}s")
        exit_codes.append(p.returncode)

    failed = [i for i, code in enumerate(exit_codes) if code != 0]
    if failed:
        print(f"\nERROR: GPU(s) {failed} failed. Check gpumd.log in their directories.")
        sys.exit(1)

    print(f"\nAll {len(processes)} GPUMD processes completed successfully.")

    # Merge results
    if args.merge:
        args.output.mkdir(parents=True, exist_ok=True)

        # Merge summaries
        summary_files = [d / "qct_initial_summary.csv" for d in run_dirs]
        merged_summary = args.output / "qct_initial_summary.csv"
        n_merged = merge_summaries(summary_files, merged_summary)
        print(f"Merged {n_merged} replica summaries -> {merged_summary}")

        # Merge trajectories
        traj_files = [d / "qct_trajectory.xyz" for d in run_dirs]
        merged_traj = args.output / "qct_trajectory.xyz"
        n_frames = merge_trajectories(traj_files, merged_traj)
        print(f"Merged {n_frames} trajectory frames -> {merged_traj}")

        # Merge thermo
        thermo_files = [d / "qct_thermo.csv" for d in run_dirs]
        merged_thermo = args.output / "qct_thermo.csv"
        with merged_thermo.open("w", encoding="utf-8", newline="") as out:
            writer = None
            offset = 0
            for f in thermo_files:
                if not f.is_file():
                    continue
                with f.open(encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    if writer is None:
                        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    for row in reader:
                        row["replica"] = str(int(row.get("replica", 0)) + offset)
                        writer.writerow(row)
                # Count unique replicas
                with f.open() as fh:
                    content = fh.read()
                replicas_in_file = set()
                for line in content.split("\n")[1:]:  # skip header
                    parts = line.split(",")
                    if len(parts) > 0 and parts[0].strip().isdigit():
                        replicas_in_file.add(int(parts[0]))
                offset += len(replicas_in_file)
        print(f"Merged thermo data -> {merged_thermo}")

        # Copy ZPE file if exists
        zpe_files = [d / "qct_zpe.csv" for d in run_dirs if (d / "qct_zpe.csv").is_file()]
        if zpe_files:
            merged_zpe = args.output / "qct_zpe.csv"
            with merged_zpe.open("w", encoding="utf-8", newline="") as out:
                writer = None
                offset = 0
                for f in zpe_files:
                    with f.open(encoding="utf-8", newline="") as fh:
                        reader = csv.DictReader(fh)
                        if writer is None:
                            writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
                            writer.writeheader()
                        for row in reader:
                            row["replica"] = str(int(row.get("replica", 0)) + offset)
                            writer.writerow(row)
                    # Count unique replicas
                    replicas_in_file = set()
                    for row in csv.DictReader(f.open(encoding="utf-8")):
                        replicas_in_file.add(int(row.get("replica", 0)))
                    offset += len(replicas_in_file)
            print(f"Merged ZPE data -> {merged_zpe}")

        # Merge HAC (thermal conductivity) with Wigner weighting
        hac_files = [d / "hac.out" for d in run_dirs]
        merged_hac = args.output / "hac.out"
        if merge_hac(hac_files, summary_files, merged_hac):
            print(f"Merged HAC (weighted) -> {merged_hac}")
        else:
            print("No hac.out files found; skipping HAC merge.")

        print(f"\nMerged results written to {args.output}/")
        print(f"Use lsc_ivr.py with --trajectory {merged_traj} --summary {merged_summary}")
        print(f"For thermal conductivity: analyze {merged_hac}")


if __name__ == "__main__":
    main()
