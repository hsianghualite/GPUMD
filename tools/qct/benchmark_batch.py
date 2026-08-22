#!/usr/bin/env python3
"""Benchmark the useful QCT batch size on a single GPU.

The tool clones a template run directory, rewrites the ``replicas`` argument of
``ensemble qct`` to each candidate value, runs GPUMD, and records replica-step
throughput and a peak memory reading from ``nvidia-smi``.  The recommendation
logic :func:`recommend_candidate` picks the smallest batch whose throughput is
at least ``threshold`` (default 0.95) of the peak throughput observed across
successful, memory-safe batches.

The CLI is intentionally light: it shells out to ``gpumd`` and ``nvidia-smi``
which must be on PATH.  CPU tests cover recommendation filtering and a fake
executable smoke path without requiring a GPU.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPLICA_PATTERN = re.compile(r"\breplicas\s+(\d+)\b")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, help="Template run directory")
    parser.add_argument("--gpumd", required=True, help="Path to the gpumd executable")
    parser.add_argument(
        "--replicas",
        type=int,
        nargs="+",
        required=True,
        help="Candidate batch sizes (e.g. 1 2 4 8 16 32)",
    )
    parser.add_argument("--steps", type=int, default=10000, help="Number of run steps")
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Drop dump_* lines from the templated run.in to reduce I/O",
    )
    parser.add_argument("--output", default="qct_batch_benchmark.csv", help="Output CSV")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Throughput fraction of peak required for recommendation",
    )
    return parser.parse_args()


def rewrite_run_in(path, replicas, steps, no_output):
    text = path.read_text(encoding="utf-8")
    if REPLICA_PATTERN.search(text) is None:
        raise ValueError(f"{path} does not contain a 'replicas N' argument")
    text = REPLICA_PATTERN.sub(f"replicas {replicas}", text)
    if no_output:
        kept = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("dump_") or stripped.startswith("dump_thermo") or stripped.startswith("dump_xyz") or stripped.startswith("dump_position") or stripped.startswith("dump_velocity"):
                continue
            kept.append(line)
        text = "\n".join(kept) + "\n"
    if re.search(r"\brun\s+\d+\b", text):
        text = re.sub(r"\brun\s+\d+\b", f"run {steps}", text)
    else:
        text = text.rstrip() + f"\nrun {steps}\n"
    path.write_text(text, encoding="utf-8")


def parse_gpumd_throughput(stderr_text):
    match = re.search(
        r"replica-steps per second\s*[:=]?\s*([0-9.eE+-]+)", stderr_text
    )
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def query_process_memory(device_index, process_pid):
    if device_index is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError):
        return None
    for line in result.stdout.splitlines():
        fields = [token.strip() for token in line.split(",")]
        if len(fields) >= 2 and fields[0].isdigit() and int(fields[0]) == process_pid:
            try:
                return float(fields[1])
            except ValueError:
                return None
    return None


def query_device_memory(device_index):
    if device_index is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError):
        return None
    for line in result.stdout.splitlines():
        try:
            return float(line.strip())
        except ValueError:
            continue
    return None


def run_one(candidate, template, gpumd, steps, no_output, device):
    workdir = Path(tempfile.mkdtemp(prefix=f"qct-bench-{candidate}-"))
    try:
        for entry in template.iterdir():
            if entry.is_dir():
                shutil.copytree(entry, workdir / entry.name)
            else:
                shutil.copy2(entry, workdir / entry.name)
        run_in = workdir / "run.in"
        if not run_in.exists():
            return {
                "replicas": candidate,
                "run_replica_steps_per_s": None,
                "return_code": -1,
                "memory_safe_at_85_percent": "unknown",
            }
        rewrite_run_in(run_in, candidate, steps, no_output)
        env = dict(os.environ)
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(device)
        start = time.time()
        proc = subprocess.Popen(
            [gpumd],
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        peak_memory = None
        while proc.poll() is None:
            memory = query_process_memory(device, proc.pid)
            if memory is not None:
                peak_memory = memory if peak_memory is None else max(peak_memory, memory)
            time.sleep(0.1)
        stdout, stderr = proc.communicate()
        elapsed = max(time.time() - start, 1.0e-9)
        throughput = None
        if proc.returncode == 0:
            throughput = parse_gpumd_throughput(stderr + "\n" + stdout)
            if throughput is None and steps > 0:
                throughput = (candidate * steps) / elapsed
        total_memory = query_device_memory(device)
        if peak_memory is None or total_memory is None:
            memory_safe = "unknown"
        else:
            memory_safe = peak_memory <= 0.85 * total_memory
        return {
            "replicas": candidate,
            "run_replica_steps_per_s": throughput,
            "return_code": proc.returncode,
            "memory_safe_at_85_percent": memory_safe,
            "peak_memory_MiB": peak_memory,
            "total_memory_MiB": total_memory,
            "elapsed_s": elapsed,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def recommend_candidate(rows, threshold=0.95):
    """Pick the smallest batch reaching ``threshold`` of peak throughput.

    Only rows with ``return_code == 0`` and ``memory_safe_at_85_percent is True``
    are considered.  Returns ``(recommended, peak)`` where ``peak`` is the row
    with the largest throughput and ``recommended`` is the smallest row whose
    throughput is at least ``threshold * peak``.  If only one row survives
    filtering, ``recommended == peak``.
    """
    valid = [
        row for row in rows
        if row.get("return_code", 0) == 0
        and row.get("memory_safe_at_85_percent") is True
        and isinstance(row.get("run_replica_steps_per_s"), (int, float))
        and math.isfinite(row["run_replica_steps_per_s"])
        and row["run_replica_steps_per_s"] > 0
    ]
    if not valid:
        raise ValueError("No successful and memory-safe benchmark rows")
    peak = max(valid, key=lambda row: row["run_replica_steps_per_s"])
    peak_throughput = peak["run_replica_steps_per_s"]
    target = threshold * peak_throughput
    candidates = [row for row in valid if row["run_replica_steps_per_s"] >= target]
    recommended = min(candidates, key=lambda row: row["replicas"])
    return recommended, peak


def main():
    args = parse_args()
    template = Path(args.template).resolve()
    if not template.is_dir():
        raise ValueError(f"Template directory {template} does not exist")
    if args.threshold <= 0 or args.threshold > 1:
        raise ValueError("--threshold must be in (0, 1]")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    gpumd = shutil.which(args.gpumd)
    if gpumd is None:
        gpumd_path = Path(args.gpumd).expanduser().resolve()
        if not gpumd_path.is_file():
            raise ValueError(f"GPUMD executable {args.gpumd} was not found")
        gpumd = str(gpumd_path)
    candidate_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    device = candidate_env.split(",", 1)[0].strip() if candidate_env else None

    rows = []
    for candidate in args.replicas:
        if candidate <= 0:
            raise ValueError("All replica candidates must be positive")
        print(f"Running benchmark with {candidate} replicas ...", flush=True)
        row = run_one(candidate, template, gpumd, args.steps, args.no_output, device)
        rows.append(row)
        throughput = row.get("run_replica_steps_per_s")
        throughput_text = "unknown" if throughput is None else f"{throughput:.6g}"
        print(
            f"  replicas={row['replicas']} "
            f"throughput={throughput_text} "
            f"return_code={row['return_code']} "
            f"memory_safe={row['memory_safe_at_85_percent']}",
            flush=True,
        )

    try:
        recommended, peak = recommend_candidate(rows, args.threshold)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    print(
        f"peak replicas={peak['replicas']} "
        f"throughput={peak['run_replica_steps_per_s']:.6g}"
    )
    print(
        f"recommended replicas={recommended['replicas']} "
        f"throughput={recommended['run_replica_steps_per_s']:.6g} "
        f"(threshold={args.threshold})"
    )

    fieldnames = [
        "replicas",
        "run_replica_steps_per_s",
        "return_code",
        "memory_safe_at_85_percent",
        "peak_memory_MiB",
        "total_memory_MiB",
        "elapsed_s",
    ]
    with Path(args.output).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
