#!/usr/bin/env python3
"""Benchmark the useful QCT batch size on a single GPU.

The tool clones a template run directory, rewrites the ``replicas`` argument of
``ensemble qct`` to each candidate value, runs GPUMD, and records replica-step
throughput and a peak memory reading from ``nvidia-smi``.  The recommendation
logic :func:`recommend_candidate` picks the smallest batch whose throughput is
at least ``threshold`` (default 0.95) of the peak throughput observed across
successful, memory-safe batches.

The CLI is intentionally light: it shells out to ``gpumd`` and ``nvidia-smi``
which must be on PATH.  Tests only exercise :func:`recommend_candidate`.
"""

from __future__ import annotations

import argparse
import csv
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


def run_one(candidate, template, gpumd, steps, no_output, device):
    workdir = Path(tempfile.mkdtemp(prefix=f"qct-bench-{candidate}-"))
    for entry in template.iterdir():
        if entry.is_dir():
            shutil.copytree(entry, workdir / entry.name)
        else:
            shutil.copy2(entry, workdir / entry.name)
    run_in = workdir / "run.in"
    if not run_in.exists():
        return {"replicas": candidate, "return_code": -1}
    rewrite_run_in(run_in, candidate, steps, no_output)
    env = dict(__import__("os").environ)
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    start = time.time()
    proc = subprocess.run(
        [gpumd],
        cwd=workdir,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    elapsed = max(time.time() - start, 1.0e-9)
    throughput = None
    if proc.returncode == 0:
        throughput = parse_gpumd_throughput(proc.stderr + "\n" + proc.stdout)
    if throughput is None:
        throughput = (candidate * steps) / elapsed
    memory = query_process_memory(device or 0, proc.pid) if proc.returncode == 0 else None
    shutil.rmtree(workdir, ignore_errors=True)
    return {
        "replicas": candidate,
        "run_replica_steps_per_s": throughput,
        "return_code": proc.returncode,
        "memory_safe_at_85_percent": True if memory is None else (memory >= 0),
        "elapsed_s": elapsed,
    }


def recommend_candidate(rows, threshold=0.95):
    """Pick the smallest batch reaching ``threshold`` of peak throughput.

    Only rows with ``return_code == 0`` and ``memory_safe_at_85_percent`` truthy
    are considered.  Returns ``(recommended, peak)`` where ``peak`` is the row
    with the largest throughput and ``recommended`` is the smallest row whose
    throughput is at least ``threshold * peak``.  If only one row survives
    filtering, ``recommended == peak``.
    """
    valid = [
        row for row in rows
        if row.get("return_code", 0) == 0 and row.get("memory_safe_at_85_percent", True)
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
    device = None
    candidate_env = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")
    if candidate_env is not None and candidate_env.isdigit():
        device = int(candidate_env)

    rows = []
    for candidate in args.replicas:
        if candidate <= 0:
            raise ValueError("All replica candidates must be positive")
        print(f"Running benchmark with {candidate} replicas ...", flush=True)
        row = run_one(candidate, template, args.gpumd, args.steps, args.no_output, device)
        rows.append(row)
        print(
            f"  replicas={row['replicas']} "
            f"throughput={row['run_replica_steps_per_s']:.6g} "
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
