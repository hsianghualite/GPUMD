#!/usr/bin/env python3
"""Report and validate per-replica QCT energy conservation."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def validate(path, expected_replicas, expected_rows, drift_tolerance, range_tolerance):
    energies = defaultdict(list)
    with Path(path).open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            energy = float(row["total_energy_eV"])
            if not math.isfinite(energy):
                raise ValueError("non-finite total energy")
            energies[int(row["replica"])].append(energy)
    if len(energies) != expected_replicas:
        raise ValueError(f"expected {expected_replicas} replicas, found {len(energies)}")
    if any(len(values) != expected_rows for values in energies.values()):
        raise ValueError("per-replica thermo row count mismatch")
    absolute_drifts = [abs(values[-1] - values[0]) for values in energies.values()]
    ranges = [max(values) - min(values) for values in energies.values()]
    maximum_drift = max(absolute_drifts)
    maximum_range = max(ranges)
    print(f"replicas={len(energies)} rows_per_replica={expected_rows}")
    print(f"max_abs_final_drift_eV={maximum_drift:.6e}")
    print(f"p95_abs_final_drift_eV={percentile(absolute_drifts, 0.95):.6e}")
    print(f"max_energy_range_eV={maximum_range:.6e}")
    print(f"p95_energy_range_eV={percentile(ranges, 0.95):.6e}")
    if maximum_drift > drift_tolerance:
        raise ValueError(f"maximum final drift exceeds {drift_tolerance} eV")
    if maximum_range > range_tolerance:
        raise ValueError(f"maximum energy range exceeds {range_tolerance} eV")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("thermo")
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--rows-per-replica", type=int, required=True)
    parser.add_argument("--drift-tolerance", type=float, default=0.01)
    parser.add_argument("--range-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    validate(
        args.thermo,
        args.replicas,
        args.rows_per_replica,
        args.drift_tolerance,
        args.range_tolerance,
    )


if __name__ == "__main__":
    main()
