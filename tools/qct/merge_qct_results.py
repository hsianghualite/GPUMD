#!/usr/bin/env python3
"""Merge per-replica QCT summary files and report channel statistics."""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", help="Per-replica qct_summary.csv files")
    parser.add_argument("--output", default="qct_ensemble.csv", help="Merged CSV output")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    fieldnames = None
    for filename in args.summaries:
        with Path(filename).open(encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames is None:
                raise ValueError(f"{filename} has no CSV header")
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"CSV columns in {filename} do not match earlier files")
            rows.extend(reader)
    if not rows:
        raise ValueError("No QCT summary rows were found")

    replica_ids = [row["replica"] for row in rows]
    if len(set(replica_ids)) != len(replica_ids):
        raise ValueError("Replica identifiers are not unique")
    rows.sort(key=lambda row: row["replica"])
    with Path(args.output).open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    statuses = Counter(row["status"] for row in rows)
    valid_rows = [row for row in rows if row["status"] == "completed"]
    channels = Counter(row["channel"] for row in valid_rows)
    print(f"Replicas: {len(rows)}")
    for status, count in sorted(statuses.items()):
        print(f"Status {status}: {count}")
    for channel, count in sorted(channels.items()):
        fraction = count / len(valid_rows) if valid_rows else float("nan")
        print(f"Channel {channel}: {count} ({fraction:.6g})")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
