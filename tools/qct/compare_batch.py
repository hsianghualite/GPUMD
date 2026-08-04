#!/usr/bin/env python3
"""Compare a native QCT batch trajectory against independent per-replica runs.

The batch trajectory is an extended XYZ file written by ``dump_qct`` where each
frame carries a ``Replica=`` metadata field.  For each replica, the tool collects
every batch frame with that replica id and compares it against the matching
frame of an independently propagated single-replica trajectory supplied through
``--reference <replica>=<file>``.  The reported metric is the maximum absolute
difference over all positions, velocities, and masses across all compared
frames and atoms.
"""

from __future__ import annotations

import argparse
import math
import shlex
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="Batch qct_trajectory.xyz file")
    parser.add_argument(
        "--reference",
        action="append",
        required=True,
        help="Single-replica reference file as <replica>=<path>; may be given multiple times",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Fail with non-zero exit when the max abs error exceeds this tolerance",
    )
    return parser.parse_args()


def header_attributes(line):
    attributes = {}
    for token in shlex.split(line):
        if "=" in token:
            key, value = token.split("=", 1)
            attributes[key.lower()] = value
    return attributes


def property_layout(schema, path):
    fields = schema.split(":")
    if len(fields) % 3:
        raise ValueError(f"Malformed Properties schema in {path}")
    layout = {}
    offset = 0
    for index in range(0, len(fields), 3):
        name = fields[index].lower()
        width = int(fields[index + 2])
        if width <= 0 or name in layout:
            raise ValueError(f"Malformed property {name!r} in {path}")
        layout[name] = (offset, width)
        offset += width
    return layout, offset


def read_extxyz(path):
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    frames = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        try:
            num_atoms = int(lines[cursor].strip())
        except ValueError as error:
            raise ValueError(f"Expected atom count at line {cursor + 1} of {path}") from error
        if num_atoms <= 0 or cursor + num_atoms + 1 >= len(lines):
            raise ValueError(f"Incomplete extxyz frame at line {cursor + 1} of {path}")
        attributes = header_attributes(lines[cursor + 1])
        if "properties" not in attributes:
            raise ValueError(f"Missing Properties in frame at line {cursor + 2} of {path}")
        layout, num_columns = property_layout(attributes["properties"], path)
        if "species" not in layout or "pos" not in layout:
            raise ValueError(f"Properties in {path} must include species and pos")
        if layout["species"][1] != 1 or layout["pos"][1] != 3:
            raise ValueError(f"Unsupported species or pos width in {path}")
        if "vel" in layout and layout["vel"][1] != 3:
            raise ValueError(f"Velocity property in {path} must have width 3")
        mass_property = "mass" if "mass" in layout else "masses" if "masses" in layout else None
        if mass_property and layout[mass_property][1] != 1:
            raise ValueError(f"Mass property in {path} must have width 1")

        symbols = []
        positions = np.empty((num_atoms, 3), dtype=float)
        masses = (
            np.empty(num_atoms, dtype=float) if mass_property else None
        )
        velocities = np.empty((num_atoms, 3), dtype=float) if "vel" in layout else None
        for atom_index in range(num_atoms):
            fields = lines[cursor + 2 + atom_index].split()
            if len(fields) != num_columns:
                raise ValueError(
                    f"Wrong column count at line {cursor + 3 + atom_index} of {path}"
                )
            species_offset = layout["species"][0]
            position_offset = layout["pos"][0]
            symbols.append(fields[species_offset])
            positions[atom_index] = [
                float(x) for x in fields[position_offset : position_offset + 3]
            ]
            if mass_property:
                masses[atom_index] = float(fields[layout[mass_property][0]])
            if velocities is not None:
                vel_offset = layout["vel"][0]
                velocities[atom_index] = [
                    float(x) for x in fields[vel_offset : vel_offset + 3]
                ]
        frames.append(
            {
                "num_atoms": num_atoms,
                "symbols": symbols,
                "positions": positions,
                "masses": masses,
                "velocities": velocities,
                "metadata": attributes,
            }
        )
        cursor += num_atoms + 2
    return frames


def parse_reference_spec(spec):
    if "=" not in spec:
        raise ValueError(f"--reference must be <replica>=<path>, got {spec!r}")
    key, value = spec.split("=", 1)
    try:
        replica = int(key)
    except ValueError as error:
        raise ValueError(f"--reference replica id must be integer, got {key!r}") from error
    return replica, Path(value)


def group_batch_by_replica(frames):
    grouped = {}
    for frame in frames:
        replica_value = frame["metadata"].get("replica")
        if replica_value is None:
            raise ValueError("Batch frame is missing Replica metadata")
        try:
            replica = int(replica_value)
        except ValueError as error:
            raise ValueError(f"Replica metadata {replica_value!r} is not integer") from error
        grouped.setdefault(replica, []).append(frame)
    return grouped


def compare_frames(batch_frames, reference_frames):
    if len(batch_frames) != len(reference_frames):
        raise ValueError(
            f"Frame count mismatch: batch has {len(batch_frames)}, reference has "
            f"{len(reference_frames)}"
        )
    max_abs_error = 0.0
    for batch_frame, reference_frame in zip(batch_frames, reference_frames):
        if batch_frame["num_atoms"] != reference_frame["num_atoms"]:
            raise ValueError("Atom count mismatch between batch and reference frame")
        if batch_frame["symbols"] != reference_frame["symbols"]:
            raise ValueError("Atom symbols/order mismatch between batch and reference frame")
        max_abs_error = max(
            max_abs_error,
            float(np.max(np.abs(batch_frame["positions"] - reference_frame["positions"]))),
        )
        if batch_frame["velocities"] is not None and reference_frame["velocities"] is not None:
            max_abs_error = max(
                max_abs_error,
                float(
                    np.max(
                        np.abs(
                            batch_frame["velocities"] - reference_frame["velocities"]
                        )
                    )
                ),
            )
        if (
            batch_frame["masses"] is not None
            and reference_frame["masses"] is not None
        ):
            max_abs_error = max(
                max_abs_error,
                float(
                    np.max(np.abs(batch_frame["masses"] - reference_frame["masses"]))
                ),
            )
    return max_abs_error


def main():
    args = parse_args()
    references = {}
    for spec in args.reference:
        replica, path = parse_reference_spec(spec)
        if replica in references:
            raise ValueError(f"Duplicate reference for replica {replica}")
        references[replica] = path

    batch_frames = read_extxyz(args.batch)
    if not batch_frames:
        raise ValueError(f"No frames found in batch file {args.batch}")

    grouped = group_batch_by_replica(batch_frames)
    max_abs_error = 0.0
    compared = 0
    for replica, reference_path in references.items():
        if replica not in grouped:
            raise ValueError(
                f"Replica {replica} appears in references but not in batch trajectory"
            )
        reference_frames = read_extxyz(reference_path)
        if not reference_frames:
            raise ValueError(f"No frames in reference {reference_path}")
        error = compare_frames(grouped[replica], reference_frames)
        max_abs_error = max(max_abs_error, error)
        compared += 1
        print(f"replica={replica} frames={len(grouped[replica])} max_abs_error={error:.6e}")

    missing = set(grouped) - set(references)
    if missing:
        print(
            f"Warning: batch replicas {sorted(missing)} have no reference file",
            file=sys.stderr,
        )

    print(f"compared_replicas={compared}")
    print(f"batch_vs_single_max_abs_error={max_abs_error:.6e}")
    if not math.isfinite(max_abs_error):
        raise ValueError("Non-finite max abs error encountered")
    if max_abs_error > args.tolerance:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
