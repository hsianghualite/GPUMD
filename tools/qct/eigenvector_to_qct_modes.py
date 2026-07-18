#!/usr/bin/env python3
"""Convert GPUMD Gamma-point eigenvectors to a native QCT modes file."""

import argparse
import math
import shlex
import struct
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Optimized extxyz structure")
    parser.add_argument("--eigenvector", required=True, help="GPUMD eigenvector.out")
    parser.add_argument("--output", required=True, help="Output qct_modes.in")
    parser.add_argument("--model-output", help="Optional QCT model.xyz output")
    parser.add_argument(
        "--exclude-lowest",
        type=int,
        default=6,
        help="Number of lowest translational/rotational modes to disable",
    )
    parser.add_argument(
        "--min-frequency",
        type=float,
        default=1.0e-3,
        help="Minimum positive angular frequency in THz for an active mode",
    )
    return parser.parse_args()


def parse_extxyz(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} is not a valid extxyz file")

    num_atoms = int(lines[0].strip())
    header = lines[1].strip()
    if len(lines) < num_atoms + 2:
        raise ValueError(f"{path} has fewer than {num_atoms} atom lines")

    attributes = {}
    for token in shlex.split(header):
        if "=" in token:
            key, value = token.split("=", 1)
            attributes[key] = value

    if "Properties" not in attributes:
        raise ValueError(f"{path} is missing the Properties attribute")

    property_tokens = attributes["Properties"].split(":")
    if len(property_tokens) % 3 != 0:
        raise ValueError(f"{path} has an invalid Properties attribute")

    offsets = {}
    column = 0
    for index in range(0, len(property_tokens), 3):
        name = property_tokens[index]
        count = int(property_tokens[index + 2])
        offsets[name] = (column, count)
        column += count

    if "species" not in offsets or "pos" not in offsets or "mass" not in offsets:
        raise ValueError(f"{path} must contain species, pos, and mass properties")
    if offsets["species"][1] != 1 or offsets["pos"][1] != 3 or offsets["mass"][1] != 1:
        raise ValueError(f"{path} has unsupported species, pos, or mass dimensions")

    symbols = []
    masses = []
    positions = []
    for atom_line in lines[2 : num_atoms + 2]:
        fields = atom_line.split()
        if len(fields) < column:
            raise ValueError(f"Malformed atom line in {path}: {atom_line}")
        species_offset = offsets["species"][0]
        position_offset = offsets["pos"][0]
        mass_offset = offsets["mass"][0]
        symbols.append(fields[species_offset])
        positions.append(tuple(float(value) for value in fields[position_offset : position_offset + 3]))
        masses.append(float(fields[mass_offset]))

    return {
        "num_atoms": num_atoms,
        "pbc": attributes.get("pbc", "F F F"),
        "lattice": attributes.get("Lattice", "0 0 0 0 0 0 0 0 0"),
        "symbols": symbols,
        "masses": masses,
        "positions": positions,
    }


def read_eigenvectors(path, num_atoms):
    dimension = num_atoms * 3
    raw = Path(path).read_bytes()
    expected_values = dimension + dimension * dimension
    if len(raw) != expected_values * 4:
        raise ValueError(
            f"{path} has {len(raw)} bytes; expected {expected_values * 4} for {num_atoms} atoms"
        )

    values = struct.unpack(f"<{expected_values}f", raw)
    omega2 = values[:dimension]
    vector_values = values[dimension:]
    modes = []
    for mode_index in range(dimension):
        start = mode_index * dimension
        components = list(vector_values[start : start + dimension])
        norm = math.sqrt(sum(component * component for component in components))
        if norm == 0.0:
            raise ValueError(f"Mode {mode_index} has a zero eigenvector")
        modes.append([component / norm for component in components])
    return omega2, modes


def signed_frequency(omega2):
    if omega2 >= 0.0:
        return math.sqrt(omega2)
    return -math.sqrt(-omega2)


def write_model(path, model):
    with Path(path).open("w", encoding="utf-8") as output:
        output.write(f"{model['num_atoms']}\n")
        output.write(
            f"pbc=\"{model['pbc']}\" Lattice=\"{model['lattice']}\" "
            "Properties=species:S:1:pos:R:3:mass:R:1\n"
        )
        for symbol, position, mass in zip(
            model["symbols"], model["positions"], model["masses"]
        ):
            output.write(
                f"{symbol} {position[0]:.17g} {position[1]:.17g} {position[2]:.17g} "
                f"{mass:.17g}\n"
            )


def write_qct_modes(path, model, omega2, modes, exclude_lowest, min_frequency):
    num_atoms = model["num_atoms"]
    num_modes = num_atoms * 3
    imaginary_modes = []
    active_modes = []
    frequencies = [signed_frequency(value) for value in omega2]

    for mode_index, frequency in enumerate(frequencies):
        if mode_index >= exclude_lowest and frequency < -min_frequency:
            imaginary_modes.append((mode_index, frequency))
        if mode_index >= exclude_lowest and frequency > min_frequency:
            active_modes.append(mode_index)

    if imaginary_modes:
        description = ", ".join(
            f"{index}:{frequency:.6g}" for index, frequency in imaginary_modes
        )
        raise ValueError(f"Imaginary vibrational modes remain after optimization: {description} THz")
    if not active_modes:
        raise ValueError("No active vibrational modes were found")

    with Path(path).open("w", encoding="utf-8") as output:
        output.write("QCT_MODES v1\n")
        output.write(f"num_atoms {num_atoms}\n")
        output.write(f"num_modes {num_modes}\n")
        output.write("frequency_unit THz\n")
        output.write("coordinate_unit Angstrom\n")
        output.write("mass_unit amu\n")
        output.write("eigenvector_type mass_weighted\n")
        output.write("normalization sum_e2_1\n")
        output.write("reference_position yes\n\n")

        output.write("atoms\n")
        output.write("# index symbol mass x0 y0 z0\n")
        for atom_index, (symbol, mass, position) in enumerate(
            zip(model["symbols"], model["masses"], model["positions"])
        ):
            output.write(
                f"{atom_index} {symbol} {mass:.17g} {position[0]:.17g} "
                f"{position[1]:.17g} {position[2]:.17g}\n"
            )
        output.write("end_atoms\n\n")

        output.write("modes\n")
        active_set = set(active_modes)
        for mode_index, (frequency, components) in enumerate(zip(frequencies, modes)):
            active = "yes" if mode_index in active_set else "no"
            output.write(
                f"mode {mode_index} frequency {frequency:.17g} active {active}\n"
            )
            for atom_index in range(num_atoms):
                ex = components[atom_index]
                ey = components[num_atoms + atom_index]
                ez = components[2 * num_atoms + atom_index]
                output.write(f"{atom_index} {ex:.17g} {ey:.17g} {ez:.17g}\n")
            output.write("end_mode\n\n")
        output.write("end_modes\n")

    return frequencies, active_modes


def main():
    args = parse_args()
    model = parse_extxyz(args.model)
    dimension = model["num_atoms"] * 3
    if args.exclude_lowest < 0 or args.exclude_lowest >= dimension:
        raise ValueError("--exclude-lowest should be in [0, 3N)")
    if args.min_frequency < 0.0:
        raise ValueError("--min-frequency should be non-negative")

    omega2, modes = read_eigenvectors(args.eigenvector, model["num_atoms"])
    frequencies, active_modes = write_qct_modes(
        args.output,
        model,
        omega2,
        modes,
        args.exclude_lowest,
        args.min_frequency,
    )
    if args.model_output:
        write_model(args.model_output, model)

    print(f"Converted {dimension} modes for {model['num_atoms']} atoms")
    print(f"Active vibrational modes: {len(active_modes)}")
    print(
        "Angular-frequency range: "
        f"{min(frequencies[index] for index in active_modes):.6g} to "
        f"{max(frequencies[index] for index in active_modes):.6g} THz"
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, struct.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
