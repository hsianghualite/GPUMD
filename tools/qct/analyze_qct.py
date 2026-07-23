#!/usr/bin/env python3
"""Analyze one GPUMD quasi-classical trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

LINEAR_ALGEBRA_ERROR = RuntimeError if np is None else np.linalg.LinAlgError


AMU_A2_FS2_TO_EV = 103.6426965268
HBAR_EV_FS = 0.6582119569

# Single-bond covalent radii in Angstrom. Less common elements can be supplied
# through covalent_radii in the analysis JSON file.
COVALENT_RADII = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Kr": 1.16,
    "Rb": 2.20,
    "Sr": 1.95,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Xe": 1.40,
    "Cs": 2.44,
    "Ba": 2.15,
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.90,
    "Yb": 1.87,
    "Lu": 1.87,
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Tl": 1.45,
    "Pb": 1.46,
    "Bi": 1.48,
}

ATOMIC_MASSES = {
    "H": 1.008,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403,
    "Si": 28.085,
    "P": 30.973762,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.90447,
}


@dataclass
class Frame:
    symbols: list
    positions: np.ndarray
    masses: np.ndarray
    velocities: np.ndarray | None
    lattice: np.ndarray | None
    pbc: np.ndarray
    time_fs: float | None
    metadata: dict = field(default_factory=dict)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, help="GPUMD extxyz trajectory")
    parser.add_argument("--thermo", help="GPUMD thermo.out; defaults beside trajectory")
    parser.add_argument(
        "--initial",
        help="Sampled initial phase-point extxyz; defaults to qct_initial.xyz",
    )
    parser.add_argument(
        "--bond-reference",
        help="Optimized reactant extxyz used to define the expected initial bond graph",
    )
    parser.add_argument("--config", help="QCT analysis JSON configuration")
    parser.add_argument("--replica", help="Replica identifier; defaults to trajectory directory")
    parser.add_argument("--output", help="Summary CSV output")
    parser.add_argument("--modes-output", help="Final product-mode CSV output")
    parser.add_argument("--append", action="store_true", help="Append one row to summary CSV")
    parser.add_argument("--time-step", type=float, help="MD time step in fs")
    parser.add_argument("--dump-interval", type=int, help="Trajectory output interval in MD steps")
    parser.add_argument("--bond-scale", type=float, help="Covalent-radius bond cutoff scale")
    parser.add_argument("--persistence", type=int, help="Required consecutive frames per state")
    parser.add_argument(
        "--energy-drift-tolerance",
        type=float,
        help="Maximum absolute total-energy drift in eV",
    )
    parser.add_argument(
        "--stationary",
        help="Stationary-point extxyz; defaults to qct_stationary.xyz beside trajectory",
    )
    parser.add_argument(
        "--reaction-eigenvector",
        help="GPUMD QCT eigenvector.out; defaults to qct_eigenvector.out beside trajectory",
    )
    parser.add_argument("--reaction-mode", type=int, help="Reaction-mode index; auto-detected if omitted")
    parser.add_argument(
        "--reaction-deadband",
        type=float,
        help="Reaction-coordinate deadband in sqrt(amu)*Angstrom",
    )
    parser.add_argument(
        "--reaction-hysteresis",
        type=float,
        help="Additional reaction-coordinate hysteresis in sqrt(amu)*Angstrom",
    )
    parser.add_argument("--reaction-output", help="Reaction-coordinate CSV output")
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

        symbols = []
        positions = np.empty((num_atoms, 3), dtype=float)
        masses = np.empty(num_atoms, dtype=float)
        velocities = np.empty((num_atoms, 3), dtype=float) if "vel" in layout else None
        if "vel" in layout and layout["vel"][1] != 3:
            raise ValueError(f"Velocity property in {path} must have width 3")
        mass_property = "mass" if "mass" in layout else "masses" if "masses" in layout else None
        if mass_property and layout[mass_property][1] != 1:
            raise ValueError(f"Mass property in {path} must have width 1")

        for atom_index in range(num_atoms):
            fields = lines[cursor + 2 + atom_index].split()
            if len(fields) != num_columns:
                raise ValueError(
                    f"Wrong column count at line {cursor + 3 + atom_index} of {path}"
                )
            species_offset = layout["species"][0]
            position_offset = layout["pos"][0]
            symbol = fields[species_offset]
            symbols.append(symbol)
            positions[atom_index] = [float(x) for x in fields[position_offset : position_offset + 3]]
            if mass_property:
                masses[atom_index] = float(fields[layout[mass_property][0]])
            elif symbol in ATOMIC_MASSES:
                masses[atom_index] = ATOMIC_MASSES[symbol]
            else:
                raise ValueError(f"Mass is missing for element {symbol} in {path}")
            if velocities is not None:
                velocity_offset = layout["vel"][0]
                velocities[atom_index] = [
                    float(x) for x in fields[velocity_offset : velocity_offset + 3]
                ]

        lattice = None
        if "lattice" in attributes:
            lattice_values = [float(x) for x in attributes["lattice"].split()]
            if len(lattice_values) != 9:
                raise ValueError(f"Lattice in {path} must contain 9 values")
            lattice = np.asarray(lattice_values).reshape(3, 3)
        pbc_tokens = attributes.get("pbc", "F F F").split()
        if len(pbc_tokens) != 3:
            raise ValueError(f"pbc in {path} must contain 3 values")
        pbc = np.asarray([token.upper().startswith("T") for token in pbc_tokens], dtype=bool)
        time_fs = float(attributes["time"]) if "time" in attributes else None
        frames.append(Frame(symbols, positions, masses, velocities, lattice, pbc, time_fs, attributes))
        cursor += num_atoms + 2
    if not frames:
        raise ValueError(f"No extxyz frames found in {path}")
    return frames


def minimum_image(delta, frame):
    if frame.lattice is None or not np.any(frame.pbc):
        return delta
    fractional = np.linalg.solve(frame.lattice.T, delta)
    fractional[frame.pbc] -= np.rint(fractional[frame.pbc])
    return fractional @ frame.lattice


def pair_key(symbol_a, symbol_b):
    return "-".join(sorted((symbol_a, symbol_b)))


def configured_bond_cutoffs(config):
    cutoffs = {}
    for key, value in config.get("bond_cutoffs_A", {}).items():
        symbols = key.split("-")
        cutoff = float(value)
        if len(symbols) != 2 or not all(symbols) or cutoff <= 0.0:
            raise ValueError(f"Invalid bond cutoff {key!r}")
        cutoffs[pair_key(*symbols)] = cutoff
    return cutoffs


def find_bonds(frame, bond_scale, radii, bond_cutoffs):
    bonds = set()
    for atom_i in range(len(frame.symbols) - 1):
        symbol_i = frame.symbols[atom_i]
        if symbol_i not in radii:
            raise ValueError(f"No covalent radius is available for {symbol_i}")
        for atom_j in range(atom_i + 1, len(frame.symbols)):
            symbol_j = frame.symbols[atom_j]
            if symbol_j not in radii:
                raise ValueError(f"No covalent radius is available for {symbol_j}")
            key = pair_key(symbol_i, symbol_j)
            cutoff = bond_cutoffs.get(key, bond_scale * (radii[symbol_i] + radii[symbol_j]))
            distance = np.linalg.norm(
                minimum_image(frame.positions[atom_j] - frame.positions[atom_i], frame)
            )
            if distance < cutoff:
                bonds.add((atom_i, atom_j))
    return frozenset(bonds)


def connected_components(num_atoms, bonds):
    neighbors = [[] for _ in range(num_atoms)]
    for atom_i, atom_j in bonds:
        neighbors[atom_i].append(atom_j)
        neighbors[atom_j].append(atom_i)
    components = []
    unseen = set(range(num_atoms))
    while unseen:
        first = min(unseen)
        stack = [first]
        unseen.remove(first)
        component = []
        while stack:
            atom = stack.pop()
            component.append(atom)
            for neighbor in neighbors[atom]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return sorted(components, key=lambda component: component[0])


def molecular_formula(symbols):
    counts = Counter(symbols)
    order = []
    if "C" in counts:
        order.append("C")
    if "H" in counts:
        order.append("H")
    order.extend(sorted(symbol for symbol in counts if symbol not in {"C", "H"}))
    return "".join(symbol + (str(counts[symbol]) if counts[symbol] != 1 else "") for symbol in order)


def fragment_formulas(symbols, components):
    return sorted(molecular_formula([symbols[index] for index in component]) for component in components)


def normalize_bonds(entries, index_base, num_atoms, keyword):
    normalized = set()
    for entry in entries:
        if len(entry) != 2:
            raise ValueError(f"Each {keyword} entry must contain two atom indices")
        atom_i, atom_j = (int(entry[0]) - index_base, int(entry[1]) - index_base)
        if atom_i == atom_j or min(atom_i, atom_j) < 0 or max(atom_i, atom_j) >= num_atoms:
            raise ValueError(f"Invalid atom index in {keyword}")
        normalized.add(tuple(sorted((atom_i, atom_j))))
    return normalized


def match_channel(channel, initial_bonds, final_bonds, formulas, index_base, num_atoms):
    if not any(key in channel for key in ("formed_bonds", "broken_bonds", "fragments")):
        raise ValueError(
            f"Channel {channel.get('name', '<unnamed>')} has no bond or fragment criteria"
        )
    formed = set(final_bonds) - set(initial_bonds)
    broken = set(initial_bonds) - set(final_bonds)
    expected_formed = normalize_bonds(
        channel.get("formed_bonds", []), index_base, num_atoms, "formed_bonds"
    )
    expected_broken = normalize_bonds(
        channel.get("broken_bonds", []), index_base, num_atoms, "broken_bonds"
    )
    if not expected_formed.issubset(formed) or not expected_broken.issubset(broken):
        return False
    if channel.get("exact_bond_changes", False):
        if formed != expected_formed or broken != expected_broken:
            return False
    if "fragments" in channel and sorted(channel["fragments"]) != sorted(formulas):
        return False
    return True


def classify_channel(config, initial_bonds, final_bonds, symbols, components):
    formulas = fragment_formulas(symbols, components)
    if final_bonds == initial_bonds:
        return "unreacted", formulas
    index_base = int(config.get("atom_index_base", 0))
    for channel in config.get("channels", []):
        if "name" not in channel:
            raise ValueError("Every configured channel must have a name")
        if match_channel(
            channel,
            initial_bonds,
            final_bonds,
            formulas,
            index_base,
            len(symbols),
        ):
            return channel["name"], formulas
    prefix = "dissociated" if len(components) > 1 else "product"
    return f"{prefix}:{'+'.join(formulas)}", formulas


def first_persistent_final_state(signatures, final_signature, persistence):
    run_length = 0
    for frame_index, signature in enumerate(signatures):
        if signature == final_signature:
            run_length += 1
            if run_length == persistence:
                return frame_index - persistence + 1
        else:
            run_length = 0
    return None


def frame_time(frame, frame_index, time_step, dump_interval):
    if frame.time_fs is not None:
        return frame.time_fs
    if time_step is None or dump_interval is None:
        return math.nan
    return (frame_index + 1) * time_step * dump_interval


def read_thermo(path, replica=None):
    if path is None or not Path(path).is_file():
        return None
    first_line = Path(path).read_text(encoding="utf-8").splitlines()[0].strip().lower()
    if first_line.startswith("replica,"):
        rows = []
        with Path(path).open(encoding="utf-8", newline="") as input_file:
            for row in csv.DictReader(input_file):
                if replica is None or row.get("replica") == str(replica):
                    rows.append(row)
        if not rows:
            raise ValueError(f"No thermo rows found for replica {replica!r} in {path}")
        thermo = np.asarray(
            [
                [
                    float(row["temperature_K"]),
                    float(row["kinetic_energy_eV"]),
                    float(row["potential_energy_eV"]),
                ]
                for row in rows
            ],
            dtype=float,
        )
    else:
        thermo = np.loadtxt(path, ndmin=2)
    if thermo.shape[1] < 3:
        raise ValueError(f"{path} must contain at least T, K, and U columns")
    if not np.all(np.isfinite(thermo[:, :3])):
        raise ValueError(f"{path} contains non-finite thermodynamic data")
    return thermo


def read_batch_thermo(path):
    """Read a replica-tagged QCT thermo CSV once and group it by replica."""
    groups = {}
    with Path(path).open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or not reader.fieldnames or reader.fieldnames[0] != "replica":
            return None
        for row in reader:
            groups.setdefault(str(row["replica"]), []).append(
                [
                    float(row["temperature_K"]),
                    float(row["kinetic_energy_eV"]),
                    float(row["potential_energy_eV"]),
                ]
            )
    result = {}
    for replica, values in groups.items():
        thermo = np.asarray(values, dtype=float)
        if thermo.ndim != 2 or thermo.shape[1] < 3 or not np.all(np.isfinite(thermo[:, :3])):
            raise ValueError(f"{path} contains invalid thermodynamic data for replica {replica}")
        result[replica] = thermo
    return result


def inertia_and_rotation_energy(positions, velocities, masses):
    total_mass = np.sum(masses)
    center = np.sum(masses[:, None] * positions, axis=0) / total_mass
    center_velocity = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    relative_position = positions - center
    relative_velocity = velocities - center_velocity
    inertia = np.zeros((3, 3))
    angular_momentum = np.zeros(3)
    for mass, position, velocity in zip(masses, relative_position, relative_velocity):
        inertia += mass * (np.dot(position, position) * np.eye(3) - np.outer(position, position))
        angular_momentum += mass * np.cross(position, velocity)
    angular_velocity = np.linalg.pinv(inertia, rcond=1.0e-12) @ angular_momentum
    energy = 0.5 * np.dot(angular_momentum, angular_velocity) * AMU_A2_FS2_TO_EV
    vibrational_velocity = relative_velocity - np.cross(angular_velocity, relative_position)
    return energy, center, center_velocity, angular_velocity, vibrational_velocity


def energy_partition(frame, components):
    if frame.velocities is None:
        return {
            "kinetic_from_velocity_eV": math.nan,
            "center_of_mass_energy_eV": math.nan,
            "fragment_translation_energy_eV": math.nan,
            "rotation_energy_eV": math.nan,
            "vibrational_kinetic_energy_eV": math.nan,
        }
    masses = frame.masses
    velocities = frame.velocities
    total_mass = np.sum(masses)
    global_velocity = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    kinetic = 0.5 * np.sum(masses[:, None] * velocities**2) * AMU_A2_FS2_TO_EV
    center_of_mass_energy = (
        0.5 * total_mass * np.dot(global_velocity, global_velocity) * AMU_A2_FS2_TO_EV
    )
    translation = 0.0
    rotation = 0.0
    vibration_kinetic = 0.0
    for component in components:
        indices = np.asarray(component, dtype=int)
        fragment_mass = np.sum(masses[indices])
        fragment_velocity = (
            np.sum(masses[indices, None] * velocities[indices], axis=0) / fragment_mass
        )
        translation += (
            0.5
            * fragment_mass
            * np.dot(fragment_velocity - global_velocity, fragment_velocity - global_velocity)
            * AMU_A2_FS2_TO_EV
        )
        rotation_energy, _, _, _, vibrational_velocity = inertia_and_rotation_energy(
            frame.positions[indices], velocities[indices], masses[indices]
        )
        rotation += rotation_energy
        vibration_kinetic += (
            0.5
            * np.sum(masses[indices, None] * vibrational_velocity**2)
            * AMU_A2_FS2_TO_EV
        )
    return {
        "kinetic_from_velocity_eV": kinetic,
        "center_of_mass_energy_eV": center_of_mass_energy,
        "fragment_translation_energy_eV": translation,
        "rotation_energy_eV": rotation,
        "vibrational_kinetic_energy_eV": vibration_kinetic,
    }


def load_gpumd_modes(path, num_atoms):
    dimension = num_atoms * 3
    values = np.fromfile(path, dtype=np.float32)
    expected_values = dimension + dimension * dimension
    if values.size != expected_values:
        raise ValueError(
            f"{path} contains {values.size} floats; expected {expected_values} for {num_atoms} atoms"
        )
    omega2 = values[:dimension].astype(float)
    frequencies = np.copysign(np.sqrt(np.abs(omega2)), omega2)
    eigenvectors = values[dimension:].reshape(dimension, dimension).astype(float)
    norms = np.linalg.norm(eigenvectors, axis=1)
    if np.any(~np.isfinite(values)) or np.any(np.abs(norms - 1.0) > 1.0e-3):
        raise ValueError(f"Invalid frequencies or eigenvectors in {path}")
    largest_components = np.argmax(np.abs(eigenvectors), axis=1)
    signs = eigenvectors[np.arange(dimension), largest_components]
    eigenvectors[signs < 0.0] *= -1.0
    return frequencies, eigenvectors


def mass_weighted_alignment(current, reference, masses):
    current_center = np.sum(masses[:, None] * current, axis=0) / np.sum(masses)
    reference_center = np.sum(masses[:, None] * reference, axis=0) / np.sum(masses)
    current_centered = current - current_center
    reference_centered = reference - reference_center
    covariance = current_centered.T @ (masses[:, None] * reference_centered)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return current_centered @ rotation, reference_centered, rotation


def qct_initial_metadata(path):
    metadata = {}
    path = Path(path)
    if not path.is_file():
        return metadata
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("# "):
            continue
        fields = line[2:].split(None, 1)
        if len(fields) == 2:
            metadata[fields[0]] = fields[1]
    return metadata


def resolve_analysis_path(value, default, config_directory, trajectory_directory):
    if value is None:
        path = trajectory_directory / default
    else:
        path = Path(value)
        if not path.is_absolute():
            path = config_directory / path
    return path.resolve()


def reaction_coordinate_series(
    frames,
    initial_frame,
    stationary_frame,
    eigenvectors,
    reaction_mode,
    time_step,
    dump_interval,
    deadband,
    hysteresis,
):
    if stationary_frame.symbols != initial_frame.symbols:
        raise ValueError("Stationary structure symbols/order do not match the trajectory")
    if not np.allclose(stationary_frame.masses, initial_frame.masses, rtol=1.0e-6, atol=1.0e-8):
        raise ValueError("Stationary structure masses do not match the trajectory")
    if reaction_mode < 0 or reaction_mode >= eigenvectors.shape[0]:
        raise ValueError("Reaction-mode index is outside the eigenvector file")

    num_atoms = len(initial_frame.symbols)
    mode = eigenvectors[reaction_mode].reshape(3, num_atoms).T
    sqrt_mass = np.sqrt(initial_frame.masses)[:, None]
    rows = []
    all_frames = [initial_frame] + list(frames)
    for frame_index, frame in enumerate(all_frames):
        if frame.symbols != initial_frame.symbols:
            raise ValueError("Stationary reaction-coordinate frame symbols/order do not match")
        aligned, reference_centered, rotation = mass_weighted_alignment(
            frame.positions, stationary_frame.positions, initial_frame.masses
        )
        displacement = aligned - reference_centered
        Q = float(np.sum(sqrt_mass * mode * displacement))
        P = math.nan
        if frame.velocities is not None:
            center_velocity = np.sum(
                initial_frame.masses[:, None] * frame.velocities, axis=0
            ) / np.sum(initial_frame.masses)
            velocity_aligned = (frame.velocities - center_velocity) @ rotation
            P = float(np.sum(sqrt_mass * mode * velocity_aligned))
        if frame_index == 0:
            time_fs = 0.0
        else:
            time_fs = frame_time(frame, frame_index - 1, time_step, dump_interval)
        rows.append(
            {
                "frame": frame_index,
                "time_fs": time_fs,
                "Q_rxn_sqrt_amu_A": Q,
                "P_rxn_sqrt_amu_A_per_fs": P,
            }
        )

    enter_threshold = deadband
    exit_threshold = deadband + hysteresis
    side = 0
    initial_side = 0
    crossings = 0
    for row in rows:
        coordinate = row["Q_rxn_sqrt_amu_A"]
        if side == 0:
            if coordinate > enter_threshold:
                side = 1
            elif coordinate < -enter_threshold:
                side = -1
            if side != 0 and initial_side == 0:
                initial_side = side
        elif side == 1 and coordinate < -exit_threshold:
            side = -1
            crossings += 1
        elif side == -1 and coordinate > exit_threshold:
            side = 1
            crossings += 1
        row["side"] = side

    return rows, {
        "reaction_coordinate_available": True,
        "reaction_mode_index": reaction_mode,
        "reaction_coordinate_initial_Q": rows[0]["Q_rxn_sqrt_amu_A"],
        "reaction_coordinate_initial_P": rows[0]["P_rxn_sqrt_amu_A_per_fs"],
        "reaction_coordinate_final_Q": rows[-1]["Q_rxn_sqrt_amu_A"],
        "reaction_coordinate_final_P": rows[-1]["P_rxn_sqrt_amu_A_per_fs"],
        "reaction_coordinate_min_Q": min(row["Q_rxn_sqrt_amu_A"] for row in rows),
        "reaction_coordinate_max_Q": max(row["Q_rxn_sqrt_amu_A"] for row in rows),
        "reaction_initial_side": initial_side,
        "reaction_final_side": side,
        "reaction_crossing_count": crossings,
        "reaction_recrossing_count": crossings,
    }, rows


def write_reaction_coordinate(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def project_product_modes(frame, mode_config, config_directory, index_base):
    name = mode_config.get("name", "product")
    indices = np.asarray([int(index) - index_base for index in mode_config["atoms"]], dtype=int)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= len(frame.symbols)):
        raise ValueError(f"Invalid atom list for product mode set {name}")
    reference_path = config_directory / mode_config["reference"]
    eigenvector_path = config_directory / mode_config["eigenvector"]
    reference = read_extxyz(reference_path)[0]
    if len(reference.symbols) != len(indices):
        raise ValueError(f"Reference atom count does not match product mode set {name}")
    symbols = [frame.symbols[index] for index in indices]
    if symbols != reference.symbols:
        raise ValueError(f"Atom symbols/order do not match product mode set {name}")
    masses = frame.masses[indices]
    if not np.allclose(masses, reference.masses, rtol=1.0e-6, atol=1.0e-8):
        raise ValueError(f"Atom masses do not match product mode set {name}")
    if frame.velocities is None:
        raise ValueError("Product mode projection requires velocities in the trajectory")

    current_aligned, reference_centered, rotation = mass_weighted_alignment(
        frame.positions[indices], reference.positions, masses
    )
    displacement = current_aligned - reference_centered
    mass_weighted_rmsd = math.sqrt(
        np.sum(masses[:, None] * displacement**2) / np.sum(masses)
    )
    max_rmsd = float(mode_config.get("max_rmsd_A", 0.3))
    if max_rmsd <= 0.0:
        raise ValueError(f"max_rmsd_A must be positive for product mode set {name}")
    diagnostic = {
        "product": name,
        "valid": mass_weighted_rmsd <= max_rmsd,
        "mass_weighted_rmsd_A": mass_weighted_rmsd,
        "max_rmsd_A": max_rmsd,
    }
    if not diagnostic["valid"]:
        return [], diagnostic
    _, _, _, _, vibrational_velocity = inertia_and_rotation_energy(
        frame.positions[indices], frame.velocities[indices], masses
    )
    velocity_aligned = vibrational_velocity @ rotation
    frequencies, eigenvectors = load_gpumd_modes(eigenvector_path, len(indices))
    exclude_lowest = int(mode_config.get("exclude_lowest", 6))
    min_frequency = float(mode_config.get("min_frequency", 1.0e-3))
    gaussian_width = float(mode_config.get("gaussian_width", 0.05))
    if exclude_lowest < 0 or exclude_lowest >= len(frequencies) or gaussian_width <= 0.0:
        raise ValueError(f"Invalid mode selection settings for {name}")

    rows = []
    sqrt_mass = np.sqrt(masses)[:, None]
    for mode_index in range(exclude_lowest, len(frequencies)):
        frequency = frequencies[mode_index]
        if frequency < -min_frequency:
            raise ValueError(f"Product mode set {name} has imaginary mode {mode_index}")
        if frequency <= min_frequency:
            continue
        eigenvector = eigenvectors[mode_index].reshape(3, len(indices)).T
        Q = np.sum(sqrt_mass * eigenvector * displacement)
        P = np.sum(sqrt_mass * eigenvector * velocity_aligned)
        omega_fs = 2.0 * math.pi * frequency * 1.0e-3
        energy = 0.5 * (P * P + (omega_fs * Q) ** 2) * AMU_A2_FS2_TO_EV
        quantum_continuous = energy / (HBAR_EV_FS * omega_fs) - 0.5
        quantum_nearest = max(0, int(math.floor(quantum_continuous + 0.5)))
        gaussian_weight = math.exp(
            -0.5 * ((quantum_continuous - quantum_nearest) / gaussian_width) ** 2
        )
        zpe = 0.5 * HBAR_EV_FS * omega_fs
        rows.append(
            {
                "product": name,
                "mode": mode_index,
                "frequency_THz": frequency,
                "Q_sqrt_amu_A": Q,
                "P_sqrt_amu_A_per_fs": P,
                "energy_eV": energy,
                "quantum_continuous": quantum_continuous,
                "quantum_standard_bin": quantum_nearest,
                "gaussian_bin_weight": gaussian_weight,
                "below_zpe": energy < zpe,
            }
        )
    return rows, diagnostic


def resolve_setting(args_value, config, key, default):
    return args_value if args_value is not None else config.get(key, default)


def analyze(
    args,
    frames_override=None,
    initial_frame_override=None,
    bond_reference_frame_override=None,
    thermo_override=None,
    replica_override=None,
    reaction_output_override=None,
):
    trajectory_path = Path(args.trajectory).resolve()
    config_path = Path(args.config).resolve() if args.config else None
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
    config_directory = config_path.parent if config_path else Path.cwd()
    frames = frames_override if frames_override is not None else read_extxyz(trajectory_path)
    num_atoms = len(frames[0].symbols)
    for frame in frames:
        if len(frame.symbols) != num_atoms or frame.symbols != frames[0].symbols:
            raise ValueError("Atom count, symbols, or order changes within the trajectory")

    initial_path = Path(args.initial).resolve() if args.initial else trajectory_path.parent / "qct_initial.xyz"
    initial_frame = (
        initial_frame_override
        if initial_frame_override is not None
        else read_extxyz(initial_path)[0] if initial_path.is_file() else frames[0]
    )
    if initial_frame.symbols != frames[0].symbols:
        raise ValueError("Initial structure symbols/order do not match trajectory")

    reaction_summary = {
        "reaction_coordinate_available": False,
        "reaction_mode_index": "",
        "reaction_coordinate_initial_Q": math.nan,
        "reaction_coordinate_initial_P": math.nan,
        "reaction_coordinate_final_Q": math.nan,
        "reaction_coordinate_final_P": math.nan,
        "reaction_coordinate_min_Q": math.nan,
        "reaction_coordinate_max_Q": math.nan,
        "reaction_initial_side": "",
        "reaction_final_side": "",
        "reaction_crossing_count": "",
        "reaction_recrossing_count": "",
    }
    reaction_rows = None
    stationary_setting = str(config.get("stationary_point", "auto"))
    if stationary_setting not in {"auto", "minimum", "saddle"}:
        raise ValueError("stationary_point should be auto, minimum, or saddle")
    stationary_value = args.stationary or config.get("stationary_structure")
    eigenvector_value = args.reaction_eigenvector or config.get("reaction_eigenvector")
    stationary_path = resolve_analysis_path(
        stationary_value, "qct_stationary.xyz", config_directory, trajectory_path.parent
    )
    eigenvector_path = resolve_analysis_path(
        eigenvector_value, "qct_eigenvector.out", config_directory, trajectory_path.parent
    )
    reaction_files_requested = (
        args.stationary is not None
        or args.reaction_eigenvector is not None
        or "stationary_structure" in config
        or "reaction_eigenvector" in config
        or stationary_setting == "saddle"
        or stationary_path.is_file()
        or eigenvector_path.is_file()
    )
    if reaction_files_requested and stationary_path.is_file() != eigenvector_path.is_file():
        raise ValueError("Stationary structure and reaction eigenvector must be provided together")

    configured_bond_reference = config.get("bond_reference")
    if args.bond_reference:
        bond_reference_path = Path(args.bond_reference).resolve()
    elif configured_bond_reference:
        bond_reference_path = config_directory / configured_bond_reference
    else:
        bond_reference_path = initial_path
    bond_reference_frame = (
        bond_reference_frame_override
        if bond_reference_frame_override is not None
        else read_extxyz(bond_reference_path)[0]
    )
    if bond_reference_frame.symbols != frames[0].symbols:
        raise ValueError("Bond-reference symbols/order do not match trajectory")

    bond_scale = float(resolve_setting(args.bond_scale, config, "bond_scale", 1.25))
    persistence = int(resolve_setting(args.persistence, config, "persistence_frames", 5))
    time_step = resolve_setting(args.time_step, config, "time_step_fs", None)
    dump_interval = resolve_setting(args.dump_interval, config, "dump_interval", None)
    energy_tolerance = float(
        resolve_setting(args.energy_drift_tolerance, config, "energy_drift_tolerance_eV", 0.05)
    )
    if bond_scale <= 0.0 or persistence <= 0 or energy_tolerance < 0.0:
        raise ValueError("bond_scale and persistence must be positive; energy tolerance cannot be negative")
    if time_step is not None and float(time_step) <= 0.0:
        raise ValueError("time_step_fs must be positive")
    if dump_interval is not None and int(dump_interval) <= 0:
        raise ValueError("dump_interval must be positive")

    if stationary_path.is_file() and eigenvector_path.is_file():
        stationary_frame = read_extxyz(stationary_path)[0]
        frequencies, eigenvectors = load_gpumd_modes(eigenvector_path, num_atoms)
        min_frequency = float(config.get("reaction_min_frequency_THz", 1.0e-3))
        if min_frequency < 0.0:
            raise ValueError("reaction_min_frequency_THz cannot be negative")
        reaction_mode = args.reaction_mode
        if reaction_mode is None and "reaction_mode_index" in config:
            reaction_mode = int(config["reaction_mode_index"])
        if reaction_mode is None:
            metadata = qct_initial_metadata(initial_path.parent / "qct_initial.out")
            if "reaction_mode" in metadata:
                reaction_mode = int(metadata["reaction_mode"])
        imaginary_modes = np.flatnonzero(frequencies < -min_frequency)
        if reaction_mode is None:
            if len(imaginary_modes) == 1:
                reaction_mode = int(imaginary_modes[0])
            elif stationary_setting == "saddle":
                raise ValueError("Saddle analysis requires exactly one imaginary reaction mode")
        if reaction_mode is not None:
            if len(imaginary_modes) != 1 or int(imaginary_modes[0]) != reaction_mode:
                raise ValueError("Reaction eigenvector must contain exactly one matching imaginary mode")
            deadband = float(
                resolve_setting(
                    args.reaction_deadband,
                    config,
                    "reaction_coordinate_deadband_sqrt_amu_A",
                    0.05,
                )
            )
            hysteresis = float(
                resolve_setting(
                    args.reaction_hysteresis,
                    config,
                    "reaction_coordinate_hysteresis_sqrt_amu_A",
                    0.02,
                )
            )
            if deadband < 0.0 or hysteresis < 0.0:
                raise ValueError("Reaction-coordinate deadband and hysteresis cannot be negative")
            _, reaction_summary, reaction_rows = reaction_coordinate_series(
                frames,
                initial_frame,
                stationary_frame,
                eigenvectors,
                reaction_mode,
                time_step,
                dump_interval,
                deadband,
                hysteresis,
            )
            reaction_output_value = (
                reaction_output_override
                or args.reaction_output
                or config.get("reaction_coordinate_output")
            )
            reaction_output_path = resolve_analysis_path(
                reaction_output_value,
                "qct_reaction_coordinate.csv",
                config_directory,
                trajectory_path.parent,
            )
            write_reaction_coordinate(reaction_output_path, reaction_rows)
        elif stationary_setting == "saddle":
            raise ValueError("Saddle analysis did not find an imaginary reaction mode")

    radii = dict(COVALENT_RADII)
    radii.update({key: float(value) for key, value in config.get("covalent_radii", {}).items()})
    if any(radius <= 0.0 for radius in radii.values()):
        raise ValueError("Covalent radii must be positive")
    bond_cutoffs = configured_bond_cutoffs(config)
    initial_bonds = find_bonds(bond_reference_frame, bond_scale, radii, bond_cutoffs)
    sampled_initial_bonds = find_bonds(initial_frame, bond_scale, radii, bond_cutoffs)
    initial_topology_match = sampled_initial_bonds == initial_bonds
    initial_topology_valid = True if reaction_summary["reaction_coordinate_available"] else initial_topology_match
    signatures = [find_bonds(frame, bond_scale, radii, bond_cutoffs) for frame in frames]
    final_signature = signatures[-1]
    stable = len(signatures) >= persistence and all(
        signature == final_signature for signature in signatures[-persistence:]
    )
    components = connected_components(num_atoms, final_signature)
    channel, formulas = classify_channel(
        config, initial_bonds, final_signature, frames[-1].symbols, components
    )
    reaction_frame = None
    if stable and final_signature != initial_bonds:
        reaction_frame = first_persistent_final_state(signatures, final_signature, persistence)
    reaction_time = (
        frame_time(frames[reaction_frame], reaction_frame, time_step, dump_interval)
        if reaction_frame is not None
        else math.nan
    )

    thermo_path = Path(args.thermo).resolve() if args.thermo else trajectory_path.parent / "thermo.out"
    thermo = thermo_override if thermo_override is not None else read_thermo(
        thermo_path, replica_override if replica_override is not None else args.replica
    )
    energy_initial = energy_final = energy_drift = energy_max_deviation = energy_range = math.nan
    if thermo is not None:
        total_energy = thermo[:, 1] + thermo[:, 2]
        energy_initial = total_energy[0]
        energy_final = total_energy[-1]
        energy_drift = energy_final - energy_initial
        energy_max_deviation = np.max(np.abs(total_energy - energy_initial))
        energy_range = np.max(total_energy) - np.min(total_energy)

    status = "completed" if stable else "unfinished"
    if thermo is not None and energy_max_deviation > energy_tolerance:
        status = "failed_energy_conservation"
    if not initial_topology_valid:
        status = "invalid_initial_topology"
    partition = energy_partition(frames[-1], components)
    reference_energies = config.get("product_reference_energies_eV", {})
    vibrational_energy = math.nan
    if thermo is not None and channel in reference_energies:
        vibrational_energy = (
            energy_final
            - partition["center_of_mass_energy_eV"]
            - partition["fragment_translation_energy_eV"]
            - partition["rotation_energy_eV"]
            - float(reference_energies[channel])
        )

    formed = sorted(set(final_signature) - set(initial_bonds))
    broken = sorted(set(initial_bonds) - set(final_signature))
    initial_formed = sorted(set(sampled_initial_bonds) - set(initial_bonds))
    initial_broken = sorted(set(initial_bonds) - set(sampled_initial_bonds))
    summary = {
        "replica": (
            str(replica_override)
            if replica_override is not None
            else args.replica or trajectory_path.parent.name
        ),
        "status": status,
        "channel": channel,
        "reaction_time_fs": reaction_time,
        "num_frames": len(frames),
        "num_fragments": len(components),
        "fragments": "+".join(formulas),
        "initial_topology_valid": initial_topology_valid,
        "initial_topology_checked": not reaction_summary["reaction_coordinate_available"],
        "initial_topology_mismatch": not initial_topology_match,
        "initial_formed_bonds": json.dumps(initial_formed, separators=(",", ":")),
        "initial_broken_bonds": json.dumps(initial_broken, separators=(",", ":")),
        "formed_bonds": json.dumps(formed, separators=(",", ":")),
        "broken_bonds": json.dumps(broken, separators=(",", ":")),
        "energy_initial_eV": energy_initial,
        "energy_final_eV": energy_final,
        "energy_drift_eV": energy_drift,
        "energy_max_deviation_eV": energy_max_deviation,
        "energy_range_eV": energy_range,
        **partition,
        "vibrational_energy_estimate_eV": vibrational_energy,
        **reaction_summary,
    }

    mode_rows = []
    mode_diagnostics = []
    index_base = int(config.get("atom_index_base", 0))
    for mode_config in config.get("product_modes", []):
        selected_channels = mode_config.get("channels")
        if selected_channels is not None and channel not in selected_channels:
            continue
        mode_indices = {
            int(index) - index_base for index in mode_config.get("atoms", [])
        }
        if not any(mode_indices.issubset(set(component)) for component in components):
            raise ValueError(
                f"Atoms for product mode set {mode_config.get('name', 'product')} "
                "do not form one final fragment"
            )
        projected_rows, diagnostic = project_product_modes(
            frames[-1], mode_config, config_directory, index_base
        )
        mode_rows.extend(projected_rows)
        mode_diagnostics.append(diagnostic)
    summary["mode_projection_requested"] = bool(mode_diagnostics)
    summary["mode_projection_valid"] = (
        all(diagnostic["valid"] for diagnostic in mode_diagnostics)
        if mode_diagnostics
        else ""
    )
    summary["max_product_mode_rmsd_A"] = (
        max(diagnostic["mass_weighted_rmsd_A"] for diagnostic in mode_diagnostics)
        if mode_diagnostics
        else math.nan
    )
    summary["invalid_mode_products"] = "+".join(
        diagnostic["product"] for diagnostic in mode_diagnostics if not diagnostic["valid"]
    )
    return summary, mode_rows


def analyze_batch(args):
    trajectory_path = Path(args.trajectory).resolve()
    frames = read_extxyz(trajectory_path)
    replica_values = [frame.metadata.get("replica") for frame in frames]
    if all(value is None for value in replica_values):
        return None
    if any(value is None for value in replica_values):
        raise ValueError("Every frame in a batch trajectory must contain Replica metadata")

    groups = {}
    for frame, replica in zip(frames, replica_values):
        groups.setdefault(str(replica), []).append(frame)
    if args.replica is not None:
        requested = str(args.replica)
        if requested not in groups:
            raise ValueError(f"Replica {requested} was not found in {trajectory_path}")
        groups = {requested: groups[requested]}

    initial_path = Path(args.initial).resolve() if args.initial else trajectory_path.parent / "qct_initial.xyz"
    initial_by_replica = {}
    if initial_path.is_file():
        for frame in read_extxyz(initial_path):
            replica = frame.metadata.get("replica")
            if replica is not None:
                initial_by_replica[str(replica)] = frame

    config_path = Path(args.config).resolve() if args.config else None
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
    configured_reaction_output = args.reaction_output or config.get("reaction_coordinate_output")
    if configured_reaction_output:
        reaction_base = Path(configured_reaction_output)
        if not reaction_base.is_absolute():
            reaction_base = (config_path.parent if config_path else Path.cwd()) / reaction_base
    else:
        reaction_base = trajectory_path.parent / "qct_reaction_coordinate.csv"

    thermo_path = Path(args.thermo).resolve() if args.thermo else trajectory_path.parent / "qct_thermo.csv"
    if not thermo_path.is_file() and args.thermo is None:
        fallback = trajectory_path.parent / "thermo.out"
        thermo_path = fallback if fallback.is_file() else thermo_path

    thermo_by_replica = read_batch_thermo(thermo_path) if thermo_path.is_file() else None
    summaries = []
    mode_rows = []
    for replica, replica_frames in groups.items():
        initial_frame = initial_by_replica.get(replica, replica_frames[0])
        if thermo_by_replica is not None:
            if replica not in thermo_by_replica:
                raise ValueError(f"No thermo rows found for replica {replica!r} in {thermo_path}")
            thermo = thermo_by_replica[replica]
        else:
            thermo = read_thermo(thermo_path, replica) if thermo_path.is_file() else None
        reaction_output = reaction_base.with_name(
            f"{reaction_base.stem}_replica_{replica}{reaction_base.suffix}"
        )
        summary, replica_modes = analyze(
            args,
            frames_override=replica_frames,
            initial_frame_override=initial_frame,
            bond_reference_frame_override=(
                initial_frame
                if args.bond_reference is None and "bond_reference" not in config
                else None
            ),
            thermo_override=thermo,
            replica_override=replica,
            reaction_output_override=str(reaction_output),
        )
        summaries.append(summary)
        for row in replica_modes:
            row["replica"] = replica
        mode_rows.extend(replica_modes)
    return summaries, mode_rows


def write_csv(path, rows, append=False):
    if not rows:
        return
    path = Path(path)
    mode = "a" if append else "w"
    write_header = not append or not path.exists() or path.stat().st_size == 0
    with path.open(mode, encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if np is None:
        raise RuntimeError("analyze_qct.py requires NumPy")
    trajectory_path = Path(args.trajectory).resolve()
    summary_path = Path(args.output) if args.output else trajectory_path.parent / "qct_summary.csv"
    modes_path = (
        Path(args.modes_output)
        if args.modes_output
        else trajectory_path.parent / "qct_modes_final.csv"
    )
    batch_result = analyze_batch(args)
    if batch_result is None:
        summary, mode_rows = analyze(args)
        summaries = [summary]
    else:
        summaries, mode_rows = batch_result
    write_csv(summary_path, summaries, append=args.append)
    if mode_rows:
        write_csv(modes_path, mode_rows)
    elif modes_path.exists():
        modes_path.write_text("", encoding="utf-8")
    for summary in summaries:
        print(f"Replica: {summary['replica']}")
        print(f"Status: {summary['status']}")
        print(f"Channel: {summary['channel']}")
        print(f"Fragments: {summary['fragments']}")
        if math.isfinite(summary["energy_drift_eV"]):
            print(
                "Energy drift/range: "
                f"{summary['energy_drift_eV']:.6g} / {summary['energy_range_eV']:.6g} eV"
            )
        if summary["reaction_coordinate_available"]:
            print(
                "Reaction coordinate crossings/recrossings: "
                f"{summary['reaction_crossing_count']} / "
                f"{summary['reaction_recrossing_count']}"
            )
        if summary["mode_projection_requested"] and not summary["mode_projection_valid"]:
            print(
                "Product mode projection rejected: "
                f"{summary['invalid_mode_products']} exceeds its harmonic RMSD limit"
            )
    if mode_rows:
        print(f"Projected product modes: {len(mode_rows)}")


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        LINEAR_ALGEBRA_ERROR,
        json.JSONDecodeError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
