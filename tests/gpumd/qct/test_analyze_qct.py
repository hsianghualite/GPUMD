import csv
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYZER = REPO_ROOT / "tools/qct/analyze_qct.py"
MERGER = REPO_ROOT / "tools/qct/merge_qct_results.py"
COMPARATOR = REPO_ROOT / "tools/qct/compare_batch.py"


def write_h2_frame(output, distance, velocity, time):
    output.write("2\n")
    output.write(
        f'Time={time} pbc="F F F" Lattice="20 0 0 0 20 0 0 0 20" '
        "Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3\n"
    )
    output.write(f"H {-0.5 * distance} 0 0 1.008 {-velocity} 0 0\n")
    output.write(f"H {0.5 * distance} 0 0 1.008 {velocity} 0 0\n")


def write_h2_batch_frame(output, replica, distance, velocity, time, step):
    output.write("2\n")
    output.write(
        f'Time={time} Replica={replica} Step={step} Seed={100 + replica} '
        'pbc="F F F" Lattice="20 0 0 0 20 0 0 0 20" '
        "Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3\n"
    )
    output.write(f"H {-0.5 * distance} 0 0 1.008 {-velocity} 0 0\n")
    output.write(f"H {0.5 * distance} 0 0 1.008 {velocity} 0 0\n")


def write_qct_modes(path):
    dimension = 6
    omega2 = [0.0] * 5 + [-10000.0]
    modes = []
    for mode in range(dimension):
        vector = [0.0] * dimension
        if mode < 5:
            vector[mode] = 1.0
        else:
            vector[0] = -(2.0**-0.5)
            vector[3] = 2.0**-0.5
        modes.extend(vector)
    path.write_bytes(struct.pack(f"{dimension + dimension * dimension}f", *(omega2 + modes)))


def run_analyzer(tmp_path, config):
    config_path = tmp_path / "qct_analysis.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--trajectory",
            str(tmp_path / "trajectory.xyz"),
            "--initial",
            str(tmp_path / "initial.xyz"),
            "--thermo",
            str(tmp_path / "thermo.out"),
            "--config",
            str(config_path),
            "--replica",
            "test-1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with (tmp_path / "qct_summary.csv").open(encoding="utf-8", newline="") as input_file:
        summary = next(csv.DictReader(input_file))
    return result, summary


def test_persistent_dissociation_and_energy_partition(tmp_path):
    with (tmp_path / "initial.xyz").open("w", encoding="utf-8") as output:
        write_h2_frame(output, 0.74, 0.0, 0.0)
    with (tmp_path / "trajectory.xyz").open("w", encoding="utf-8") as output:
        write_h2_frame(output, 0.74, 0.01, 1.0)
        write_h2_frame(output, 2.00, 0.01, 2.0)
        write_h2_frame(output, 2.10, 0.01, 3.0)
        write_h2_frame(output, 2.20, 0.01, 4.0)
    np.savetxt(
        tmp_path / "thermo.out",
        np.asarray(
            [
                [300, 0.010, -1.0],
                [300, 0.011, -1.0],
                [300, 0.012, -1.0],
                [300, 0.013, -1.0],
            ]
        ),
    )
    config = {
        "persistence_frames": 3,
        "energy_drift_tolerance_eV": 0.01,
        "channels": [
            {
                "name": "H2_dissociation",
                "broken_bonds": [[0, 1]],
                "fragments": ["H", "H"],
                "exact_bond_changes": True,
            }
        ],
    }
    result, summary = run_analyzer(tmp_path, config)
    assert "Channel: H2_dissociation" in result.stdout
    assert summary["status"] == "completed"
    assert summary["channel"] == "H2_dissociation"
    assert float(summary["reaction_time_fs"]) == 2.0
    assert summary["fragments"] == "H+H"
    assert abs(float(summary["energy_drift_eV"]) - 0.003) < 1.0e-12
    assert abs(float(summary["fragment_translation_energy_eV"]) - 0.010446) < 1.0e-5
    assert abs(float(summary["rotation_energy_eV"])) < 1.0e-12
    assert abs(float(summary["vibrational_kinetic_energy_eV"])) < 1.0e-12


def test_product_normal_mode_projection(tmp_path):
    with (tmp_path / "initial.xyz").open("w", encoding="utf-8") as output:
        write_h2_frame(output, 0.74, 0.0, 0.0)
    with (tmp_path / "trajectory.xyz").open("w", encoding="utf-8") as output:
        for time in range(1, 4):
            write_h2_frame(output, 0.74, 0.001, float(time))
    np.savetxt(tmp_path / "thermo.out", np.asarray([[300, 0.1, -1.0]] * 3))

    dimension = 6
    omega2 = [0.0] * 5 + [10000.0]
    modes = []
    for mode in range(dimension):
        vector = [0.0] * dimension
        if mode == 5:
            vector[0] = 2.0**-0.5
            vector[1] = -(2.0**-0.5)
        else:
            vector[mode] = 1.0
        modes.extend(vector)
    (tmp_path / "eigenvector.out").write_bytes(
        struct.pack(f"{dimension + dimension * dimension}f", *(omega2 + modes))
    )
    config = {
        "persistence_frames": 3,
        "product_modes": [
            {
                "name": "H2",
                "channels": ["unreacted"],
                "atoms": [0, 1],
                "reference": "initial.xyz",
                "eigenvector": "eigenvector.out",
                "exclude_lowest": 5,
            }
        ],
    }
    result, summary = run_analyzer(tmp_path, config)
    assert summary["channel"] == "unreacted"
    assert "Projected product modes: 1" in result.stdout
    with (tmp_path / "qct_modes_final.csv").open(encoding="utf-8", newline="") as input_file:
        mode = next(csv.DictReader(input_file))
    assert mode["product"] == "H2"
    assert int(mode["mode"]) == 5
    assert float(mode["energy_eV"]) > 0.0

    with (tmp_path / "trajectory.xyz").open("w", encoding="utf-8") as output:
        for time in range(1, 4):
            write_h2_frame(output, 0.76, 0.001, float(time))
    config["product_modes"][0]["max_rmsd_A"] = 0.001
    rejected_result, rejected_summary = run_analyzer(tmp_path, config)
    assert rejected_summary["mode_projection_valid"] == "False"
    assert "Product mode projection rejected" in rejected_result.stdout
    assert (tmp_path / "qct_modes_final.csv").stat().st_size == 0


def test_saddle_reaction_coordinate_and_recrossing(tmp_path):
    with (tmp_path / "initial.xyz").open("w", encoding="utf-8") as output:
        write_h2_frame(output, 0.74, 0.01, 0.0)
    with (tmp_path / "qct_stationary.xyz").open("w", encoding="utf-8") as output:
        write_h2_frame(output, 0.74, 0.0, 0.0)
    with (tmp_path / "qct_initial.out").open("w", encoding="utf-8") as output:
        output.write("# QCT_INITIAL v1\n# reaction_mode 5\n")
    write_qct_modes(tmp_path / "qct_eigenvector.out")
    with (tmp_path / "trajectory.xyz").open("w", encoding="utf-8") as output:
        write_h2_frame(output, 0.82, 0.01, 1.0)
        write_h2_frame(output, 1.00, 0.01, 2.0)
        write_h2_frame(output, 0.60, -0.01, 3.0)
        write_h2_frame(output, 0.55, -0.01, 4.0)
    np.savetxt(tmp_path / "thermo.out", np.asarray([[300, 0.1, -1.0]] * 4))

    _, summary = run_analyzer(
        tmp_path,
        {
            "stationary_point": "saddle",
            "reaction_mode_index": 5,
            "reaction_coordinate_deadband_sqrt_amu_A": 0.01,
            "reaction_coordinate_hysteresis_sqrt_amu_A": 0.0,
            "bond_cutoffs_A": {"H-H": 2.0},
            "persistence_frames": 2,
        },
    )
    assert summary["status"] == "completed"
    assert summary["initial_topology_valid"] == "True"
    assert summary["initial_topology_checked"] == "False"
    assert summary["reaction_coordinate_available"] == "True"
    assert int(summary["reaction_mode_index"]) == 5
    assert int(summary["reaction_crossing_count"]) == 1
    assert int(summary["reaction_recrossing_count"]) == 1
    assert int(summary["reaction_initial_side"]) == 1
    assert int(summary["reaction_final_side"]) == -1
    reaction_rows = list(csv.DictReader((tmp_path / "qct_reaction_coordinate.csv").open()))
    assert len(reaction_rows) == 5


def test_merge_replica_summaries(tmp_path):
    fieldnames = ["replica", "status", "channel"]
    rows = [
        {"replica": "2", "status": "completed", "channel": "unreacted"},
        {"replica": "1", "status": "completed", "channel": "reaction"},
    ]
    inputs = []
    for index, row in enumerate(rows):
        path = tmp_path / f"summary-{index}.csv"
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        inputs.append(path)
    merged = tmp_path / "merged.csv"
    result = subprocess.run(
        [sys.executable, str(MERGER), *(str(path) for path in inputs), "--output", str(merged)],
        check=True,
        capture_output=True,
        text=True,
    )
    with merged.open(encoding="utf-8", newline="") as input_file:
        merged_rows = list(csv.DictReader(input_file))
    assert [row["replica"] for row in merged_rows] == ["1", "2"]
    assert "Channel reaction: 1 (0.5)" in result.stdout
    assert "Channel unreacted: 1 (0.5)" in result.stdout


def test_batch_trajectory_analysis(tmp_path):
    with (tmp_path / "qct_initial.xyz").open("w", encoding="utf-8") as output:
        for replica in range(2):
            write_h2_batch_frame(output, replica, 0.74, 0.0, 0.0, 0)
    with (tmp_path / "qct_trajectory.xyz").open("w", encoding="utf-8") as output:
        for step in range(1, 4):
            for replica in range(2):
                write_h2_batch_frame(output, replica, 0.74, 0.01, float(step), step)
    for replica in range(2):
        with (tmp_path / f"replica{replica}.xyz").open("w", encoding="utf-8") as output:
            for step in range(1, 4):
                write_h2_frame(output, 0.74, 0.01, float(step))
    with (tmp_path / "qct_thermo.csv").open("w", encoding="utf-8", newline="") as output:
        output.write(
            "replica,step,time_fs,temperature_K,kinetic_energy_eV,"
            "potential_energy_eV,total_energy_eV\n"
        )
        for step in range(1, 4):
            for replica in range(2):
                output.write(f"{replica},{step},{step},300,0.1,-1.0,-0.9\n")
    config_path = tmp_path / "qct_analysis.json"
    config_path.write_text(json.dumps({"persistence_frames": 2}), encoding="utf-8")
    summary_path = tmp_path / "qct_summary.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--trajectory",
            str(tmp_path / "qct_trajectory.xyz"),
            "--initial",
            str(tmp_path / "qct_initial.xyz"),
            "--thermo",
            str(tmp_path / "qct_thermo.csv"),
            "--config",
            str(config_path),
            "--output",
            str(summary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with summary_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert [row["replica"] for row in rows] == ["0", "1"]
    assert all(row["status"] == "completed" for row in rows)
    assert "Replica: 0" in result.stdout
    assert "Replica: 1" in result.stdout
    compare = subprocess.run(
        [
            sys.executable,
            str(COMPARATOR),
            "--batch",
            str(tmp_path / "qct_trajectory.xyz"),
            "--reference",
            f"0={tmp_path / 'replica0.xyz'}",
            "--reference",
            f"1={tmp_path / 'replica1.xyz'}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "batch_vs_single_max_abs_error=0.000000e+00" in compare.stdout


def test_invalid_sampled_initial_topology(tmp_path):
    with (tmp_path / "optimized.xyz").open("w", encoding="utf-8") as output:
        write_h2_frame(output, 0.74, 0.0, 0.0)
    with (tmp_path / "initial.xyz").open("w", encoding="utf-8") as output:
        write_h2_frame(output, 2.0, 0.0, 0.0)
    with (tmp_path / "trajectory.xyz").open("w", encoding="utf-8") as output:
        for time in range(1, 4):
            write_h2_frame(output, 2.0, 0.01, float(time))
    np.savetxt(tmp_path / "thermo.out", np.asarray([[300, 0.1, -1.0]] * 3))
    _, summary = run_analyzer(
        tmp_path,
        {
            "persistence_frames": 3,
            "bond_reference": "optimized.xyz",
        },
    )
    assert summary["status"] == "invalid_initial_topology"
    assert summary["initial_topology_valid"] == "False"
    assert summary["initial_broken_bonds"] == "[[0,1]]"
