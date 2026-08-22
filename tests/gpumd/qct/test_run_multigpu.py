import csv
import math
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "run_multigpu", REPO_ROOT / "tools/qct/run_multigpu.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _write_summary(path: Path, log_weight: float, seed: int = 1) -> None:
    path.write_text(
        "replica,seed,wigner_weight,log_wigner_weight\n"
        f"0,{seed},1,{log_weight}\n",
        encoding="utf-8",
    )


def _write_hac(path: Path, value: float, time_shift: float = 0.0) -> None:
    rows = np.zeros((2, 11), dtype=float)
    rows[:, 0] = [0.0, 1.0 + time_shift]
    rows[:, 1:] = value
    np.savetxt(path, rows)


def test_rewrite_run_in_uses_shared_eigenvector_and_absolute_potential(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    potential = tmp_path / "potential.txt"
    potential.write_text("potential\n", encoding="utf-8")
    eigenvector = tmp_path / "modes.bin"
    eigenvector.write_bytes(b"modes")
    source = template / "run.in"
    source.write_text(
        "potential ../potential.txt\n"
        "ensemble lsc_ivr 300 seed 7 replicas 4 hessian_displacement 0.001\n",
        encoding="utf-8",
    )

    destination = tmp_path / "run.out"
    RUNNER.rewrite_run_in(source, destination, 1, 12345, eigenvector.resolve(), 3)
    text = destination.read_text(encoding="utf-8")

    assert f"potential {potential.resolve()}" in text
    assert "seed 12345" in text
    assert "replicas 1" in text
    assert "hessian_displacement" not in text
    assert f"eigenvector {eigenvector.resolve()} exclude_lowest 3" in text
    references = RUNNER._referenced_input_hashes(destination)
    assert set(references) == {str(potential.resolve()), str(eigenvector.resolve())}
    assert references[str(potential.resolve())] == RUNNER._sha256_file(potential)
    assert references[str(eigenvector.resolve())] == RUNNER._sha256_file(eigenvector)


def test_cuda_process_environment_normalizes_wsl_driver_mapping(monkeypatch):
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "invalid-container-device")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/custom/lib")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "stale-token")

    environment = RUNNER._cuda_process_environment("0")

    assert "NVIDIA_VISIBLE_DEVICES" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    if Path("/usr/lib/wsl/lib").is_dir():
        assert environment["LD_LIBRARY_PATH"].split(":")[:2] == [
            "/usr/lib/wsl/lib", "/custom/lib"
        ]

    unrestricted = RUNNER._cuda_process_environment()
    assert "CUDA_VISIBLE_DEVICES" not in unrestricted


def test_copy_template_skips_generated_hessian_and_hac_outputs(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "run.in").write_text("run 1\n", encoding="utf-8")
    (template / "model.xyz").write_text("input\n", encoding="utf-8")
    (template / "qct_hessian.out").write_text("old\n", encoding="utf-8")
    (template / "qct_eigenvector.out").write_bytes(b"old")
    (template / "hac.out").write_text("old\n", encoding="utf-8")
    (template / "hac_replica.out").write_text("old\n", encoding="utf-8")
    (template / "hac_reweighting.csv").write_text("old\n", encoding="utf-8")

    destination = tmp_path / "replica"
    RUNNER._copy_template(template, destination)

    assert (destination / "model.xyz").is_file()
    assert not (destination / "run.in").exists()
    assert not (destination / "qct_hessian.out").exists()
    assert not (destination / "qct_eigenvector.out").exists()
    assert not (destination / "hac.out").exists()
    assert not (destination / "hac_replica.out").exists()
    assert not (destination / "hac_reweighting.csv").exists()


def test_merge_hac_uses_log_weights_and_writes_diagnostics(tmp_path):
    hac_files = [tmp_path / "hac0.out", tmp_path / "hac1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_hac(hac_files[0], 2.0)
    _write_hac(hac_files[1], 4.0)
    _write_summary(summary_files[0], 1000.0)
    _write_summary(summary_files[1], 999.0)

    diagnostics = RUNNER.merge_hac(
        hac_files,
        summary_files,
        tmp_path / "merged.out",
        [0, 1],
        tmp_path / "uncertainty.csv",
        tmp_path / "manifest.json",
    )

    merged = np.loadtxt(tmp_path / "merged.out")
    expected = (np.e * 2.0 + 4.0) / (np.e + 1.0)
    assert np.allclose(merged[:, 1:], expected)
    assert np.array_equal(merged[:, 0], [0.0, 1.0])
    assert diagnostics["effective_replicas"] == pytest.approx(
        (np.e + 1.0) ** 2 / (np.e**2 + 1.0)
    )
    assert (tmp_path / "uncertainty.csv").is_file()
    assert json.loads((tmp_path / "manifest.json").read_text())["acceptance_errors"] == []


def test_merge_hac_rejects_time_grid_mismatch(tmp_path):
    hac_files = [tmp_path / "hac0.out", tmp_path / "hac1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_hac(hac_files[0], 2.0)
    _write_hac(hac_files[1], 4.0, time_shift=0.1)
    _write_summary(summary_files[0], 0.0)
    _write_summary(summary_files[1], 0.0)

    with pytest.raises(ValueError, match="time-grid mismatch"):
        RUNNER.merge_hac(hac_files, summary_files, tmp_path / "merged.out", [0, 1])


def test_merge_hac_preserves_diagnostics_when_acceptance_fails(tmp_path):
    hac_files = [tmp_path / "hac0.out", tmp_path / "hac1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_hac(hac_files[0], 2.0)
    _write_hac(hac_files[1], 4.0)
    _write_summary(summary_files[0], 0.0)
    _write_summary(summary_files[1], -10.0)
    manifest = tmp_path / "manifest.json"

    with pytest.raises(ValueError, match="acceptance failed"):
        RUNNER.merge_hac(
            hac_files,
            summary_files,
            tmp_path / "merged.out",
            [0, 1],
            tmp_path / "uncertainty.csv",
            manifest,
            min_effective_replicas=2.0,
            max_normalized_weight=0.6,
        )

    diagnostics = json.loads(manifest.read_text())
    assert diagnostics["acceptance_errors"]
    assert (tmp_path / "merged.out").is_file()
    assert (tmp_path / "uncertainty.csv").is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_merge_hac_rejects_wrong_schema(tmp_path):
    hac_files = [tmp_path / "hac0.out", tmp_path / "hac1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_hac(hac_files[0], 2.0)
    np.savetxt(hac_files[1], np.zeros((2, 10)))
    _write_summary(summary_files[0], 0.0)
    _write_summary(summary_files[1], 0.0)

    with pytest.raises(ValueError, match="11-column"):
        RUNNER.merge_hac(hac_files, summary_files, tmp_path / "merged.out", [0, 1])


def test_validate_task_rejects_stale_artifacts_and_wrong_seed(tmp_path):
    _write_summary(tmp_path / "qct_initial_summary.csv", 0.0, seed=7)
    _write_hac(tmp_path / "hac.out", 1.0)
    (tmp_path / "gpumd.log").write_text("Finished running GPUMD.\n", encoding="utf-8")

    started_at = time.time() + 2.0
    with pytest.raises(ValueError, match="predates this task"):
        RUNNER._validate_task_artifacts(tmp_path, "hac", 7, started_at)

    started_at = time.time() - 1.0
    with pytest.raises(ValueError, match="Summary seed mismatch"):
        RUNNER._validate_task_artifacts(tmp_path, "hac", 8, started_at)

    _write_summary(tmp_path / "qct_initial_summary.csv", 0.0, seed=8)
    (tmp_path / "gpumd.log").write_text("incomplete\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completion marker is missing"):
        RUNNER._validate_task_artifacts(tmp_path, "hac", 8, started_at)


def test_failed_process_records_failed_manifest(tmp_path):
    run_dir = tmp_path / "replica"
    run_dir.mkdir()
    executable = tmp_path / "fail.py"
    executable.write_text("#!/usr/bin/env python3\nraise SystemExit(9)\n", encoding="utf-8")
    executable.chmod(0o755)

    return_code = RUNNER._run_replica_task(
        0, run_dir, str(executable), "0", [], 12345, "hac", {"input_hash": "test"}
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert return_code == 9
    assert manifest["status"] == "failed"
    assert manifest["return_code"] == 9


def test_launch_error_records_failed_manifest(tmp_path):
    run_dir = tmp_path / "replica"
    run_dir.mkdir()
    not_executable = tmp_path / "gpumd"
    not_executable.write_text("not executable\n", encoding="utf-8")

    return_code = RUNNER._run_replica_task(
        0, run_dir, str(not_executable), "0", [], 12345, "hac", {}
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert return_code == 127
    assert manifest["status"] == "failed"
    assert manifest["validation_error"].startswith("Failed to launch GPUMD:")


def test_device_queues_never_overlap_processes_on_the_same_gpu(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "run.in").write_text(
        "ensemble lsc_ivr 300 seed 1 replicas 1 anharmonic_reweighting no\n"
        "compute_hac 1 2 1\nrun 1\n",
        encoding="utf-8",
    )
    fake_gpumd = tmp_path / "serialized_gpumd.py"
    fake_gpumd.write_text(
        "#!/usr/bin/env python3\n"
        "import os, re, time\n"
        "from pathlib import Path\n"
        "text = Path('run.in').read_text()\n"
        "seed = int(re.search(r'\\bseed\\s+(\\d+)', text).group(1))\n"
        "lock = Path(os.environ['LOCK_ROOT']) / ('gpu_' + os.environ['CUDA_VISIBLE_DEVICES'])\n"
        "try:\n"
        "    lock.mkdir()\n"
        "except FileExistsError:\n"
        "    raise SystemExit(17)\n"
        "try:\n"
        "    time.sleep(0.35 if seed == 12345 else 0.05)\n"
        "    Path('qct_initial_summary.csv').write_text(\n"
        "        'replica,seed,wigner_weight,log_wigner_weight\\n' + f'0,{seed},1,0\\n')\n"
        "    Path('hac.out').write_text('0 ' + ' '.join(['1'] * 10) + '\\n')\n"
        "    print('Finished running GPUMD.')\n"
        "finally:\n"
        "    lock.rmdir()\n",
        encoding="utf-8",
    )
    fake_gpumd.chmod(0o755)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0,1"
    environment["LOCK_ROOT"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/qct/run_multigpu.py"),
            "--template", str(template),
            "--gpumd", str(fake_gpumd),
            "--total-replicas", "3",
            "--num-gpus", "2",
            "--workflow", "hac",
            "--no-merge",
            "--output", str(tmp_path / "output"),
        ],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_hac_workflow_runs_five_independent_processes_and_resumes(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "model.xyz").write_text(
        "1\npbc=\"T T T\" Properties=species:S:1:pos:R:3:mass:R:1\nX 0 0 0 1\n",
        encoding="utf-8",
    )
    (template / "run.in").write_text(
        "ensemble lsc_ivr 300 seed 1 replicas 1 anharmonic_reweighting yes\n"
        "compute_hac 1 2 1\nrun 1\n",
        encoding="utf-8",
    )
    fake_gpumd = tmp_path / "fake_gpumd.py"
    fake_gpumd.write_text(
        "#!/usr/bin/env python3\n"
        "import os, re\n"
        "from pathlib import Path\n"
        "text = Path('run.in').read_text()\n"
        "seed = int(re.search(r'\\bseed\\s+(\\d+)', text).group(1))\n"
        "with Path(os.environ['CALL_LOG']).open('a') as log:\n"
        "    log.write(f'{seed}\\n')\n"
        "count = Path('call_count')\n"
        "count.write_text(str((int(count.read_text()) if count.exists() else 0) + 1))\n"
        "Path('qct_initial_summary.csv').write_text(\n"
        "    'replica,seed,wigner_weight,log_wigner_weight\\n'\n"
        "    + f'0,{seed},1,0\\n')\n"
        "rows = [' '.join([str(t)] + [str(float(seed))] * 10) for t in (0, 1)]\n"
        "Path('hac.out').write_text('\\n'.join(rows) + '\\n')\n"
        "print('DEVICE=' + os.environ['CUDA_VISIBLE_DEVICES'])\n"
        "print('Finished running GPUMD.')\n",
        encoding="utf-8",
    )
    fake_gpumd.chmod(0o755)
    output = tmp_path / "output"
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/qct/run_multigpu.py"),
        "--template", str(template),
        "--gpumd", str(fake_gpumd),
        "--total-replicas", "5",
        "--num-gpus", "2",
        "--base-seed", "12345",
        "--workflow", "hac",
        "--output", str(output),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "2,GPU-test-uuid"
    call_log = tmp_path / "calls.log"
    environment["CALL_LOG"] = str(call_log)

    first = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    resume = subprocess.run(command + ["--resume"], env=environment, capture_output=True, text=True)
    assert resume.returncode == 0, resume.stderr
    assert sorted(call_log.read_text(encoding="utf-8").splitlines()) == [
        str(seed) for seed in range(12345, 12350)
    ]

    with (template / "run.in").open("a", encoding="utf-8") as run_input:
        run_input.write("# invalidate the template hash\n")
    invalidated = subprocess.run(
        command + ["--resume"], env=environment, capture_output=True, text=True
    )
    assert invalidated.returncode == 0, invalidated.stderr
    assert sorted(call_log.read_text(encoding="utf-8").splitlines()) == sorted(
        [str(seed) for seed in range(12345, 12350)] * 2
    )

    manifests = []
    for replica_id in range(5):
        run_dir = output / "runs" / f"replica_{replica_id:06d}"
        manifests.append(json.loads((run_dir / "manifest.json").read_text()))
        assert (run_dir / "call_count").read_text() == "1"
    assert [manifest["seed"] for manifest in manifests] == list(range(12345, 12350))
    assert [manifest["device"] for manifest in manifests] == [
        "2", "GPU-test-uuid", "2", "GPU-test-uuid", "2"
    ]
    with (output / "qct_initial_summary.csv").open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert [int(row["replica"]) for row in rows] == list(range(5))
    diagnostics = json.loads((output / "hac_merge_manifest.json").read_text())
    assert diagnostics["replica_count"] == 5
    assert diagnostics["effective_replicas"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Tests for merge_dos (P1-3: Wigner-weighted DOS merge)
# ---------------------------------------------------------------------------

def _write_dos(path: Path, values: list[float], freq_shift: float = 0.0) -> None:
    """Write a 2-column dos.out: frequency_THz, dos."""
    rows = np.array(
        [[i * 0.1 + freq_shift, v] for i, v in enumerate(values)],
        dtype=float,
    )
    np.savetxt(path, rows)


def test_merge_dos_wigner_weighted(tmp_path):
    """DOS merge should use log-space Wigner weights for normalization."""
    dos_files = [tmp_path / "dos0.out", tmp_path / "dos1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_dos(dos_files[0], [2.0, 4.0])
    _write_dos(dos_files[1], [4.0, 8.0])
    _write_summary(summary_files[0], 0.0)       # weight = 1
    _write_summary(summary_files[1], math.log(3.0))  # weight = 3

    RUNNER.merge_dos(dos_files, summary_files, tmp_path / "dos_merged.out", [0, 1])

    merged = np.loadtxt(tmp_path / "dos_merged.out")
    # w0=1, w1=3, normalized: w0'=0.25, w1'=0.75
    # expected[0] = 0.25*2 + 0.75*4 = 3.5
    # expected[1] = 0.25*4 + 0.75*8 = 7.0
    assert merged[0, 1] == pytest.approx(3.5)
    assert merged[1, 1] == pytest.approx(7.0)
    assert merged[0, 0] == pytest.approx(0.0)
    assert merged[1, 0] == pytest.approx(0.1)


def test_merge_dos_rejects_frequency_mismatch(tmp_path):
    """DOS merge should raise on frequency grid mismatch."""
    dos_files = [tmp_path / "dos0.out", tmp_path / "dos1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_dos(dos_files[0], [2.0, 4.0])
    _write_dos(dos_files[1], [4.0, 8.0], freq_shift=0.001)
    _write_summary(summary_files[0], 0.0)
    _write_summary(summary_files[1], 0.0)

    with pytest.raises(ValueError, match="frequency grid mismatch"):
        RUNNER.merge_dos(dos_files, summary_files, tmp_path / "dos_merged.out", [0, 1])


def test_merge_dos_missing_file_raises(tmp_path):
    """DOS merge should raise if a replica file is missing."""
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_summary(summary_files[0], 0.0)
    _write_summary(summary_files[1], 0.0)
    _write_dos(tmp_path / "dos0.out", [1.0, 2.0])
    # dos1.out is intentionally absent

    with pytest.raises(ValueError, match="Missing DOS file"):
        RUNNER.merge_dos(
            [tmp_path / "dos0.out", tmp_path / "dos1.out"],
            summary_files,
            tmp_path / "dos_merged.out",
            [0, 1],
        )


# ---------------------------------------------------------------------------
# Tests for merge_dipole (P1-4: Wigner-weighted IR spectrum merge)
# ---------------------------------------------------------------------------

def _write_dipole(path: Path, values: list[tuple[float, float, float]]) -> None:
    """Write dipole.out: step dx dy dz."""
    lines = [f"{i} {v[0]} {v[1]} {v[2]}" for i, v in enumerate(values)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_merge_dipole_wigner_weighted(tmp_path):
    """Dipole merge should use Wigner weights."""
    dipole_files = [tmp_path / "dip0.out", tmp_path / "dip1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_dipole(dipole_files[0], [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    _write_dipole(dipole_files[1], [(3.0, 0.0, 0.0), (6.0, 0.0, 0.0)])
    _write_summary(summary_files[0], 0.0)       # weight=1
    _write_summary(summary_files[1], math.log(3.0))  # weight=3

    RUNNER.merge_dipole(dipole_files, summary_files, tmp_path / "dip_merged.out", [0, 1])

    merged = np.loadtxt(tmp_path / "dip_merged.out")
    # w0'=0.25, w1'=0.75
    # step 0: 0.25*1 + 0.75*3 = 2.5
    # step 1: 0.25*2 + 0.75*6 = 5.0
    assert merged[0, 1] == pytest.approx(2.5)
    assert merged[1, 1] == pytest.approx(5.0)
    assert merged[0, 0] == 0.0
    assert merged[1, 0] == 1.0


def test_merge_dipole_missing_file_raises(tmp_path):
    """Dipole merge should raise on missing file."""
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_summary(summary_files[0], 0.0)
    _write_summary(summary_files[1], 0.0)
    _write_dipole(tmp_path / "dip0.out", [(1.0, 0.0, 0.0)])

    with pytest.raises(ValueError, match="Missing dipole file"):
        RUNNER.merge_dipole(
            [tmp_path / "dip0.out", tmp_path / "dip1.out"],
            summary_files,
            tmp_path / "dip_merged.out",
            [0, 1],
        )


# ---------------------------------------------------------------------------
# Tests for merge_hnemd (P0-3: LSC-IVR NEMD workflow)
# ---------------------------------------------------------------------------

def _write_thermo(path: Path, values: list[list[float]]) -> None:
    """Write thermo.out: step, temperature, Kx, Ky, Kz, Px, Py, Pz."""
    lines = [" ".join([str(i)] + [str(v) for v in row]) for i, row in enumerate(values)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_merge_hnemd_wigner_weighted(tmp_path):
    """NEMD thermo merge should use Wigner weights."""
    thermo_files = [tmp_path / "thermo0.out", tmp_path / "thermo1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    # 2 steps, 4 columns (step, temp, Kx, Px)
    _write_thermo(thermo_files[0], [[300.0, 1.0, 10.0], [300.0, 2.0, 20.0]])
    _write_thermo(thermo_files[1], [[300.0, 3.0, 30.0], [300.0, 6.0, 60.0]])
    _write_summary(summary_files[0], 0.0)       # weight=1
    _write_summary(summary_files[1], math.log(3.0))  # weight=3

    RUNNER.merge_hnemd(thermo_files, summary_files, tmp_path / "thermo_merged.out", [0, 1])

    merged = np.loadtxt(tmp_path / "thermo_merged.out")
    # w0'=0.25, w1'=0.75
    # step 0, temp: 0.25*300 + 0.75*300 = 300
    # step 0, Kx: 0.25*1 + 0.75*3 = 2.5
    # step 0, Px: 0.25*10 + 0.75*30 = 25.0
    assert merged[0, 1] == pytest.approx(300.0)
    assert merged[0, 2] == pytest.approx(2.5)
    assert merged[0, 3] == pytest.approx(25.0)
    assert merged[1, 2] == pytest.approx(5.0)
    assert merged[1, 3] == pytest.approx(50.0)


def test_merge_hnemd_missing_file_raises(tmp_path):
    """NEMD merge should raise on missing thermo file."""
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_summary(summary_files[0], 0.0)
    _write_summary(summary_files[1], 0.0)
    _write_thermo(tmp_path / "thermo0.out", [[300.0, 1.0, 10.0]])

    with pytest.raises(ValueError, match="Missing thermo file"):
        RUNNER.merge_hnemd(
            [tmp_path / "thermo0.out", tmp_path / "thermo1.out"],
            summary_files,
            tmp_path / "thermo_merged.out",
            [0, 1],
        )


# ---------------------------------------------------------------------------
# Tests for _is_replica_complete (P0-4: checkpoint/resume)
# ---------------------------------------------------------------------------

def _write_complete_replica(run_dir: Path, workflow: str = "hac", seed: int = 12345) -> None:
    """Create a directory that looks like a completed replica run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    # Write required artifacts
    if workflow == "hac":
        _write_hac(run_dir / "hac.out", 1.0)
        _write_summary(run_dir / "qct_initial_summary.csv", 0.0, seed=seed)
    elif workflow == "dos":
        _write_dos(run_dir / "dos.out", [1.0, 2.0])
        _write_summary(run_dir / "qct_initial_summary.csv", 0.0, seed=seed)
    elif workflow == "ir":
        _write_dipole(run_dir / "dipole.out", [(1.0, 0.0, 0.0)])
        _write_summary(run_dir / "qct_initial_summary.csv", 0.0, seed=seed)
    elif workflow == "hnemd":
        _write_thermo(run_dir / "thermo.out", [[300.0, 1.0, 10.0]])
        _write_summary(run_dir / "qct_initial_summary.csv", 0.0, seed=seed)
    # gpumd.log with completion marker
    (run_dir / "gpumd.log").write_text("Running...\nFinished running GPUMD.\n", encoding="utf-8")
    # manifest
    manifest = {
        "replica": 0,
        "seed": seed,
        "device": "0",
        "status": "completed",
        "return_code": 0,
        "started_at": time.time() - 1.0,
        "completed_at": time.time(),
        "workflow": workflow,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_is_replica_complete_hac_success(tmp_path):
    """Completed HAC replica should be detected as complete."""
    run_dir = tmp_path / "replica_0"
    _write_complete_replica(run_dir, "hac", 12345)
    assert RUNNER._is_replica_complete(run_dir, "hac", 12345) is True


def test_is_replica_complete_dos_success(tmp_path):
    """Completed DOS replica should be detected as complete."""
    run_dir = tmp_path / "replica_0"
    _write_complete_replica(run_dir, "dos", 12345)
    assert RUNNER._is_replica_complete(run_dir, "dos", 12345) is True


def test_is_replica_complete_ir_success(tmp_path):
    """Completed IR replica should be detected as complete."""
    run_dir = tmp_path / "replica_0"
    _write_complete_replica(run_dir, "ir", 12345)
    assert RUNNER._is_replica_complete(run_dir, "ir", 12345) is True


def test_is_replica_complete_hnemd_success(tmp_path):
    """Completed NEMD replica should be detected as complete."""
    run_dir = tmp_path / "replica_0"
    _write_complete_replica(run_dir, "hnemd", 12345)
    assert RUNNER._is_replica_complete(run_dir, "hnemd", 12345) is True


def test_is_replica_complete_missing_manifest(tmp_path):
    """Replica without manifest should be incomplete."""
    run_dir = tmp_path / "replica_0"
    run_dir.mkdir()
    assert RUNNER._is_replica_complete(run_dir, "hac", 12345) is False


def test_is_replica_complete_missing_artifact(tmp_path):
    """Replica with missing hac.out should be incomplete."""
    run_dir = tmp_path / "replica_0"
    _write_complete_replica(run_dir, "hac", 12345)
    (run_dir / "hac.out").unlink()
    assert RUNNER._is_replica_complete(run_dir, "hac", 12345) is False


def test_is_replica_complete_wrong_seed(tmp_path):
    """Replica with wrong seed should be incomplete."""
    run_dir = tmp_path / "replica_0"
    _write_complete_replica(run_dir, "hac", 99999)
    assert RUNNER._is_replica_complete(run_dir, "hac", 12345) is False


def test_is_replica_complete_nonzero_return_code(tmp_path):
    """Replica with return_code != 0 should be incomplete."""
    run_dir = tmp_path / "replica_0"
    _write_complete_replica(run_dir, "hac", 12345)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["return_code"] = 1
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert RUNNER._is_replica_complete(run_dir, "hac", 12345) is False


def test_is_replica_complete_missing_completion_marker(tmp_path):
    """Replica without 'Finished running GPUMD' in log should be incomplete."""
    run_dir = tmp_path / "replica_0"
    _write_complete_replica(run_dir, "hac", 12345)
    (run_dir / "gpumd.log").write_text("Crashed!\n", encoding="utf-8")
    assert RUNNER._is_replica_complete(run_dir, "hac", 12345) is False


def test_is_replica_complete_empty_artifact(tmp_path):
    """Replica with empty artifact file should be incomplete."""
    run_dir = tmp_path / "replica_0"
    _write_complete_replica(run_dir, "hac", 12345)
    (run_dir / "hac.out").write_text("", encoding="utf-8")
    assert RUNNER._is_replica_complete(run_dir, "hac", 12345) is False


# ---------------------------------------------------------------------------
# Tests for _detect_workflow (all workflow types)
# ---------------------------------------------------------------------------

def test_detect_workflow_hac(tmp_path):
    """compute_hac keyword should be detected as hac workflow."""
    run_in = tmp_path / "run.in"
    run_in.write_text("ensemble lsc_ivr 300 seed 1 replicas 1\ncompute_hac 10 100 5\nrun 1000\n", encoding="utf-8")
    assert RUNNER._detect_workflow(run_in) == "hac"


def test_detect_workflow_hnemd(tmp_path):
    """compute_hnemd keyword should be detected as hnemd workflow."""
    run_in = tmp_path / "run.in"
    run_in.write_text("ensemble lsc_ivr 300 seed 1 replicas 1\ncompute_hnemd 0.001\nrun 1000\n", encoding="utf-8")
    assert RUNNER._detect_workflow(run_in) == "hnemd"


def test_detect_workflow_dos(tmp_path):
    """compute_dos keyword should be detected as dos workflow."""
    run_in = tmp_path / "run.in"
    run_in.write_text("ensemble lsc_ivr 300 seed 1 replicas 1\ncompute_dos 1 200 5\nrun 1000\n", encoding="utf-8")
    assert RUNNER._detect_workflow(run_in) == "dos"


def test_detect_workflow_ir(tmp_path):
    """dump_dipole keyword should be detected as ir workflow."""
    run_in = tmp_path / "run.in"
    run_in.write_text("ensemble lsc_ivr 300 seed 1 replicas 1\ndump_dipole 10\nrun 1000\n", encoding="utf-8")
    assert RUNNER._detect_workflow(run_in) == "ir"


def test_detect_workflow_trajectory(tmp_path):
    """dump_qct keyword should be detected as trajectory workflow."""
    run_in = tmp_path / "run.in"
    run_in.write_text("ensemble lsc_ivr 300 seed 1 replicas 1\ndump_qct 10\nrun 1000\n", encoding="utf-8")
    assert RUNNER._detect_workflow(run_in) == "trajectory"


def test_detect_workflow_no_keyword_raises(tmp_path):
    """No known keyword should raise ValueError."""
    run_in = tmp_path / "run.in"
    run_in.write_text("ensemble lsc_ivr 300 seed 1 replicas 1\nrun 1000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not infer workflow"):
        RUNNER._detect_workflow(run_in)


def test_detect_workflow_ignores_comments(tmp_path):
    """Commented keywords should not trigger workflow detection."""
    run_in = tmp_path / "run.in"
    run_in.write_text("# compute_hac 10 100 5\nensemble lsc_ivr 300 seed 1 replicas 1\ndump_qct 10\nrun 1000\n", encoding="utf-8")
    assert RUNNER._detect_workflow(run_in) == "trajectory"


# ---------------------------------------------------------------------------
# Tests for temperature_sweep (P1-2)
# ---------------------------------------------------------------------------

def test_temperature_sweep_creates_directories_and_index(tmp_path):
    """temperature_sweep should create per-T directories and an index file."""
    import tempfile

    template = tmp_path / "template"
    template.mkdir()
    (template / "model.xyz").write_text(
        "1\npbc=\"T T T\" Properties=species:S:1:pos:R:3:mass:R:1\nX 0 0 0 1\n",
        encoding="utf-8",
    )
    # Template run.in must have __TEMPERATURE__ placeholder
    (template / "run.in").write_text(
        "ensemble lsc_ivr __TEMPERATURE__ seed 1 replicas 1\n"
        "compute_hac 1 2 1\nrun 1\n",
        encoding="utf-8",
    )

    # Create a fake GPUMD that writes required outputs
    fake_gpumd = tmp_path / "fake_gpumd.py"
    fake_gpumd.write_text(
        "#!/usr/bin/env python3\n"
        "import re\n"
        "from pathlib import Path\n"
        "text = Path('run.in').read_text()\n"
        "seed = int(re.search(r'\\bseed\\s+(\\d+)', text).group(1))\n"
        "Path('qct_initial_summary.csv').write_text(\n"
        "    'replica,seed,wigner_weight,log_wigner_weight\\n' + f'0,{seed},1,0\\n')\n"
        "rows = [' '.join([str(t)] + ['1.0'] * 10) for t in (0, 1)]\n"
        "Path('hac.out').write_text('\\n'.join(rows) + '\\n')\n"
        "print('Finished running GPUMD.')\n",
        encoding="utf-8",
    )
    fake_gpumd.chmod(0o755)

    output = tmp_path / "sweep_output"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0,1"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/qct/run_multigpu.py"),
            "--template", str(template),
            "--gpumd", str(fake_gpumd),
            "--total-replicas", "2",
            "--num-gpus", "2",
            "--temperature-sweep", "100,300",
            "--replicas-per-temperature", "2",
            "--output", str(output),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    # Check per-T directories exist
    assert (output / "T_100K").is_dir()
    assert (output / "T_300K").is_dir()

    # Check index file
    index = json.loads((output / "temperature_sweep_index.json").read_text())
    assert index["temperatures_K"] == [100, 300]
    assert index["replicas_per_temperature"] == 2
    assert index["directories"] == ["T_100K", "T_300K"]

    # Check per-T template has temperature substituted
    t100_run = (output / "T_100K" / "template" / "run.in").read_text()
    assert "ensemble lsc_ivr 100 " in t100_run
    t300_run = (output / "T_300K" / "template" / "run.in").read_text()
    assert "ensemble lsc_ivr 300 " in t300_run


def test_temperature_sweep_requires_placeholder(tmp_path):
    """temperature_sweep should raise if __TEMPERATURE__ is missing."""
    template = tmp_path / "template"
    template.mkdir()
    (template / "model.xyz").write_text(
        "1\npbc=\"T T T\" Properties=species:S:1:pos:R:3:mass:R:1\nX 0 0 0 1\n",
        encoding="utf-8",
    )
    (template / "run.in").write_text(
        "ensemble lsc_ivr 300 seed 1 replicas 1\ncompute_hac 1 2 1\nrun 1\n",
        encoding="utf-8",
    )
    fake_gpumd = tmp_path / "fake_gpumd.py"
    fake_gpumd.write_text("#!/usr/bin/env python3\nprint('Finished running GPUMD.')\n", encoding="utf-8")
    fake_gpumd.chmod(0o755)

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools/qct/run_multigpu.py"),
                "--template", str(template),
                "--gpumd", str(fake_gpumd),
                "--total-replicas", "1",
                "--temperature-sweep", "100",
                "--output", str(tmp_path / "sweep"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )


# ---------------------------------------------------------------------------
# Tests for _required_artifacts (all workflow types)
# ---------------------------------------------------------------------------

def test_required_artifacts_all_workflows():
    """_required_artifacts should return correct tuples for each workflow."""
    hac = RUNNER._required_artifacts("hac")
    assert "qct_initial_summary.csv" in hac
    assert "hac.out" in hac

    hnemd = RUNNER._required_artifacts("hnemd")
    assert "qct_initial_summary.csv" in hnemd
    assert "thermo.out" in hnemd

    dos = RUNNER._required_artifacts("dos")
    assert "qct_initial_summary.csv" in dos
    assert "dos.out" in dos

    ir = RUNNER._required_artifacts("ir")
    assert "qct_initial_summary.csv" in ir
    assert "dipole.out" in ir

    trajectory = RUNNER._required_artifacts("trajectory")
    assert "qct_trajectory.xyz" in trajectory
    assert "qct_initial.xyz" in trajectory
    assert "qct_thermo.csv" in trajectory

    with pytest.raises(ValueError, match="Unsupported workflow"):
        RUNNER._required_artifacts("invalid")


# ---------------------------------------------------------------------------
# Tests for _merge_csv_artifact
# ---------------------------------------------------------------------------

def test_merge_csv_artifact_concatenates_replicas(tmp_path):
    """_merge_csv_artifact should concatenate CSV files with replica IDs."""
    files = []
    for rid in range(3):
        path = tmp_path / f"thermo_{rid}.csv"
        path.write_text(
            "replica,step,temperature\n"
            f"{rid},0,300\n"
            f"{rid},1,300\n",
            encoding="utf-8",
        )
        files.append(path)

    output = tmp_path / "merged.csv"
    RUNNER._merge_csv_artifact(files, output, [0, 1, 2])

    with output.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 6  # 2 rows per replica * 3 replicas
    assert "replica" in rows[0]
    # The merge function overwrites replica IDs with the correct values
    replicas_in_rows = sorted(int(r["replica"]) for r in rows)
    assert replicas_in_rows == [0, 0, 1, 1, 2, 2]


# ---------------------------------------------------------------------------
# Test for _atomic_json (atomic file writes)
# ---------------------------------------------------------------------------

def test_atomic_json_writes_valid_json(tmp_path):
    """_atomic_json should write valid JSON and not leave temp files."""
    path = tmp_path / "manifest.json"
    data = {"key": "value", "number": 42, "nested": {"a": 1}}
    RUNNER._atomic_json(path, data)

    result = json.loads(path.read_text(encoding="utf-8"))
    assert result == data
    # No temp file should remain
    assert not (tmp_path / "manifest.json.tmp").exists()


# ---------------------------------------------------------------------------
# Test for load_wigner_log_weights edge cases
# ---------------------------------------------------------------------------

def test_load_wigner_log_weights_rejects_nan(tmp_path):
    """load_wigner_log_weights should reject NaN."""
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "replica,seed,log_wigner_weight\n0,12345,nan\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid log_wigner_weight"):
        RUNNER.load_wigner_log_weights(summary)


def test_load_wigner_log_weights_rejects_inf(tmp_path):
    """load_wigner_log_weights should reject +inf."""
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "replica,seed,log_wigner_weight\n0,12345,inf\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid log_wigner_weight"):
        RUNNER.load_wigner_log_weights(summary)


def test_load_wigner_log_weights_linear_weight_zero(tmp_path):
    """Linear wigner_weight=0 should produce log weight -inf."""
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "replica,seed,wigner_weight\n0,12345,0\n",
        encoding="utf-8",
    )
    weights = RUNNER.load_wigner_log_weights(summary)
    assert weights[0] == -math.inf


def test_load_wigner_log_weights_duplicate_raises(tmp_path):
    """Duplicate replica rows should raise."""
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "replica,seed,log_wigner_weight\n0,12345,0\n0,12345,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate Wigner weight"):
        RUNNER.load_wigner_log_weights(summary)
