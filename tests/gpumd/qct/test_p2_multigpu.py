"""Tests for P2 run_multigpu.py functions: merge_sdc and blockwise HAC uncertainty."""

import importlib.util
import json
import math
import sys
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


def _write_sdc(path: Path, value: float, n_rows: int = 10) -> None:
    """Write sdc.out with 7 columns: time_ps, msd_x, msd_y, msd_z, sdc_x, sdc_y, sdc_z."""
    data = np.zeros((n_rows, 7), dtype=float)
    data[:, 0] = np.arange(n_rows, dtype=float) * 0.001  # time in ps
    data[:, 1:4] = value  # msd
    data[:, 4:7] = value * 2  # sdc
    np.savetxt(path, data)


# ---------------------------------------------------------------------------
# Tests for SDC workflow detection (P2 §4.3)
# ---------------------------------------------------------------------------

def test_detect_workflow_sdc(tmp_path):
    """compute_sdc keyword should be detected as sdc workflow."""
    run_in = tmp_path / "run.in"
    run_in.write_text(
        "ensemble lsc_ivr 300 seed 1 replicas 1\ncompute_sdc 10 100 5\nrun 1000\n",
        encoding="utf-8",
    )
    assert RUNNER._detect_workflow(run_in) == "sdc"


def test_required_artifacts_sdc():
    """SDC workflow should require qct_initial_summary.csv and sdc.out."""
    artifacts = RUNNER._required_artifacts("sdc")
    assert "qct_initial_summary.csv" in artifacts
    assert "sdc.out" in artifacts


def test_sdc_in_generated_artifacts():
    """sdc.out should be in GENERATED_ARTIFACTS."""
    assert "sdc.out" in RUNNER.GENERATED_ARTIFACTS


# ---------------------------------------------------------------------------
# Tests for merge_sdc (P2 §4.3)
# ---------------------------------------------------------------------------

def test_merge_sdc_wigner_weighted(tmp_path):
    """merge_sdc should apply Wigner weights correctly."""
    sdc_files = [tmp_path / "sdc0.out", tmp_path / "sdc1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_sdc(sdc_files[0], 1.0)
    _write_sdc(sdc_files[1], 3.0)
    _write_summary(summary_files[0], 0.0)  # w=1
    _write_summary(summary_files[1], math.log(2.0))  # w=2

    RUNNER.merge_sdc(sdc_files, summary_files, tmp_path / "merged_sdc.out", [0, 1])

    merged = np.loadtxt(tmp_path / "merged_sdc.out")
    # Weighted average: (1*1 + 2*3) / (1+2) = 7/3 ≈ 2.333 for msd
    expected_msd = (1.0 + 2.0 * 3.0) / 3.0
    expected_sdc = (1.0 * 2.0 + 2.0 * 6.0) / 3.0
    assert np.allclose(merged[:, 1:4], expected_msd)
    assert np.allclose(merged[:, 4:7], expected_sdc)


def test_merge_sdc_equal_weights(tmp_path):
    """Equal weights → simple average."""
    sdc_files = [tmp_path / "sdc0.out", tmp_path / "sdc1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_sdc(sdc_files[0], 2.0)
    _write_sdc(sdc_files[1], 4.0)
    _write_summary(summary_files[0], 0.0)
    _write_summary(summary_files[1], 0.0)

    RUNNER.merge_sdc(sdc_files, summary_files, tmp_path / "merged.out", [0, 1])

    merged = np.loadtxt(tmp_path / "merged.out")
    assert np.allclose(merged[:, 1:4], 3.0)  # (2+4)/2
    assert np.allclose(merged[:, 4:7], 6.0)  # (4+8)/2


def test_merge_sdc_time_grid_mismatch(tmp_path):
    """Time grid mismatch should raise."""
    sdc_files = [tmp_path / "sdc0.out", tmp_path / "sdc1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_sdc(sdc_files[0], 2.0)
    _write_sdc(sdc_files[1], 4.0)

    # Different time grids
    data = np.zeros((10, 7))
    data[:, 0] = np.arange(10) * 0.002  # different dt
    data[:, 1:4] = 4.0
    np.savetxt(sdc_files[1], data)

    _write_summary(summary_files[0], 0.0)
    _write_summary(summary_files[1], 0.0)

    with pytest.raises(ValueError, match="time-grid mismatch"):
        RUNNER.merge_sdc(sdc_files, summary_files, tmp_path / "merged.out", [0, 1])


def test_merge_sdc_missing_file(tmp_path):
    """Missing SDC file should raise."""
    sdc_files = [tmp_path / "sdc0.out", tmp_path / "missing.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_sdc(sdc_files[0], 2.0)
    _write_summary(summary_files[0], 0.0)
    _write_summary(summary_files[1], 0.0)

    with pytest.raises(ValueError, match="Missing SDC file"):
        RUNNER.merge_sdc(sdc_files, summary_files, tmp_path / "merged.out", [0, 1])


def test_merge_sdc_empty_list(tmp_path):
    """Empty file list should be a no-op."""
    RUNNER.merge_sdc([], [], tmp_path / "merged.out", [])
    assert not (tmp_path / "merged.out").exists()


# ---------------------------------------------------------------------------
# Tests for blockwise HAC uncertainty (P2 §7.1)
# ---------------------------------------------------------------------------

def test_merge_hac_blockwise_uncertainty(tmp_path):
    """merge_hac with block_size should produce blockwise uncertainty columns."""
    # Need enough rows for block analysis: 20 rows, block_size=5 → 4 blocks
    n_rows = 20
    hac_files = [tmp_path / "hac0.out", tmp_path / "hac1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]

    for i, (hf, sf) in enumerate(zip(hac_files, summary_files)):
        data = np.zeros((n_rows, 11))
        data[:, 0] = np.arange(n_rows, dtype=float) * 0.1  # time in ps
        data[:, 1:] = float(i + 1)  # different constant values
        np.savetxt(hf, data)
        _write_summary(sf, 0.0)  # equal weights

    diagnostics = RUNNER.merge_hac(
        hac_files,
        summary_files,
        tmp_path / "merged.out",
        [0, 1],
        tmp_path / "uncertainty.csv",
        tmp_path / "manifest.json",
        block_size=5,
    )

    assert "blockwise_uncertainty" in diagnostics
    assert diagnostics["blockwise_uncertainty"]["block_size"] == 5
    assert diagnostics["blockwise_uncertainty"]["n_blocks_total"] == 8  # 4 blocks * 2 replicas

    # Check uncertainty file has block columns
    import csv
    with (tmp_path / "uncertainty.csv").open() as f:
        reader = csv.reader(f)
        header = next(reader)
    assert "block_se_kappa_avg" in header
    assert "combined_se_kappa_avg" in header


def test_merge_hac_no_blockwise_when_disabled(tmp_path):
    """Without block_size, blockwise uncertainty should be zero/None."""
    hac_files = [tmp_path / "hac0.out", tmp_path / "hac1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_hac(hac_files[0], 2.0)
    _write_hac(hac_files[1], 4.0)
    _write_summary(summary_files[0], 0.0)
    _write_summary(summary_files[1], 0.0)

    diagnostics = RUNNER.merge_hac(
        hac_files,
        summary_files,
        tmp_path / "merged.out",
        [0, 1],
        tmp_path / "uncertainty.csv",
        tmp_path / "manifest.json",
    )

    assert diagnostics["blockwise_uncertainty"]["block_size"] is None
    assert diagnostics["blockwise_uncertainty"]["n_blocks_total"] == 0


def test_merge_hac_blockwise_too_short(tmp_path):
    """Block size larger than data should not crash, just no blocks."""
    hac_files = [tmp_path / "hac0.out", tmp_path / "hac1.out"]
    summary_files = [tmp_path / "summary0.csv", tmp_path / "summary1.csv"]
    _write_hac(hac_files[0], 2.0)  # only 2 rows
    _write_hac(hac_files[1], 4.0)
    _write_summary(summary_files[0], 0.0)
    _write_summary(summary_files[1], 0.0)

    diagnostics = RUNNER.merge_hac(
        hac_files,
        summary_files,
        tmp_path / "merged.out",
        [0, 1],
        tmp_path / "uncertainty.csv",
        tmp_path / "manifest.json",
        block_size=10,  # > 2*2=4 rows available
    )

    # Should not crash, just no blockwise analysis
    assert diagnostics["blockwise_uncertainty"]["n_blocks_total"] == 0
