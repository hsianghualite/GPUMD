"""Tests for the FB-LSC-IVR post-processing tool (tools/qct/fb_lsc_ivr.py).

Tests:
1. HAC file loading and validation
2. Wigner-weighted merge
3. FB kappa computation (Green-Kubo integration)
4. Blockwise uncertainty separation
5. CSV output format
6. Manifest output
7. Error handling (mismatched grids, missing files, zero weights)
"""

import importlib.util
import json
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FB_PATH = REPO_ROOT / "tools/qct/fb_lsc_ivr.py"

_spec = importlib.util.spec_from_file_location("fb_lsc_ivr", FB_PATH)
mod = importlib.util.module_from_spec(_spec)
sys.modules["fb_lsc_ivr"] = mod
_spec.loader.exec_module(mod)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _write_hac(path: Path, value: float, n_rows: int = 10, dt_ps: float = 0.01):
    """Write a synthetic hac.out file with 11 columns."""
    data = np.zeros((n_rows, 11))
    data[:, 0] = np.arange(n_rows) * dt_ps  # time in ps
    # HAC columns (1-5): use value
    data[:, 1:6] = value
    # RTC columns (6-10): cumulative integral (trapezoidal)
    for i in range(1, n_rows):
        dt = data[i, 0] - data[i-1, 0]
        data[i, 6:11] = data[i-1, 6:11] + (data[i, 1:6] + data[i-1, 1:6]) * 0.5 * dt
    np.savetxt(path, data, fmt="%25.15e")


def _write_summary(path: Path, log_weight: float, seed: int = 1):
    """Write a synthetic qct_initial_summary.csv file."""
    path.write_text(
        "replica,seed,wigner_weight,log_wigner_weight\n"
        f"0,{seed},1,{log_weight}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Tests: HAC file loading
# ---------------------------------------------------------------------------

class TestLoadHacFile:
    def test_basic_load(self, tmp_path):
        """Should load a valid hac.out file."""
        path = tmp_path / "hac.out"
        _write_hac(path, 1.0, n_rows=5)
        data = mod.load_hac_file(path)
        assert len(data.time_ps) == 5
        assert data.hac.shape == (5, 5)
        assert data.rtc.shape == (5, 5)
        assert np.allclose(data.hac, 1.0)

    def test_wrong_columns(self, tmp_path):
        """Should reject files with wrong number of columns."""
        path = tmp_path / "hac.out"
        np.savetxt(path, np.zeros((5, 10)))
        with pytest.raises(ValueError, match="11 columns"):
            mod.load_hac_file(path)

    def test_non_finite_data(self, tmp_path):
        """Should reject non-finite data."""
        path = tmp_path / "hac.out"
        data = np.zeros((5, 11))
        data[2, 1] = float('nan')
        np.savetxt(path, data)
        with pytest.raises(ValueError, match="non-finite"):
            mod.load_hac_file(path)

    def test_non_increasing_time(self, tmp_path):
        """Should reject non-increasing time grids."""
        path = tmp_path / "hac.out"
        data = np.zeros((3, 11))
        data[:, 0] = [0.0, 2.0, 1.0]  # not increasing
        np.savetxt(path, data)
        with pytest.raises(ValueError, match="increasing"):
            mod.load_hac_file(path)

    def test_single_row(self, tmp_path):
        """Should handle single-row files."""
        path = tmp_path / "hac.out"
        data = np.zeros((1, 11))
        data[0, 0] = 0.0
        data[0, 1:6] = 3.0
        np.savetxt(path, data)
        result = mod.load_hac_file(path)
        assert len(result.time_ps) == 1


# ---------------------------------------------------------------------------
# Tests: Wigner weight loading
# ---------------------------------------------------------------------------

class TestLoadWignerWeights:
    def test_basic_load(self, tmp_path):
        """Should load log_wigner_weight from summary."""
        path = tmp_path / "summary.csv"
        _write_summary(path, math.log(2.0))
        assert mod.load_wigner_log_weights(path) == pytest.approx(math.log(2.0))

    def test_no_weight_column(self, tmp_path):
        """Should default to 0.0 (classical, w=1) when no weight column."""
        path = tmp_path / "summary.csv"
        path.write_text("replica,seed\n0,1\n", encoding="utf-8")
        assert mod.load_wigner_log_weights(path) == 0.0

    def test_multiple_rows_error(self, tmp_path):
        """Should error on multiple rows."""
        path = tmp_path / "summary.csv"
        path.write_text(
            "replica,seed,log_wigner_weight\n0,1,0.0\n1,2,1.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="1 row"):
            mod.load_wigner_log_weights(path)


# ---------------------------------------------------------------------------
# Tests: Multi-replica loading
# ---------------------------------------------------------------------------

class TestLoadMultipleHac:
    def test_basic_multi_load(self, tmp_path):
        """Should load multiple hac.out files."""
        files = [tmp_path / f"hac{i}.out" for i in range(3)]
        for f in files:
            _write_hac(f, 1.0, n_rows=10)

        datasets = mod.load_multiple_hac(files)
        assert len(datasets) == 3
        for d in datasets:
            assert d.log_wigner_weight == 0.0  # no summary → classical

    def test_with_summary(self, tmp_path):
        """Should load Wigner weights from summary files."""
        hac_files = [tmp_path / f"hac{i}.out" for i in range(2)]
        summary_files = [tmp_path / f"s{i}.csv" for i in range(2)]
        for f in hac_files:
            _write_hac(f, 1.0, n_rows=10)
        _write_summary(summary_files[0], 0.0)
        _write_summary(summary_files[1], math.log(3.0))

        datasets = mod.load_multiple_hac(hac_files, summary_files)
        assert datasets[0].log_wigner_weight == 0.0
        assert datasets[1].log_wigner_weight == pytest.approx(math.log(3.0))

    def test_time_grid_mismatch(self, tmp_path):
        """Should reject mismatched time grids."""
        f1 = tmp_path / "hac0.out"
        f2 = tmp_path / "hac1.out"
        _write_hac(f1, 1.0, n_rows=10, dt_ps=0.01)
        _write_hac(f2, 2.0, n_rows=10, dt_ps=0.02)

        with pytest.raises(ValueError, match="Time grid mismatch"):
            mod.load_multiple_hac([f1, f2])

    def test_empty_list(self):
        """Should reject empty file list."""
        with pytest.raises(ValueError, match="No hac.out"):
            mod.load_multiple_hac([])


# ---------------------------------------------------------------------------
# Tests: Weight normalization
# ---------------------------------------------------------------------------

class TestNormalizeWeights:
    def test_equal_weights(self):
        """Equal log weights → equal normalized weights."""
        normalized = mod._normalize_weights([0.0, 0.0, 0.0])
        assert np.allclose(normalized, 1.0 / 3.0)

    def test_weighted(self):
        """Unequal weights should normalize correctly."""
        w1 = 0.0  # log(1)
        w2 = math.log(3.0)  # log(3)
        normalized = mod._normalize_weights([w1, w2])
        assert normalized[0] == pytest.approx(0.25)
        assert normalized[1] == pytest.approx(0.75)

    def test_zero_weight(self):
        """-inf weight → 0 normalized."""
        normalized = mod._normalize_weights([0.0, -math.inf])
        assert normalized[0] == pytest.approx(1.0)
        assert normalized[1] == pytest.approx(0.0)

    def test_all_zero_raises(self):
        """All -inf weights should raise."""
        with pytest.raises(ValueError, match="All Wigner weights are zero"):
            mod._normalize_weights([-math.inf, -math.inf])


# ---------------------------------------------------------------------------
# Tests: FB kappa computation
# ---------------------------------------------------------------------------

class TestFBKappa:
    def test_single_replica(self, tmp_path):
        """Single replica → no uncertainty, just the RTC values."""
        path = tmp_path / "hac0.out"
        _write_hac(path, 1.0, n_rows=10)

        datasets = mod.load_multiple_hac([path])
        result = mod.compute_fb_kappa(datasets)

        assert result.n_replicas == 1
        assert result.effective_replicas == pytest.approx(1.0)
        assert len(result.kappa_x) == 10
        # All SE should be 0 for single replica
        assert np.allclose(result.kappa_x_se, 0.0)

    def test_equal_weights_two_replicas(self, tmp_path):
        """Two equal-weight replicas → simple average."""
        f1 = tmp_path / "hac0.out"
        f2 = tmp_path / "hac1.out"
        _write_hac(f1, 2.0, n_rows=10)
        _write_hac(f2, 4.0, n_rows=10)

        datasets = mod.load_multiple_hac([f1, f2])
        result = mod.compute_fb_kappa(datasets)

        assert result.n_replicas == 2
        assert result.effective_replicas == pytest.approx(2.0)
        # Kappa should be average of two replicas
        assert result.kappa_x[5] > 0  # should be nonzero from RTC

    def test_wigner_weighted(self, tmp_path):
        """Wigner weights should bias the merge."""
        f1 = tmp_path / "hac0.out"
        f2 = tmp_path / "hac1.out"
        s1 = tmp_path / "s0.csv"
        s2 = tmp_path / "s1.csv"
        _write_hac(f1, 2.0, n_rows=10)
        _write_hac(f2, 4.0, n_rows=10)
        _write_summary(s1, 0.0)       # w=1
        _write_summary(s2, math.log(3.0))  # w=3

        datasets = mod.load_multiple_hac([f1, f2], [s1, s2])
        result = mod.compute_fb_kappa(datasets)

        # w1=1/4, w2=3/4 → effective_replicas = 1/(1/16 + 9/16) = 16/10 = 1.6
        assert result.effective_replicas == pytest.approx(1.6, abs=0.01)

    def test_blockwise_uncertainty(self, tmp_path):
        """Block size should produce blockwise uncertainty."""
        f1 = tmp_path / "hac0.out"
        f2 = tmp_path / "hac1.out"
        _write_hac(f1, 2.0, n_rows=20)
        _write_hac(f2, 4.0, n_rows=20)

        datasets = mod.load_multiple_hac([f1, f2])
        result = mod.compute_fb_kappa(datasets, block_size=5)

        assert result.block_size == 5
        assert result.n_blocks > 0  # 4 blocks/replica * 2 = 8
        assert np.all(np.isfinite(result.block_se_kappa_avg))

    def test_blockwise_disabled(self, tmp_path):
        """Without block_size, blockwise SE should be zero."""
        f1 = tmp_path / "hac0.out"
        f2 = tmp_path / "hac1.out"
        _write_hac(f1, 2.0, n_rows=10)
        _write_hac(f2, 4.0, n_rows=10)

        datasets = mod.load_multiple_hac([f1, f2])
        result = mod.compute_fb_kappa(datasets)

        assert result.block_size is None
        assert result.n_blocks == 0
        assert np.allclose(result.block_se_kappa_avg, 0.0)

    def test_blockwise_too_short(self, tmp_path):
        """Block size larger than data should not crash."""
        f1 = tmp_path / "hac0.out"
        f2 = tmp_path / "hac1.out"
        _write_hac(f1, 2.0, n_rows=5)
        _write_hac(f2, 4.0, n_rows=5)

        datasets = mod.load_multiple_hac([f1, f2])
        result = mod.compute_fb_kappa(datasets, block_size=10)

        assert result.n_blocks == 0  # too short for blocks

    def test_combined_se(self, tmp_path):
        """Combined SE should be sqrt(replica_SE^2 + block_SE^2)."""
        f1 = tmp_path / "hac0.out"
        f2 = tmp_path / "hac1.out"
        _write_hac(f1, 2.0, n_rows=20)
        _write_hac(f2, 4.0, n_rows=20)

        datasets = mod.load_multiple_hac([f1, f2])
        result = mod.compute_fb_kappa(datasets, block_size=5)

        expected = np.sqrt(result.kappa_avg_se ** 2 + result.block_se_kappa_avg ** 2)
        assert np.allclose(result.combined_se_kappa_avg, expected)


# ---------------------------------------------------------------------------
# Tests: CSV output
# ---------------------------------------------------------------------------

class TestCSVOutput:
    def test_csv_written(self, tmp_path):
        """Should write a valid CSV file."""
        f1 = tmp_path / "hac0.out"
        _write_hac(f1, 1.0, n_rows=5)

        datasets = mod.load_multiple_hac([f1])
        result = mod.compute_fb_kappa(datasets)

        out = tmp_path / "fb_kappa.csv"
        mod.write_fb_kappa_csv(out, result)
        assert out.is_file()

        with out.open() as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0][0] == "time_ps"
        assert "kappa_avg" in rows[0]
        assert "combined_se_kappa_avg" in rows[0]
        assert len(rows) == 6  # header + 5 data

    def test_manifest_written(self, tmp_path):
        """Should write a valid JSON manifest."""
        f1 = tmp_path / "hac0.out"
        _write_hac(f1, 1.0, n_rows=5)

        datasets = mod.load_multiple_hac([f1])
        result = mod.compute_fb_kappa(datasets)

        out = tmp_path / "manifest.json"
        mod.write_fb_manifest(out, result, [f1])
        assert out.is_file()

        manifest = json.loads(out.read_text())
        assert manifest["schema"] == "FB_LSC_IVR_KAPPA_v1"
        assert manifest["n_replicas"] == 1
        assert "endpoint" in manifest
        assert "kappa_avg" in manifest["endpoint"]


# ---------------------------------------------------------------------------
# Tests: CLI smoke test (HAC mode)
# ---------------------------------------------------------------------------

class TestCLI:
    def test_hac_mode_cli(self, tmp_path):
        """Should run via CLI in HAC mode."""
        f1 = tmp_path / "hac0.out"
        f2 = tmp_path / "hac1.out"
        _write_hac(f1, 2.0, n_rows=10)
        _write_hac(f2, 4.0, n_rows=10)

        out = tmp_path / "out.csv"
        manifest = tmp_path / "manifest.json"

        import subprocess
        result = subprocess.run(
            [sys.executable, str(FB_PATH),
             "--hac", str(f1), str(f2),
             "--output", str(out),
             "--manifest", str(manifest),
             "--block-size", "5"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        assert result.returncode == 0
        assert out.is_file()
        assert manifest.is_file()

    def test_no_mode_raises(self):
        """Should require --hac or --trajectory."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(FB_PATH), "--output", "/tmp/dummy.csv"],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode != 0
