"""Tests for P2 LSC-IVR functions: symmetric correlation, adaptive timestep, mode correlations."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LSC_IVR_PATH = REPO_ROOT / "tools/qct/lsc_ivr.py"
ANALYZER_PATH = REPO_ROOT / "tools/qct/analyze_qct.py"

_spec_analyzer = importlib.util.spec_from_file_location("analyze_qct", ANALYZER_PATH)
mod_analyzer = importlib.util.module_from_spec(_spec_analyzer)
sys.modules["analyze_qct"] = mod_analyzer
_spec_analyzer.loader.exec_module(mod_analyzer)

_spec_lsc = importlib.util.spec_from_file_location("lsc_ivr", LSC_IVR_PATH)
mod_lsc = importlib.util.module_from_spec(_spec_lsc)
sys.modules["lsc_ivr"] = mod_lsc
_spec_lsc.loader.exec_module(mod_lsc)

aq = mod_analyzer
HBAR_EV_FS = mod_lsc.HBAR_EV_FS


# ---------------------------------------------------------------------------
# Helper to create synthetic frames
# ---------------------------------------------------------------------------

def make_frame(positions, velocities, masses, symbols, time_fs=0.0, replica=0, step=0):
    """Create an analyze_qct.Frame-like object."""
    return aq.Frame(
        symbols=symbols,
        positions=np.array(positions, dtype=float),
        masses=np.array(masses, dtype=float),
        velocities=np.array(velocities, dtype=float),
        lattice=None,
        pbc=False,
        time_fs=time_fs,
        metadata={"replica": replica, "step": step},
    )


def make_harmonic_frames(n_frames, dt_fs, omega, n_replicas=4):
    """Create frames for a 1D harmonic oscillator (single atom along x)."""
    rng = np.random.default_rng(42)
    masses = np.array([1.0])
    symbols = ["H"]
    replica_trajs = {}
    for rid in range(n_replicas):
        frames = []
        q0 = rng.normal(0.0, 1.0)
        p0 = rng.normal(0.0, 1.0)
        for i in range(n_frames):
            t = i * dt_fs
            q = q0 * math.cos(omega * t) + (p0 / omega) * math.sin(omega * t)
            p = -q0 * omega * math.sin(omega * t) + p0 * math.cos(omega * t)
            frames.append(make_frame(
                positions=[[q, 0.0, 0.0]],
                velocities=[[p, 0.0, 0.0]],
                masses=masses,
                symbols=symbols,
                time_fs=t,
                replica=rid,
                step=i,
            ))
        replica_trajs[rid] = frames
    return replica_trajs


# ---------------------------------------------------------------------------
# Tests for recommend_timestep (P2 §3.2)
# ---------------------------------------------------------------------------

class TestRecommendTimestep:
    def test_basic_recommendation(self):
        """Should recommend dt = period / steps_per_period."""
        freqs = np.array([10.0, 20.0, 50.0])  # THz
        dt = mod_lsc.recommend_timestep(freqs, target_steps_per_period=20)
        # Max freq = 50 THz → period = 1000/50 = 20 fs → dt = 20/20 = 1.0 fs
        assert dt == pytest.approx(1.0, abs=1e-6)

    def test_high_frequency_smaller_dt(self):
        """Higher frequency should give smaller timestep."""
        freqs_low = np.array([10.0])
        freqs_high = np.array([100.0])
        dt_low = mod_lsc.recommend_timestep(freqs_low, target_steps_per_period=20)
        dt_high = mod_lsc.recommend_timestep(freqs_high, target_steps_per_period=20)
        assert dt_high < dt_low

    def test_no_positive_frequencies(self):
        """All-zero or empty frequencies → max_dt."""
        freqs = np.array([0.0, 0.0, 0.0])
        dt = mod_lsc.recommend_timestep(freqs, max_dt_fs=2.0)
        assert dt == pytest.approx(2.0)

    def test_respects_min_max(self):
        """Should clamp to min/max."""
        freqs = np.array([1e6])  # extremely high → very small dt
        dt = mod_lsc.recommend_timestep(freqs, max_dt_fs=1.0, min_dt_fs=0.1)
        assert dt == pytest.approx(0.1)

    def test_custom_steps_per_period(self):
        """More steps per period → smaller dt."""
        freqs = np.array([5.0])
        dt_10 = mod_lsc.recommend_timestep(freqs, target_steps_per_period=10, max_dt_fs=100.0)
        dt_40 = mod_lsc.recommend_timestep(freqs, target_steps_per_period=40, max_dt_fs=100.0)
        assert dt_40 == pytest.approx(dt_10 / 4.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests for parse_hessian_frequencies (P2 §3.2)
# ---------------------------------------------------------------------------

class TestParseHessianFrequencies:
    def test_basic_load(self, tmp_path):
        """Should load frequencies from a text file."""
        freqs_data = np.array([10.0, 20.0, 30.0, 0.0, -5.0])
        path = tmp_path / "hessian.out"
        np.savetxt(path, freqs_data)
        result = mod_lsc.parse_hessian_frequencies(path)
        assert len(result) == 5
        assert result[0] == pytest.approx(10.0)

    def test_single_value(self, tmp_path):
        """Single frequency should work."""
        path = tmp_path / "hessian.out"
        np.savetxt(path, np.array([42.0]))
        result = mod_lsc.parse_hessian_frequencies(path)
        assert result[0] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Tests for compute_symmetric_correlation (P2 §3.1)
# ---------------------------------------------------------------------------

class TestSymmetricCorrelation:
    def test_harmonic_oscillator(self):
        """Symmetric correlation of harmonic oscillator should be cos(ωt)."""
        omega = 0.1  # rad/fs
        dt_fs = 1.0
        n_frames = 101  # odd so midpoint is exact
        replica_trajs = make_harmonic_frames(n_frames, dt_fs, omega, n_replicas=4)
        weights = {rid: 1.0 for rid in range(4)}

        # Position operator (x-component, atom 0)
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})

        corr = mod_lsc.compute_symmetric_correlation(
            replica_trajs, weights, op, op,
        )

        # Check that correlation at t=0 is positive (variance of Q0)
        assert corr.c_ab_normalized[0] > 0

        # Check oscillation: at small t, correlation should be decreasing from C(0)
        if len(corr.c_ab_normalized) > 3:
            assert corr.c_ab_normalized[1] < corr.c_ab_normalized[0]

    def test_equal_weights(self):
        """With equal weights, result should match simple average."""
        omega = 0.05
        dt_fs = 1.0
        n_frames = 51
        replica_trajs = make_harmonic_frames(n_frames, dt_fs, omega, n_replicas=3)
        weights = {rid: 1.0 for rid in range(3)}

        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})

        corr = mod_lsc.compute_symmetric_correlation(
            replica_trajs, weights, op, op,
        )

        assert len(corr.c_ab_normalized) > 0
        assert len(corr.time_fs) == len(corr.c_ab_normalized)
        assert np.all(np.isfinite(corr.c_ab_normalized))

    def test_max_lag_truncation(self):
        """max_lag_fs should truncate the correlation length."""
        omega = 0.05
        dt_fs = 1.0
        n_frames = 101
        replica_trajs = make_harmonic_frames(n_frames, dt_fs, omega, n_replicas=2)
        weights = {rid: 1.0 for rid in range(2)}

        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})

        corr_full = mod_lsc.compute_symmetric_correlation(
            replica_trajs, weights, op, op,
        )
        corr_trunc = mod_lsc.compute_symmetric_correlation(
            replica_trajs, weights, op, op, max_lag_fs=5.0,
        )

        assert len(corr_trunc.c_ab_normalized) <= len(corr_full.c_ab_normalized)
        assert corr_trunc.time_fs[-1] <= 5.0 + dt_fs

    def test_zero_weights_raise(self):
        """All-zero weights should raise."""
        n_frames = 21
        replica_trajs = make_harmonic_frames(n_frames, 1.0, 0.1, n_replicas=2)
        weights = {rid: 0.0 for rid in range(2)}
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})

        with pytest.raises(ValueError, match="zero"):
            mod_lsc.compute_symmetric_correlation(replica_trajs, weights, op, op)


# ---------------------------------------------------------------------------
# Tests for compute_mode_correlations (P2 §4.6)
# ---------------------------------------------------------------------------

class TestModeCorrelations:
    def test_single_mode_harmonic(self):
        """Single-mode correlation should be oscillatory."""
        n_frames = 50
        dt_fs = 1.0
        omega = 0.1
        replica_trajs = make_harmonic_frames(n_frames, dt_fs, omega, n_replicas=2)
        weights = {rid: 1.0 for rid in range(2)}

        # Single atom: eigenvector is [1, 0, 0] (x mode)
        eigvecs = np.array([[1.0, 0.0, 0.0]])
        masses = np.array([1.0])

        results = mod_lsc.compute_mode_correlations(
            replica_trajs, weights, eigvecs, masses,
            dt_fs=dt_fs,
        )

        assert len(results) == 1
        assert results[0].mode_index == 0
        assert len(results[0].c_kk_normalized) > 0
        assert np.all(np.isfinite(results[0].c_kk_normalized))

    def test_multiple_modes(self):
        """Multiple modes should each have their own correlation."""
        n_frames = 30
        dt_fs = 1.0
        omega = 0.1
        replica_trajs = make_harmonic_frames(n_frames, dt_fs, omega, n_replicas=2)
        weights = {rid: 1.0 for rid in range(2)}

        # Two modes: x and y (y is zero since only x oscillates)
        eigvecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        masses = np.array([1.0])

        results = mod_lsc.compute_mode_correlations(
            replica_trajs, weights, eigvecs, masses,
            dt_fs=dt_fs,
        )

        assert len(results) == 2
        # Mode 0 (x) should have nonzero correlation
        assert abs(results[0].c_kk_normalized[0]) > 1e-10
        # Mode 1 (y) should have ~zero correlation since no y motion
        assert abs(results[1].c_kk_normalized[0]) < 1e-6

    def test_max_lag(self):
        """max_lag_fs should truncate."""
        n_frames = 50
        dt_fs = 1.0
        omega = 0.1
        replica_trajs = make_harmonic_frames(n_frames, dt_fs, omega, n_replicas=2)
        weights = {rid: 1.0 for rid in range(2)}

        eigvecs = np.array([[1.0, 0.0, 0.0]])
        masses = np.array([1.0])

        results = mod_lsc.compute_mode_correlations(
            replica_trajs, weights, eigvecs, masses,
            dt_fs=dt_fs,
            max_lag_fs=10.0,
        )

        assert results[0].time_fs[-1] <= 10.0 + dt_fs

    def test_write_csv(self, tmp_path):
        """CSV output should be valid."""
        n_frames = 20
        dt_fs = 1.0
        omega = 0.1
        replica_trajs = make_harmonic_frames(n_frames, dt_fs, omega, n_replicas=2)
        weights = {rid: 1.0 for rid in range(2)}

        eigvecs = np.array([[1.0, 0.0, 0.0]])
        masses = np.array([1.0])

        results = mod_lsc.compute_mode_correlations(
            replica_trajs, weights, eigvecs, masses,
            dt_fs=dt_fs,
        )

        path = tmp_path / "modes.csv"
        mod_lsc.write_mode_correlations_csv(path, results)
        assert path.is_file()

        import csv
        with path.open() as f:
            reader = csv.reader(f)
            rows = list(reader)

        # header + n_frames data rows
        assert rows[0] == ["mode", "frequency_THz", "time_fs",
                          "C_kk_raw", "C_kk_normalized", "std_error"]
        assert len(rows) == 1 + n_frames  # 1 header + data
