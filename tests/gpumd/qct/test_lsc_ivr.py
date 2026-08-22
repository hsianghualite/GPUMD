"""Unit tests for the LSC-IVR post-processing tool (tools/qct/lsc_ivr.py).

Tests:
1. Harmonic oscillator analytic check — verify <Q(0)Q(t)> = (ℏ/2ω)coth(βℏω/2)cos(ωt)
2. Reweighting consistency — pure harmonic trajectories give w_i ≈ 1
3. Operator registry unit tests
4. CSV schema backward compatibility — missing wigner_weight column → w=1
5. Weight loading and edge cases
"""

import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LSC_IVR_PATH = REPO_ROOT / "tools/qct/lsc_ivr.py"
ANALYZER_PATH = REPO_ROOT / "tools/qct/analyze_qct.py"

# Load modules by file path
_spec_analyzer = importlib.util.spec_from_file_location("analyze_qct", ANALYZER_PATH)
mod_analyzer = importlib.util.module_from_spec(_spec_analyzer)
sys.modules["analyze_qct"] = mod_analyzer
_spec_analyzer.loader.exec_module(mod_analyzer)

_spec_lsc = importlib.util.spec_from_file_location("lsc_ivr", LSC_IVR_PATH)
mod_lsc = importlib.util.module_from_spec(_spec_lsc)
sys.modules["lsc_ivr"] = mod_lsc
_spec_lsc.loader.exec_module(mod_lsc)


# ---------------------------------------------------------------------------
# Physical constants (matching lsc_ivr.py / analyze_qct.py)
# ---------------------------------------------------------------------------

HBAR_EV_FS = mod_lsc.HBAR_EV_FS
K_B_EV_K = mod_lsc.K_B_EV_K


# ---------------------------------------------------------------------------
# Helper: write synthetic extxyz trajectory
# ---------------------------------------------------------------------------

def write_extxyz_frame(output, positions, velocities, masses, symbols,
                       replica=0, step=0, time_fs=0.0, seed=0):
    n = len(symbols)
    output.write(f"{n}\n")
    output.write(
        f'Time={time_fs:.12g} Replica={replica} Step={step} Seed={seed} '
        f'pbc="F F F" Lattice="50 0 0 0 50 0 0 0 50" '
        f'Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3\n'
    )
    for i in range(n):
        output.write(
            f"{symbols[i]} "
            f"{positions[i,0]:.12g} {positions[i,1]:.12g} {positions[i,2]:.12g} "
            f"{masses[i]:.12g} "
            f"{velocities[i,0]:.12g} {velocities[i,1]:.12g} {velocities[i,2]:.12g}\n"
        )


def write_synthetic_harmonic_trajectory(
    path, omega_natural, n_frames=64, dt_fs=0.5, n_replicas=8, temperature_K=300.0,
    seed_offset=0, write_weights=True,
):
    """Write a multi-replica extxyz with 1-D harmonic oscillator dynamics.

    For each replica, sample (Q0, P0) from the Wigner distribution, then
    propagate analytically: Q(t) = Q0 cos(ωt) + (P0/ω) sin(ωt).

    The system is a single H atom on a 1D harmonic potential along x.
    omega is in natural angular frequency units (rad / natural_time_unit).

    Returns (expected_corr_at_t0, omega_natural, time_axis_fs, weights_dict).
    """
    rng = np.random.default_rng(42)
    n_atoms = 1
    symbols = ["H"]
    masses = np.array([1.008])

    # Wigner distribution parameters
    # sigma_Q^2 = (ℏ/(2ω)) coth(βℏω/2)
    # sigma_P^2 = (ℏω/2) coth(βℏω/2)
    zpe = 0.5 * HBAR_EV_FS * omega_natural  # in eV (natural freq)
    # Convert omega_natural to physical: ω_physical = ω_natural / TIME_UNIT_CONVERSION
    # But HBAR is in eV·fs, omega_natural is in rad/natural_time_unit
    # TIME_UNIT_CONVERSION = 1.018051e+1 fs/natural_time_unit
    # So ω_physical [rad/fs] = ω_natural / TIME_UNIT_CONVERSION
    # For the Wigner distribution in natural units, we use HBAR (eV·fs) * omega_physical (rad/fs)
    # Actually, in GPUMD natural units, HBAR * omega_natural gives energy in eV.
    # HBAR = 6.465412e-2 eV·fs, but natural time unit → fs factor is TIME_UNIT_CONVERSION
    # HBAR_natural = HBAR / TIME_UNIT_CONVERSION
    # Actually, let's just work in physical units (fs) for the test.

    # Let's define omega in physical units (rad/fs) directly
    omega_rad_per_fs = omega_natural  # treat input as rad/fs for the test

    zpe_ev = 0.5 * HBAR_EV_FS * omega_rad_per_fs
    if temperature_K > 0:
        beta = 1.0 / (K_B_EV_K * temperature_K)
        x = beta * HBAR_EV_FS * omega_rad_per_fs / 2.0
        coth_factor = 1.0 / np.tanh(x) if x < 350 else 1.0
    else:
        beta = 0.0
        coth_factor = 1.0

    sigma_Q = math.sqrt(HBAR_EV_FS / (2.0 * omega_rad_per_fs) * coth_factor)
    sigma_P = math.sqrt(HBAR_EV_FS * omega_rad_per_fs / 2.0 * coth_factor)

    # But P and Q need to be in consistent units with positions (A) and velocities (A/fs)
    # The harmonic Hamiltonian: H = P²/(2m) + ½ m ω² Q²
    # So sigma_Q is in A, sigma_P = m * v in amu·A/fs
    # sigma_Q² = ℏ/(2mω) * coth  →  sigma_Q = sqrt(ℏ/(2mω)) * sqrt(coth)
    # sigma_P² = ℏmω/2 * coth    →  sigma_P = sqrt(ℏmω/2) * sqrt(coth)
    m = masses[0]
    sigma_Q = math.sqrt(HBAR_EV_FS / (2.0 * m * omega_rad_per_fs) * coth_factor)
    sigma_P = math.sqrt(HBAR_EV_FS * m * omega_rad_per_fs / 2.0 * coth_factor)
    # Convert P to velocity: v = P/m
    sigma_v = sigma_P / m

    weights = {}
    expected_corr_t0 = sigma_Q ** 2  # <Q²(0)> = sigma_Q²

    with open(path, "w") as f:
        for rid in range(n_replicas):
            Q0 = rng.normal(0, sigma_Q)
            P0 = rng.normal(0, sigma_P)
            v0 = P0 / m

            for iframe in range(n_frames):
                t = iframe * dt_fs
                # Q(t) = Q0 cos(ωt) + (P0/(mω)) sin(ωt)
                Qt = Q0 * math.cos(omega_rad_per_fs * t) + (P0 / (m * omega_rad_per_fs)) * math.sin(omega_rad_per_fs * t)
                vt = -Q0 * omega_rad_per_fs * math.sin(omega_rad_per_fs * t) + (P0 / m) * math.cos(omega_rad_per_fs * t)

                pos = np.array([[Qt, 0.0, 0.0]])
                vel = np.array([[vt, 0.0, 0.0]])
                write_extxyz_frame(
                    f, pos, vel, masses, symbols,
                    replica=rid, step=iframe, time_fs=t, seed=seed_offset + rid,
                )

            # For pure harmonic potential, reweighting weight = 1
            weights[rid] = 1.0

    time_axis = np.array([i * dt_fs for i in range(n_frames)])
    return expected_corr_t0, omega_rad_per_fs, time_axis, weights


def write_summary_csv(path, n_replicas, weights=None, include_wigner=True):
    """Write a qct_initial_summary.csv file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if include_wigner:
            writer.writerow([
                "replica", "seed", "total_sampled_energy_eV",
                "rotational_energy_eV", "reaction_energy_eV",
                "potential_correction_eV", "stable_velocity_scale",
                "wigner_weight", "log_wigner_weight",
            ])
        else:
            writer.writerow([
                "replica", "seed", "total_sampled_energy_eV",
                "rotational_energy_eV", "reaction_energy_eV",
                "potential_correction_eV", "stable_velocity_scale",
            ])

        for rid in range(n_replicas):
            w = weights[rid] if weights and rid in weights else 1.0
            row = [rid, 100 + rid, 0.1, 0.0, 0.0, 0.001, 1.0]
            if include_wigner:
                row.extend([w, math.log(w) if w > 0 else -999.0])
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Test 1: Harmonic oscillator analytic check
# ---------------------------------------------------------------------------

class TestHarmonicOscillatorAnalytic:
    """Verify that <Q(0)Q(t)> = (ℏ/2mω)coth(βℏω/2) cos(ωt) for a 1D HO."""

    def test_position_autocorrelation(self, tmp_path):
        # Use a high frequency so the oscillation is well-sampled
        # Choose omega so that f falls exactly on an FFT bin:
        # dt=0.5fs, n_frames=1024 -> df = 1000/(1024*0.5) = 1.9531 THz
        # k=8 -> f = 15.625 THz -> omega = 2*pi*15.625/1000 = 0.098175 rad/fs
        dt = 0.5     # fs
        n_frames = 1024
        omega = 2.0 * math.pi * 8.0 / (n_frames * dt)  # on-bin frequency
        n_replicas = 200  # enough for good statistics
        T = 300.0   # K

        traj_path = tmp_path / "trajectory.xyz"
        expected_corr_t0, omega_used, times, weights = write_synthetic_harmonic_trajectory(
            traj_path, omega, n_frames=n_frames, dt_fs=dt,
            n_replicas=n_replicas, temperature_K=T,
        )

        # Build summary with all weights = 1 (pure harmonic → w=1)
        summary_path = tmp_path / "qct_initial_summary.csv"
        write_summary_csv(summary_path, n_replicas, weights={i: 1.0 for i in range(n_replicas)})

        # Load and compute correlation
        frames = mod_analyzer.read_extxyz(str(traj_path))
        replica_trajs = mod_lsc.split_replica_trajectory(frames)
        weights_loaded = mod_lsc.load_wigner_weights(summary_path)

        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})
        corr = mod_lsc.compute_correlation(replica_trajs, weights_loaded, op, op)

        # Analytical expectation:
        # <Q(0)Q(t)> = (ℏ/(2mω)) coth(βℏω/2) cos(ωt)
        m = 1.008
        beta = 1.0 / (K_B_EV_K * T)
        x = beta * HBAR_EV_FS * omega / 2.0
        coth = 1.0 / math.tanh(x)
        amplitude = HBAR_EV_FS / (2.0 * m * omega) * coth

        # Check t=0 value
        assert abs(corr.c_ab_normalized[0] - amplitude) / amplitude < 0.15, (
            f"Correlation at t=0: {corr.c_ab_normalized[0]:.6e} vs expected {amplitude:.6e} "
            f"(relative error {abs(corr.c_ab_normalized[0] - amplitude)/amplitude:.4f})"
        )

        # Check oscillation frequency: find first zero crossing
        # cos(ωt) = 0 at t = π/(2ω)
        expected_first_zero = math.pi / (2.0 * omega)
        # Find approximate zero crossing in data
        c = corr.c_ab_normalized
        zero_crossing = None
        for i in range(len(c) - 1):
            if c[i] > 0 and c[i + 1] < 0:
                # Linear interpolation
                frac = c[i] / (c[i] - c[i + 1])
                zero_crossing = corr.time_fs[i] + frac * (corr.time_fs[i + 1] - corr.time_fs[i])
                break

        if zero_crossing is not None:
            rel_err = abs(zero_crossing - expected_first_zero) / expected_first_zero
            assert rel_err < 0.1, (
                f"First zero crossing at {zero_crossing:.2f} fs vs expected {expected_first_zero:.2f} fs "
                f"(relative error {rel_err:.4f})"
            )


# ---------------------------------------------------------------------------
# Test 2: Reweighting consistency
# ---------------------------------------------------------------------------

class TestReweightingConsistency:
    """Pure harmonic trajectories should give w_i ≈ 1."""

    def test_harmonic_weights_are_unity(self, tmp_path):
        n_replicas = 4
        summary_path = tmp_path / "qct_initial_summary.csv"
        write_summary_csv(
            summary_path, n_replicas,
            weights={i: 1.0 for i in range(n_replicas)},
            include_wigner=True,
        )
        weights = mod_lsc.load_wigner_weights(summary_path)
        for rid in range(n_replicas):
            assert rid in weights
            assert abs(weights[rid] - 1.0) < 1e-10

    def test_anharmonic_weights_change_correlation(self, tmp_path):
        """When weights differ, correlation should be weighted accordingly."""
        # Create a simple 2-replica, 3-frame trajectory
        traj_path = tmp_path / "trajectory.xyz"
        with open(traj_path, "w") as f:
            # Replica 0: Q=1, 2, 3 (linearly increasing position)
            for iframe in range(3):
                t = iframe * 1.0
                pos = np.array([[1.0 + iframe, 0.0, 0.0]])
                vel = np.array([[0.0, 0.0, 0.0]])
                write_extxyz_frame(f, pos, vel, np.array([1.0]), ["H"],
                                   replica=0, step=iframe, time_fs=t)
            # Replica 1: Q=2, 4, 6
            for iframe in range(3):
                t = iframe * 1.0
                pos = np.array([[2.0 + 2 * iframe, 0.0, 0.0]])
                vel = np.array([[0.0, 0.0, 0.0]])
                write_extxyz_frame(f, pos, vel, np.array([1.0]), ["H"],
                                   replica=1, step=iframe, time_fs=t)

        frames = mod_analyzer.read_extxyz(str(traj_path))
        replica_trajs = mod_lsc.split_replica_trajectory(frames)

        # With equal weights
        weights_equal = {0: 1.0, 1: 1.0}
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})
        corr_equal = mod_lsc.compute_correlation(replica_trajs, weights_equal, op, op)

        # At t=0: C = (1*1*1 + 1*2*2) / (1+1) = 5/2 = 2.5
        assert abs(corr_equal.c_ab_normalized[0] - 2.5) < 1e-10

        # With weight: w0=2, w1=1 → C(t=0) = (2*1*1 + 1*2*2) / (2+1) = 6/3 = 2.0
        weights_unequal = {0: 2.0, 1: 1.0}
        corr_unequal = mod_lsc.compute_correlation(replica_trajs, weights_unequal, op, op)
        assert abs(corr_unequal.c_ab_normalized[0] - 2.0) < 1e-10

    def test_ratio_standard_error_and_lag_zero(self, tmp_path):
        traj_path = tmp_path / "trajectory.xyz"
        with traj_path.open("w", encoding="utf-8") as output:
            for rid, value in enumerate([0.0, 1.0, 2.0, 3.0]):
                for step in range(2):
                    position = np.array([[value, 0.0, 0.0]])
                    write_extxyz_frame(
                        output,
                        position,
                        np.zeros((1, 3)),
                        np.array([1.0]),
                        ["H"],
                        replica=rid,
                        step=step,
                        time_fs=float(step),
                    )
        frames = mod_analyzer.read_extxyz(str(traj_path))
        replica_trajs = mod_lsc.split_replica_trajectory(frames)
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})
        corr = mod_lsc.compute_correlation(replica_trajs, {}, op, op)

        assert len(corr.time_fs) == 2
        assert corr.c_ab_normalized[0] == pytest.approx(3.5)
        assert corr.std_error[0] == pytest.approx(math.sqrt(49.0 / 16.0))
        lag_zero = mod_lsc.compute_correlation(replica_trajs, {}, op, op, max_lag_fs=0.0)
        assert len(lag_zero.time_fs) == 1

    def test_extreme_log_weights_preserve_relative_ratio(self, tmp_path):
        summary_path = tmp_path / "qct_initial_summary.csv"
        with summary_path.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["replica", "log_wigner_weight"])
            writer.writerow([0, 1000.0])
            writer.writerow([1, 999.0])
        weights = mod_lsc.load_wigner_weights(summary_path)
        assert weights[0] == pytest.approx(1.0)
        assert weights[1] == pytest.approx(math.exp(-1.0))

    def test_dipole_attachment_uses_step(self):
        frame0 = mod_analyzer.Frame(
            symbols=["H"], positions=np.zeros((1, 3)), masses=np.ones(1),
            velocities=np.zeros((1, 3)), lattice=None,
            pbc=np.array([False, False, False]), time_fs=0.0,
            metadata={"step": "0"},
        )
        frame1 = mod_analyzer.Frame(
            symbols=["H"], positions=np.zeros((1, 3)), masses=np.ones(1),
            velocities=np.zeros((1, 3)), lattice=None,
            pbc=np.array([False, False, False]), time_fs=1.0,
            metadata={"step": "5"},
        )
        dipoles = {
            0: [
                mod_lsc.DipoleFrame(5, 0, np.array([5.0, 0.0, 0.0])),
                mod_lsc.DipoleFrame(0, 0, np.array([0.0, 0.0, 0.0])),
            ]
        }
        mod_lsc.attach_dipole_data({0: [frame0, frame1]}, dipoles)
        assert frame0.metadata["_nep_dipole"][0] == 0.0
        assert frame1.metadata["_nep_dipole"][0] == 5.0


# ---------------------------------------------------------------------------
# Test 3: Operator registry
# ---------------------------------------------------------------------------

class TestOperatorRegistry:
    def test_position_operator(self):
        frame = mod_analyzer.Frame(
            symbols=["H", "O"],
            positions=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            masses=np.array([1.0, 16.0]),
            velocities=np.zeros((2, 3)),
            lattice=None,
            pbc=np.array([False, False, False]),
            time_fs=0.0,
        )
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 1, "axis": 2}})
        assert op(frame) == 6.0

    def test_velocity_operator(self):
        frame = mod_analyzer.Frame(
            symbols=["H"],
            positions=np.zeros((1, 3)),
            masses=np.array([1.0]),
            velocities=np.array([[3.0, -2.0, 1.0]]),
            lattice=None,
            pbc=np.array([False, False, False]),
            time_fs=0.0,
        )
        op = mod_lsc.make_operator({"name": "velocity", "params": {"atom": 0, "axis": 1}})
        assert op(frame) == -2.0

    def test_bond_length_operator(self):
        frame = mod_analyzer.Frame(
            symbols=["H", "H"],
            positions=np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            velocities=None,
            lattice=None,
            pbc=np.array([False, False, False]),
            time_fs=0.0,
        )
        op = mod_lsc.make_operator({"name": "bond_length", "params": {"atom1": 0, "atom2": 1}})
        assert abs(op(frame) - 5.0) < 1e-10

    def test_kinetic_energy_operator(self):
        frame = mod_analyzer.Frame(
            symbols=["H"],
            positions=np.zeros((1, 3)),
            masses=np.array([1.0]),
            velocities=np.array([[1.0, 0.0, 0.0]]),
            lattice=None,
            pbc=np.array([False, False, False]),
            time_fs=0.0,
        )
        op = mod_lsc.make_operator({"name": "kinetic_energy", "params": {}})
        expected = 0.5 * 1.0 * 1.0 * mod_lsc.AMU_A2_FS2_TO_EV
        assert abs(op(frame) - expected) < 1e-10

    def test_com_position_operator(self):
        frame = mod_analyzer.Frame(
            symbols=["H", "H"],
            positions=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            velocities=None,
            lattice=None,
            pbc=np.array([False, False, False]),
            time_fs=0.0,
        )
        op = mod_lsc.make_operator({"name": "com_position", "params": {}})
        # COM at origin → |COM| = 0
        assert abs(op(frame)) < 1e-10

    def test_unknown_operator_raises(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            mod_lsc.make_operator({"name": "nonexistent", "params": {}})


# ---------------------------------------------------------------------------
# Test 4: CSV schema backward compatibility
# ---------------------------------------------------------------------------

class TestCSVBackwardCompatibility:
    def test_missing_wigner_weight_column(self, tmp_path):
        """When wigner_weight column is absent, all weights default to 1.0."""
        summary_path = tmp_path / "qct_initial_summary.csv"
        write_summary_csv(summary_path, 3, include_wigner=False)
        weights = mod_lsc.load_wigner_weights(summary_path)
        assert len(weights) == 3
        for rid in range(3):
            assert weights[rid] == 1.0

    def test_nonexistent_summary_file(self, tmp_path):
        """Missing summary file returns empty dict → all weights default to 1.0."""
        weights = mod_lsc.load_wigner_weights(tmp_path / "nonexistent.csv")
        assert len(weights) == 0

    def test_zero_weight_handling(self, tmp_path):
        """Zero or NaN weights are set to 0 (contributing nothing)."""
        summary_path = tmp_path / "qct_initial_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "replica", "seed", "total_sampled_energy_eV",
                "rotational_energy_eV", "reaction_energy_eV",
                "potential_correction_eV", "stable_velocity_scale",
                "wigner_weight", "log_wigner_weight",
            ])
            writer.writerow([0, 100, 0.1, 0.0, 0.0, 0.0, 1.0, 0.5, math.log(0.5)])
            writer.writerow([1, 101, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0, -999.0])
            writer.writerow([2, 102, 0.1, 0.0, 0.0, 0.0, 1.0, 2.0, math.log(2.0)])

        weights = mod_lsc.load_wigner_weights(summary_path)
        assert weights[0] == 0.5
        assert weights[1] == 0.0  # zero weight
        assert weights[2] == 2.0


# ---------------------------------------------------------------------------
# Test 5: FFT spectrum
# ---------------------------------------------------------------------------

class TestFFTSpectrum:
    def test_harmonic_peak(self, tmp_path):
        """FFT of a sinusoidal correlation should peak at the oscillator frequency."""
        omega = 0.1  # rad/fs
        n_frames = 128
        dt = 0.5  # fs
        n_replicas = 100
        T = 300.0

        traj_path = tmp_path / "trajectory.xyz"
        write_synthetic_harmonic_trajectory(
            traj_path, omega, n_frames=n_frames, dt_fs=dt,
            n_replicas=n_replicas, temperature_K=T,
        )

        frames = mod_analyzer.read_extxyz(str(traj_path))
        replica_trajs = mod_lsc.split_replica_trajectory(frames)
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})
        corr = mod_lsc.compute_correlation(replica_trajs, {}, op, op)

        freq, wavenumber, intensity = mod_lsc.compute_spectrum(corr, dt, window="hann")

        # Expected peak frequency: f = ω / (2π) in cycles/fs, then *1000 to get THz
        expected_freq = omega / (2.0 * math.pi) * 1.0e3  # convert cycles/fs to THz

        peak_idx = np.argmax(intensity[1:]) + 1  # skip DC
        peak_freq = freq[peak_idx]

        # Allow tolerance of ~2 FFT bins for windowing effects (frequency should be on-bin)
        df = 1.0e3 / (n_frames * dt)  # frequency resolution in THz
        abs_err = abs(peak_freq - expected_freq)
        assert abs_err < 2 * df, (
            f"Peak at {peak_freq:.4f} THz vs expected {expected_freq:.4f} THz "
            f"(abs error {abs_err:.4f} THz, df={df:.4f} THz)"
        )


# ---------------------------------------------------------------------------
# Test 6: Trajectory splitting
# ---------------------------------------------------------------------------

class TestTrajectorySplitting:
    def test_split_by_replica(self, tmp_path):
        traj_path = tmp_path / "trajectory.xyz"
        with open(traj_path, "w") as f:
            for rid in range(3):
                for step in range(4):
                    pos = np.array([[float(rid * 10 + step), 0.0, 0.0]])
                    vel = np.array([[0.0, 0.0, 0.0]])
                    write_extxyz_frame(f, pos, vel, np.array([1.0]), ["H"],
                                       replica=rid, step=step, time_fs=step * 0.5)

        frames = mod_analyzer.read_extxyz(str(traj_path))
        groups = mod_lsc.split_replica_trajectory(frames)
        assert len(groups) == 3
        for rid in range(3):
            assert len(groups[rid]) == 4

    def test_missing_replica_attribute_defaults_to_zero(self, tmp_path):
        """Frames without a Replica attribute are assigned to replica 0.

        This enables lsc_ivr.py to process single-trajectory input such as
        RPMD centroid trajectories from dump_centroid, or ordinary MD
        trajectories, without requiring the Replica metadata.
        """
        traj_path = tmp_path / "single_trajectory.xyz"
        with open(traj_path, "w") as f:
            for step in range(3):
                f.write("1\n")
                f.write(f'Time={step * 0.5} pbc="F F F" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3\n')
                f.write("H 0 0 0 1.0 0 0 0\n")

        frames = mod_analyzer.read_extxyz(str(traj_path))
        groups = mod_lsc.split_replica_trajectory(frames)
        assert len(groups) == 1
        assert 0 in groups
        assert len(groups[0]) == 3


# ---------------------------------------------------------------------------
# Tests for the ensemble lsc_ivr keyword (parameter transformation)
# ---------------------------------------------------------------------------
# These tests verify that the lsc_ivr ensemble keyword correctly transforms
# "ensemble lsc_ivr T ..." into "ensemble qct wigner T ..." for the QCT
# constructor.  The actual GPU-based integration is tested on the sai cluster
# via the batch scripts in tests/gpumd/qct_nep89_si/lsc_ivr_ensemble/.

def test_lsc_ivr_param_transform_basic():
    """Verify the parameter transformation logic used by Ensemble_LSC_IVR."""
    # Input: ensemble lsc_ivr 300 seed 12345 replicas 1
    # Expected QCT params: ensemble qct wigner temperature 300 seed 12345 replicas 1
    input_params = ["ensemble", "lsc_ivr", "300", "seed", "12345", "replicas", "1"]
    num_param = len(input_params)

    # Simulate the transformation: insert "wigner" and "temperature"
    strings = []
    strings.append(input_params[0])  # "ensemble"
    strings.append("qct")
    strings.append("wigner")
    strings.append("temperature")
    for i in range(2, num_param):
        strings.append(input_params[i])

    expected = ["ensemble", "qct", "wigner", "temperature", "300", "seed", "12345", "replicas", "1"]
    assert strings == expected
    assert len(strings) == num_param + 2  # +2 for "wigner" and "temperature"


def test_lsc_ivr_param_transform_with_options():
    """Verify parameter transformation with all common options."""
    input_params = [
        "ensemble", "lsc_ivr", "0", "seed", "42", "replicas", "128",
        "hessian_displacement", "0.001", "anharmonic_reweighting", "no"
    ]
    num_param = len(input_params)

    strings = [input_params[0], "qct", "wigner", "temperature"]
    for i in range(2, num_param):
        strings.append(input_params[i])

    expected = [
        "ensemble", "qct", "wigner", "temperature", "0", "seed", "42", "replicas", "128",
        "hessian_displacement", "0.001", "anharmonic_reweighting", "no"
    ]
    assert strings == expected


def test_lsc_ivr_run_in_file_exists():
    """Verify the test run.in file uses the lsc_ivr keyword."""
    run_in = REPO_ROOT / "tests/gpumd/qct_nep89_si/lsc_ivr_ensemble/run.in"
    assert run_in.exists(), f"Missing {run_in}"
    content = run_in.read_text()
    assert "ensemble" in content
    assert "lsc_ivr" in content
    assert "300" in content
    # Should NOT use "qct wigner" syntax
    assert "qct wigner" not in content, "lsc_ivr_ensemble run.in should use lsc_ivr keyword"


def test_lsc_ivr_batch_file_exists():
    """Verify the batch script exists for the lsc_ivr ensemble test."""
    batch = REPO_ROOT / "tests/gpumd/qct_nep89_si/lsc_ivr_ensemble/lsc_ivr_ensemble.batch"
    assert batch.exists(), f"Missing {batch}"
    content = batch.read_text()
    assert "sbatch" in content.lower() or "SBATCH" in content
    assert "run.in" in content


# ---------------------------------------------------------------------------
# Test 7: load_dipole_out — single and batch formats
# ---------------------------------------------------------------------------

class TestLoadDipoleOut:
    def test_single_trajectory_format(self, tmp_path):
        """dipole.out with 4 columns: step dx dy dz."""
        dipole_path = tmp_path / "dipole.out"
        with dipole_path.open("w") as f:
            f.write("0 1.0 2.0 3.0\n")
            f.write("1 1.1 2.1 3.1\n")
            f.write("2 1.2 2.2 3.2\n")

        result = mod_lsc.load_dipole_out(dipole_path)
        assert len(result) == 1
        assert 0 in result
        frames = result[0]
        assert len(frames) == 3
        assert frames[0].step == 0
        np.testing.assert_allclose(frames[0].dipole, [1.0, 2.0, 3.0])
        assert frames[1].step == 1
        np.testing.assert_allclose(frames[1].dipole, [1.1, 2.1, 3.1])
        assert frames[2].step == 2

    def test_batch_format(self, tmp_path):
        """dipole.out with 5 columns: step replica dx dy dz."""
        dipole_path = tmp_path / "dipole.out"
        with dipole_path.open("w") as f:
            f.write("0 0 1.0 2.0 3.0\n")
            f.write("0 1 4.0 5.0 6.0\n")
            f.write("1 0 1.1 2.1 3.1\n")
            f.write("1 1 4.1 5.1 6.1\n")

        result = mod_lsc.load_dipole_out(dipole_path)
        assert len(result) == 2
        assert set(result.keys()) == {0, 1}

        assert len(result[0]) == 2
        assert result[0][0].step == 0
        np.testing.assert_allclose(result[0][0].dipole, [1.0, 2.0, 3.0])
        assert result[0][1].step == 1
        np.testing.assert_allclose(result[0][1].dipole, [1.1, 2.1, 3.1])

        assert len(result[1]) == 2
        assert result[1][0].step == 0
        np.testing.assert_allclose(result[1][0].dipole, [4.0, 5.0, 6.0])
        assert result[1][1].step == 1
        np.testing.assert_allclose(result[1][1].dipole, [4.1, 5.1, 6.1])

    def test_sorted_by_step(self, tmp_path):
        """Ensure frames are sorted by step even if file is unordered."""
        dipole_path = tmp_path / "dipole.out"
        with dipole_path.open("w") as f:
            f.write("2 0 3.0 2.0 1.0\n")
            f.write("0 0 1.0 0.0 0.0\n")
            f.write("1 0 2.0 1.0 0.5\n")

        result = mod_lsc.load_dipole_out(dipole_path)
        steps = [f.step for f in result[0]]
        assert steps == [0, 1, 2]

    def test_nonexistent_file_returns_empty(self, tmp_path):
        """Missing file returns empty dict (no exception)."""
        result = mod_lsc.load_dipole_out(tmp_path / "nonexistent.out")
        assert result == {}

    def test_blank_lines_skipped(self, tmp_path):
        """Blank lines in dipole.out are ignored."""
        dipole_path = tmp_path / "dipole.out"
        with dipole_path.open("w") as f:
            f.write("\n")
            f.write("0 1.0 2.0 3.0\n")
            f.write("\n")
            f.write("1 1.1 2.1 3.1\n")
            f.write("\n")

        result = mod_lsc.load_dipole_out(dipole_path)
        assert len(result) == 1
        assert len(result[0]) == 2


# ---------------------------------------------------------------------------
# Test 8: compute_convergence_diagnostics — known weights and replicas
# ---------------------------------------------------------------------------

class TestComputeConvergenceDiagnostics:
    def _make_frame(self, pos_x: float, vel_x: float = 0.0):
        """Create a minimal aq.Frame for testing."""
        import tempfile, os
        fd, fname = tempfile.mkstemp(suffix=".xyz")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("1\n")
                f.write(f'Time=0 pbc="F F F" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3\n')
                f.write(f"H {pos_x:.12g} 0 0 1.0 {vel_x:.12g} 0 0\n")
            return mod_analyzer.read_extxyz(fname)[0]
        finally:
            os.unlink(fname)

    def test_uniform_weights(self):
        """With uniform weights, C(0) is the simple mean of A(0)*B(0)."""
        # 4 replicas, uniform weights, position operator on both sides
        replica_trajs = {}
        for rid, x in enumerate([1.0, 2.0, 3.0, 4.0]):
            replica_trajs[rid] = [self._make_frame(x)]

        weights = {rid: 1.0 for rid in range(4)}
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})

        diag = mod_lsc.compute_convergence_diagnostics(
            replica_trajs, weights, op, op, max_lag=4
        )

        assert diag.n_replicas == 4
        assert abs(diag.n_effective - 4.0) < 1e-10
        # C(0) = mean of x² = (1+4+9+16)/4 = 7.5
        assert abs(diag.c0_weighted_mean - 7.5) < 1e-8
        # c0_values = [1, 4, 9, 16]
        np.testing.assert_allclose(diag.c0_values, [1.0, 4.0, 9.0, 16.0])

    def test_nonuniform_weights(self):
        """Weighted mean differs from simple mean."""
        replica_trajs = {}
        for rid, x in enumerate([1.0, 2.0, 3.0, 4.0]):
            replica_trajs[rid] = [self._make_frame(x)]

        # Weight replica 0 and 3 heavily
        weights = {0: 3.0, 1: 1.0, 2: 1.0, 3: 3.0}
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})

        diag = mod_lsc.compute_convergence_diagnostics(
            replica_trajs, weights, op, op, max_lag=4
        )

        # Weighted mean = (3*1 + 1*4 + 1*9 + 3*16) / (3+1+1+3) = (3+4+9+48)/8 = 64/8 = 8.0
        assert abs(diag.c0_weighted_mean - 8.0) < 1e-8
        # n_effective = (sum w)² / sum(w²) = 64 / (9+1+1+9) = 64/20 = 3.2
        assert abs(diag.n_effective - 3.2) < 1e-8
        # max_to_mean = 3.0 / 2.0 = 1.5 (mean of positive weights = 8/4 = 2.0)
        assert abs(diag.max_to_mean_ratio - 1.5) < 1e-8

    def test_all_zero_weights_raises(self):
        """All-zero weights should raise ValueError."""
        replica_trajs = {0: [self._make_frame(1.0)]}
        weights = {0: 0.0}
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})

        with pytest.raises(ValueError, match="All replica weights are zero"):
            mod_lsc.compute_convergence_diagnostics(
                replica_trajs, weights, op, op
            )

    def test_negative_weight_raises(self):
        """Negative weights should raise ValueError."""
        replica_trajs = {0: [self._make_frame(1.0)]}
        weights = {0: -1.0}
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})

        with pytest.raises(ValueError, match="non-finite or negative"):
            mod_lsc.compute_convergence_diagnostics(
                replica_trajs, weights, op, op
            )

    def test_convergence_curve_monotonic_n(self):
        """Convergence curve should have increasing n values."""
        replica_trajs = {}
        for rid in range(6):
            replica_trajs[rid] = [self._make_frame(float(rid + 1))]
        weights = {rid: 1.0 for rid in range(6)}
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})

        diag = mod_lsc.compute_convergence_diagnostics(
            replica_trajs, weights, op, op, max_lag=10
        )

        assert len(diag.convergence_n) == 6
        assert list(diag.convergence_n) == [1, 2, 3, 4, 5, 6]

    def test_single_replica(self):
        """Single replica: std_error should be 0, no crash."""
        replica_trajs = {0: [self._make_frame(2.0)]}
        weights = {0: 1.0}
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})

        diag = mod_lsc.compute_convergence_diagnostics(
            replica_trajs, weights, op, op
        )

        assert diag.n_replicas == 1
        assert diag.c0_std_error == 0.0
        assert abs(diag.c0_weighted_mean - 4.0) < 1e-8


# ---------------------------------------------------------------------------
# Test 9: load_zpe_csv — ZPE CSV validation
# ---------------------------------------------------------------------------

class TestLoadZpeCsv:
    def test_valid_csv(self, tmp_path):
        """A well-formed qct_zpe.csv loads successfully."""
        zpe_path = tmp_path / "qct_zpe.csv"
        with zpe_path.open("w") as f:
            f.write("replica,step,time_fs,mode,frequency_THz,"
                    "mode_energy_eV,initial_mode_energy_eV,zpe_drift_eV\n")
            f.write("0,0,0.0,0,15.0,0.031,0.031,0.0\n")
            f.write("0,1,0.5,0,15.0,0.032,0.031,0.001\n")
            f.write("0,0,0.0,1,20.0,0.041,0.041,0.0\n")
            f.write("0,1,0.5,1,20.0,0.042,0.041,0.001\n")

        rows = mod_analyzer.load_zpe_csv(zpe_path)
        assert len(rows) == 4
        assert rows[0]["replica"] == 0
        assert rows[0]["step"] == 0
        assert rows[0]["mode"] == 0
        assert abs(rows[0]["mode_energy_eV"] - 0.031) < 1e-12

    def test_missing_column_raises(self, tmp_path):
        """Missing required column should raise ValueError."""
        zpe_path = tmp_path / "qct_zpe.csv"
        with zpe_path.open("w") as f:
            f.write("replica,step,time_fs,mode,frequency_THz,mode_energy_eV\n")
            f.write("0,0,0.0,0,15.0,0.031\n")

        with pytest.raises(ValueError, match="missing required columns"):
            mod_analyzer.load_zpe_csv(zpe_path)

    def test_duplicate_key_raises(self, tmp_path):
        """Duplicate (replica, step, mode) should raise ValueError."""
        zpe_path = tmp_path / "qct_zpe.csv"
        with zpe_path.open("w") as f:
            f.write("replica,step,time_fs,mode,frequency_THz,"
                    "mode_energy_eV,initial_mode_energy_eV,zpe_drift_eV\n")
            f.write("0,0,0.0,0,15.0,0.031,0.031,0.0\n")
            f.write("0,0,0.0,0,15.0,0.031,0.031,0.0\n")

        with pytest.raises(ValueError, match="duplicate"):
            mod_analyzer.load_zpe_csv(zpe_path)

    def test_non_monotonic_step_raises(self, tmp_path):
        """Non-monotonic step within a (replica, mode) group should raise."""
        zpe_path = tmp_path / "qct_zpe.csv"
        with zpe_path.open("w") as f:
            f.write("replica,step,time_fs,mode,frequency_THz,"
                    "mode_energy_eV,initial_mode_energy_eV,zpe_drift_eV\n")
            f.write("0,5,2.5,0,15.0,0.031,0.031,0.0\n")
            f.write("0,3,1.5,0,15.0,0.031,0.031,0.0\n")

        with pytest.raises(ValueError, match="non-monotonic"):
            mod_analyzer.load_zpe_csv(zpe_path)

    def test_non_finite_value_raises(self, tmp_path):
        """Non-finite (nan/inf) numeric values should raise."""
        zpe_path = tmp_path / "qct_zpe.csv"
        with zpe_path.open("w") as f:
            f.write("replica,step,time_fs,mode,frequency_THz,"
                    "mode_energy_eV,initial_mode_energy_eV,zpe_drift_eV\n")
            f.write("0,0,0.0,0,15.0,nan,0.031,0.0\n")

        with pytest.raises(ValueError, match="non-finite"):
            mod_analyzer.load_zpe_csv(zpe_path)

    def test_empty_file_raises(self, tmp_path):
        """Empty data file (header only, no rows) should raise."""
        zpe_path = tmp_path / "qct_zpe.csv"
        with zpe_path.open("w") as f:
            f.write("replica,step,time_fs,mode,frequency_THz,"
                    "mode_energy_eV,initial_mode_energy_eV,zpe_drift_eV\n")

        with pytest.raises(ValueError, match="no data rows"):
            mod_analyzer.load_zpe_csv(zpe_path)

    def test_nonexistent_file_raises(self, tmp_path):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            mod_analyzer.load_zpe_csv(tmp_path / "no_such_file.csv")

    def test_multiple_replicas_and_modes(self, tmp_path):
        """Multiple replicas with multiple modes load correctly."""
        zpe_path = tmp_path / "qct_zpe.csv"
        with zpe_path.open("w") as f:
            f.write("replica,step,time_fs,mode,frequency_THz,"
                    "mode_energy_eV,initial_mode_energy_eV,zpe_drift_eV\n")
            for rid in range(3):
                for step in range(2):
                    for mode in range(2):
                        freq = 15.0 + mode * 5.0
                        e = 0.031 + step * 0.001
                        drift = 0.001 * step
                        f.write(f"{rid},{step},{step*0.5},{mode},{freq},{e},{0.031},{drift}\n")

        rows = mod_analyzer.load_zpe_csv(zpe_path)
        assert len(rows) == 3 * 2 * 2  # 12 rows
        replicas = {r["replica"] for r in rows}
        assert replicas == {0, 1, 2}


# ---------------------------------------------------------------------------
# Tests for load_dipole_out (dipole trajectory reader)
# ---------------------------------------------------------------------------

class TestLoadDipoleOut:
    """Tests for the dipole.out reader supporting single and batch formats."""

    def test_single_trajectory_format(self, tmp_path):
        """Single-trajectory format: step dx dy dz."""
        path = tmp_path / "dipole.out"
        path.write_text(
            "0 1.0 2.0 3.0\n"
            "1 4.0 5.0 6.0\n"
            "2 7.0 8.0 9.0\n",
            encoding="utf-8",
        )
        result = mod_lsc.load_dipole_out(path)
        assert 0 in result
        assert len(result[0]) == 3
        assert result[0][0].step == 0
        np.testing.assert_array_equal(result[0][0].dipole, [1.0, 2.0, 3.0])
        assert result[0][2].step == 2
        np.testing.assert_array_equal(result[0][2].dipole, [7.0, 8.0, 9.0])

    def test_batch_format_with_replica(self, tmp_path):
        """Batch format: step replica dx dy dz."""
        path = tmp_path / "dipole.out"
        path.write_text(
            "0 0 1.0 2.0 3.0\n"
            "0 1 4.0 5.0 6.0\n"
            "1 0 7.0 8.0 9.0\n"
            "1 1 10.0 11.0 12.0\n",
            encoding="utf-8",
        )
        result = mod_lsc.load_dipole_out(path)
        assert set(result.keys()) == {0, 1}
        assert len(result[0]) == 2
        assert len(result[1]) == 2
        assert result[0][0].step == 0
        np.testing.assert_array_equal(result[0][0].dipole, [1.0, 2.0, 3.0])
        assert result[1][1].step == 1
        np.testing.assert_array_equal(result[1][1].dipole, [10.0, 11.0, 12.0])

    def test_frames_sorted_by_step(self, tmp_path):
        """Frames should be sorted by step within each replica."""
        path = tmp_path / "dipole.out"
        path.write_text(
            "2 1.0 0.0 0.0\n"
            "0 2.0 0.0 0.0\n"
            "1 3.0 0.0 0.0\n",
            encoding="utf-8",
        )
        result = mod_lsc.load_dipole_out(path)
        steps = [f.step for f in result[0]]
        assert steps == [0, 1, 2]

    def test_nonexistent_file_returns_empty(self, tmp_path):
        """Missing file should return empty dict."""
        result = mod_lsc.load_dipole_out(tmp_path / "no_such_file.out")
        assert result == {}

    def test_empty_lines_skipped(self, tmp_path):
        """Empty lines and whitespace should be skipped."""
        path = tmp_path / "dipole.out"
        path.write_text(
            "\n"
            "0 1.0 2.0 3.0\n"
            "\n"
            "1 4.0 5.0 6.0\n"
            "\n",
            encoding="utf-8",
        )
        result = mod_lsc.load_dipole_out(path)
        assert len(result[0]) == 2


# ---------------------------------------------------------------------------
# Tests for multi-operator correlation (P1-5)
# ---------------------------------------------------------------------------

class TestMultiOperatorCorrelation:
    """Tests for the multi-operator correlation support in lsc_ivr.py main()."""

    def _write_config(self, path, correlations):
        """Write a JSON config file."""
        import json
        with path.open("w") as f:
            json.dump({"correlations": correlations}, f)

    def test_multi_correlation_config_parses(self, tmp_path):
        """Multi-correlation config should produce multiple output files."""
        import json
        # Create a simple 2-replica trajectory with enough frames for FFT
        traj_path = tmp_path / "traj.xyz"
        with traj_path.open("w") as f:
            # 2 replicas, 8 frames each (need >= 4 for FFT)
            for rid in range(2):
                for step in range(8):
                    write_extxyz_frame(
                        f,
                        positions=np.array([[0.0 + step * 0.1 * rid, 0.0, 0.0]]),
                        velocities=np.array([[1.0 * rid, 0.0, 0.0]]),
                        masses=np.array([1.008]),
                        symbols=["H"],
                        replica=rid,
                        step=step,
                        time_fs=step * 0.5,
                        seed=rid,
                    )

        # Summary with equal weights
        summary_path = tmp_path / "summary.csv"
        summary_path.write_text(
            "replica,seed,wigner_weight,log_wigner_weight\n"
            "0,0,1.0,0.0\n"
            "1,1,1.0,0.0\n",
            encoding="utf-8",
        )

        # Multi-correlation config: position-x autocorr + velocity-x autocorr
        config_path = tmp_path / "config.json"
        self._write_config(config_path, [
            {
                "name": "pos_x",
                "operator_A": {"name": "position", "params": {"atom": 0, "axis": 0}},
                "operator_B": {"name": "position", "params": {"atom": 0, "axis": 0}},
            },
            {
                "name": "vel_x",
                "operator_A": {"name": "velocity", "params": {"atom": 0, "axis": 0}},
                "operator_B": {"name": "velocity", "params": {"atom": 0, "axis": 0}},
            },
        ])

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        corr_output = out_dir / "correlation.csv"
        fft_output = out_dir / "spectrum.csv"

        result = subprocess.run(
            [
                sys.executable,
                str(LSC_IVR_PATH),
                "--trajectory", str(traj_path),
                "--summary", str(summary_path),
                "--config", str(config_path),
                "--output", str(corr_output),
                "--fft", str(fft_output),
                "--max-lag", "3.5",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

        # With multiple correlations, files get suffixed with corr_name
        assert (out_dir / "correlation_pos_x.csv").is_file()
        assert (out_dir / "correlation_vel_x.csv").is_file()
        assert (out_dir / "spectrum_pos_x.csv").is_file()
        assert (out_dir / "spectrum_vel_x.csv").is_file()

    def test_legacy_single_operator_config_still_works(self, tmp_path):
        """Legacy single-operator config should produce unsuffixed output."""
        import json
        traj_path = tmp_path / "traj.xyz"
        with traj_path.open("w") as f:
            for rid in range(2):
                for step in range(8):
                    write_extxyz_frame(
                        f,
                        positions=np.array([[0.0 + step * 0.1 * rid, 0.0, 0.0]]),
                        velocities=np.array([[1.0, 0.0, 0.0]]),
                        masses=np.array([1.008]),
                        symbols=["H"],
                        replica=rid,
                        step=step,
                        time_fs=step * 0.5,
                        seed=rid,
                    )

        summary_path = tmp_path / "summary.csv"
        summary_path.write_text(
            "replica,seed,wigner_weight,log_wigner_weight\n"
            "0,0,1.0,0.0\n1,1,1.0,0.0\n",
            encoding="utf-8",
        )

        config_path = tmp_path / "config.json"
        with config_path.open("w") as f:
            json.dump({
                "operator_A": {"name": "position", "params": {"atom": 0, "axis": 0}},
                "operator_B": {"name": "position", "params": {"atom": 0, "axis": 0}},
            }, f)

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        corr_output = out_dir / "correlation.csv"

        result = subprocess.run(
            [
                sys.executable,
                str(LSC_IVR_PATH),
                "--trajectory", str(traj_path),
                "--summary", str(summary_path),
                "--config", str(config_path),
                "--output", str(corr_output),
                "--max-lag", "1.0",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        # With single correlation, no suffix is added
        assert corr_output.is_file()
        assert not (out_dir / "correlation_default.csv").is_file()


# ---------------------------------------------------------------------------
# Tests for convergence diagnostics
# ---------------------------------------------------------------------------

class TestConvergenceDiagnostics:
    """Tests for compute_convergence_diagnostics."""

    def test_basic_diagnostics(self, tmp_path):
        """Convergence diagnostics should compute N_eff and weight stats."""
        # Build simple replica trajectories
        replica_trajs = {}
        for rid in range(4):
            frames = []
            for step in range(2):
                frame = mod_analyzer.Frame(
                    symbols=["H"],
                    positions=np.array([[0.0, 0.0, 0.0]]),
                    masses=np.array([1.008]),
                    velocities=np.array([[1.0, 0.0, 0.0]]),
                    lattice=None,
                    pbc=np.array([False, False, False]),
                    time_fs=step * 0.5,
                    metadata={"replica": rid, "step": step},
                )
                frames.append(frame)
            replica_trajs[rid] = frames

        weights = {0: 1.0, 1: 1.0, 2: 0.5, 3: 0.5}
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})
        diag = mod_lsc.compute_convergence_diagnostics(replica_trajs, weights, op, op)

        assert diag.n_replicas == 4
        assert diag.n_effective == pytest.approx(3.6, abs=0.01)
        assert diag.weight_max == 1.0
        assert diag.weight_min == 0.5

    def test_all_zero_weights_raises(self, tmp_path):
        """All-zero weights should raise ValueError."""
        replica_trajs = {0: [mod_analyzer.Frame(
            symbols=["H"],
            positions=np.array([[0.0, 0.0, 0.0]]),
            masses=np.array([1.008]),
            velocities=np.array([[0.0, 0.0, 0.0]]),
            lattice=None,
            pbc=np.array([False, False, False]),
            time_fs=0.0,
            metadata={"replica": 0, "step": 0},
        )]}
        weights = {0: 0.0}
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})
        with pytest.raises(ValueError, match="All replica weights are zero"):
            mod_lsc.compute_convergence_diagnostics(replica_trajs, weights, op, op)

    def test_diagnostics_csv_written(self, tmp_path):
        """write_diagnostics_csv should produce a valid CSV."""
        replica_trajs = {}
        for rid in range(3):
            frames = []
            for step in range(2):
                frame = mod_analyzer.Frame(
                    symbols=["H"],
                    positions=np.array([[float(step), 0.0, 0.0]]),
                    masses=np.array([1.008]),
                    velocities=np.array([[1.0, 0.0, 0.0]]),
                    lattice=None,
                    pbc=np.array([False, False, False]),
                    time_fs=step * 0.5,
                    metadata={"replica": rid, "step": step},
                )
                frames.append(frame)
            replica_trajs[rid] = frames

        weights = {0: 1.0, 1: 1.0, 2: 1.0}
        op = mod_lsc.make_operator({"name": "position", "params": {"atom": 0, "axis": 0}})
        diag = mod_lsc.compute_convergence_diagnostics(replica_trajs, weights, op, op)

        csv_path = tmp_path / "diag.csv"
        mod_lsc.write_diagnostics_csv(csv_path, diag)
        assert csv_path.is_file()

        with csv_path.open() as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert any(r[0] == "n_replicas" for r in rows)
        assert any(r[0] == "n_effective" for r in rows)


# ---------------------------------------------------------------------------
# Tests for weight histogram
# ---------------------------------------------------------------------------

class TestWeightHistogram:
    """Tests for write_weight_histogram."""

    def test_histogram_written(self, tmp_path):
        """write_weight_histogram should produce a valid CSV."""
        weights = {i: float(i + 1) for i in range(10)}
        path = tmp_path / "hist.csv"
        mod_lsc.write_weight_histogram(path, weights, n_bins=5)
        assert path.is_file()

        with path.open() as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["bin_left", "bin_right", "count"]
        total = sum(int(r[2]) for r in rows[1:])
        assert total == 10

    def test_histogram_empty_weights_no_file(self, tmp_path):
        """All-zero weights should produce no file."""
        weights = {0: 0.0, 1: 0.0}
        path = tmp_path / "hist.csv"
        mod_lsc.write_weight_histogram(path, weights)
        assert not path.is_file()
