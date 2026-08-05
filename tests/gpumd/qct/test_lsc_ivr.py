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

    def test_missing_replica_attribute_raises(self, tmp_path):
        traj_path = tmp_path / "bad_trajectory.xyz"
        with open(traj_path, "w") as f:
            f.write("1\n")
            f.write('Time=0.0 pbc="F F F" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3\n')
            f.write("H 0 0 0 1.0 0 0 0\n")

        frames = mod_analyzer.read_extxyz(str(traj_path))
        with pytest.raises(ValueError, match="Replica"):
            mod_lsc.split_replica_trajectory(frames)
