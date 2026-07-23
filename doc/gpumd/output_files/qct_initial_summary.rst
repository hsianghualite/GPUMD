.. _qct_initial_summary:

qct_initial_summary.csv
=======================

The ``qct_initial_summary.csv`` file is written by single- and multi-replica
harmonic QCT initialization. It contains one row per replica with these columns:

* ``replica``: zero-based replica index.
* ``seed``: seed of the accepted phase-point sampling attempt.
* ``total_sampled_energy_eV``: requested sampled energy relative to the
  stationary reference geometry.
* ``rotational_energy_eV`` and ``reaction_energy_eV``: sampled rotational and
  reaction-coordinate contributions.
* ``potential_correction_eV``: real-potential energy change from the reference
  geometry to the sampled geometry.
* ``stable_velocity_scale``: factor applied only to the correctable
  vibrational velocity after the real-potential correction. Reaction-coordinate
  and semiclassical rotational velocities are not scaled.

This file can be used to audit rejected/resampled phase points and verify the
initial per-replica energy balance.
