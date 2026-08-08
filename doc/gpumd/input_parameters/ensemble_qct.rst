.. _kw_ensemble_qct:

:attr:`ensemble` (QCT)
======================

The :attr:`qct` variant of the :attr:`ensemble` keyword is the native entry
point for quasi-classical trajectory (:term:`QCT`) calculations.

Syntax
------

Run a QCT trajectory from the phase point supplied in the input structure::

    ensemble qct

This is equivalent to::

    ensemble qct phase_point

Generate a harmonic QCT initial condition from the current ``model.xyz``
structure and then propagate it with NVE dynamics::

    ensemble qct harmonic temperature 300 hessian_displacement 0.001 min_frequency 0.001 seed 12345 zpe yes phase random replicas 1

The explicit sampling-mode names are also accepted::

    ensemble qct canonical temperature 300 seed 12345
    ensemble qct microcanonical energy 2.0 seed 12345
    ensemble qct mode_energy mode 7 energy 0.35 seed 12345
    ensemble qct semiclassical v 2 J 5 seed 12345

Run an LSC-IVR calculation with Wigner sampling and anharmonic reweighting::

    ensemble qct wigner temperature 300 seed 12345 replicas 64 hessian_displacement 0.001 anharmonic_reweighting yes

Ground-state Wigner (T = 0) without reweighting::

    ensemble qct wigner temperature 0 seed 42 replicas 128 anharmonic_reweighting no

For a first-order transition-state structure, the same interface can launch a
trajectory from the dividing surface::

    ensemble qct harmonic temperature 300 stationary_point saddle reaction_direction random seed 12345 zpe yes

For reproducibility or debugging, GPUMD can also consume a precomputed binary
normal-mode output directly::

    ensemble qct harmonic eigenvector eigenvector.out exclude_lowest 6 min_frequency 0.001 temperature 300 seed 12345 zpe yes phase random replicas 1

Parameters
----------

``phase_point``
  Use the positions and velocities already present in ``model.xyz``. The
  ``Properties`` field must contain ``vel:R:3``; GPUMD rejects this mode when
  velocities are absent instead of using its default random 300 K velocities.
  No native QCT initial-condition sampling is performed.

``harmonic``
  Backward-compatible alias for ``canonical``. Generate a QCT phase point from
  harmonic normal modes. Without ``modes`` or ``eigenvector``, GPUMD calculates
  a molecular finite-difference Hessian from the current structure in
  ``model.xyz`` before sampling.

``canonical``
  Sample each stable mode with a Boltzmann energy, optionally including its
  zero-point energy. This is the default sampling mode for ``harmonic``.

``microcanonical``
  Set the total sampled stable-mode energy with ``energy E``. The excess above
  stable-mode zero-point energy is distributed over active modes with the
  harmonic microcanonical phase-space measure. For a saddle, also provide
  ``reaction_energy`` explicitly.

``mode_energy``
  Set one active stable mode with ``mode INDEX energy E`` and assign the other
  stable modes their zero-point energies. For a saddle, also provide
  ``reaction_energy`` explicitly.

``semiclassical``
  Minimum-only EBK sampling for a diatomic molecule. ``v`` sets the vibrational
  quantum number and ``J`` sets the rotational quantum number. The rotational
  angular momentum is ``sqrt(J(J+1)) hbar`` and is placed perpendicular to the
  sampled bond with a deterministic orientation. Real-potential energy
  correction scales only the vibrational velocity, so the requested angular
  momentum is preserved.

``wigner``
  Linearized Semiclassical Initial Value Representation (LSC-IVR) sampling.
  Each active normal mode ``(Q_k, P_k)`` is drawn from the harmonic Wigner
  thermal distribution with quantum-corrected variances::

      sigma_Q^2 = (hbar / 2 omega_k) * coth(beta * hbar * omega_k / 2)
      sigma_P^2 = (hbar * omega_k / 2) * coth(beta * hbar * omega_k / 2)

  At high temperature this reduces to the classical Boltzmann result; at
  ``T = 0`` it gives the ground-state Wigner distribution.  ``temperature``
  is required (use ``0`` for ground state).  Anharmonic reweighting weights
  ``w_i = exp(-beta * Delta_V)`` are computed when
  ``anharmonic_reweighting yes`` is set (the default).  The ``zpe`` keyword
  must be ``yes`` (the default) because zero-point energy is intrinsic to the
  Wigner distribution; ``zpe no`` is rejected.  ``stationary_point saddle``
  is also rejected; use ``minimum`` or ``auto``.  If the auto Hessian detects
  a strongly imaginary mode (frequency below ``-min_frequency``), Wigner
  sampling is aborted because the structure is not a minimum.

  See :ref:`lsc_ivr` for the full LSC-IVR user guide.

``anharmonic_reweighting``
  ``yes`` (default) or ``no``.  Only relevant for ``wigner`` sampling.  When
  enabled, each replica receives a weight::

      w_i = exp(-beta * [V_real(Q_i) - V_ref - sum_k 0.5 * omega_k^2 * Q_k^2])

  where ``V_real`` is the true potential at the sampled geometry and the sum
  is the harmonic potential relative to the reference minimum.  Both
  ``wigner_weight`` and ``log_wigner_weight`` are written to
  ``qct_initial_summary.csv``.  Set to ``no`` to disable reweighting (all
  weights are 1.0), which is appropriate for ground-state (``T = 0``)
  calculations where ``beta -> infinity`` makes the weight ill-defined.

``modes``
  Path to the QCT normal-mode input file. The current implementation expects
  the text format documented in :ref:`qct_modes_in`. This input method is
  mutually exclusive with ``eigenvector``. Cartesian positions still come
  from the current ``model.xyz``.

``eigenvector``
  Path to the binary ``eigenvector.out`` produced by GPUMD's
  :attr:`compute_phonon` keyword. The current ``model.xyz`` structure supplies
  the reference Cartesian positions, masses, and atom ordering.

``hessian_displacement``
  Central finite-difference displacement in Angstrom for automatic molecular
  Hessian calculation. The default is ``0.001``. This keyword cannot be used
  with an external ``eigenvector`` or ``modes`` source.

``exclude_lowest``
  Number of modes with the smallest absolute frequencies to exclude from
  sampling. Their original mode indices are preserved. The default is ``6``,
  appropriate for the three translations and three rotations of a non-linear
  isolated molecule. Use ``5`` for a linear molecule. It only applies to an
  external ``eigenvector`` source; automatic Hessian sampling projects rigid
  translations and rotations explicitly.

``min_frequency``
  Minimum positive frequency for an active mode, in ordinary GPUMD THz. The
  default is ``0.001``. An eigenvalue below ``-min_frequency`` is considered a
  significant imaginary vibrational mode.

``stationary_point``
  ``auto`` (default), ``minimum``, or ``saddle``. ``auto`` accepts a minimum
  with zero significant imaginary modes or a first-order saddle with exactly
  one. ``minimum`` rejects any significant imaginary mode. ``saddle`` requires
  exactly one and uses it as the reaction coordinate.

``reaction_direction``
  ``positive``, ``negative``, or ``random`` for a first-order saddle. The
  reaction eigenvector is given a deterministic sign convention before this
  option is applied: the component with largest absolute value is positive.
  The default is ``random``.

``reaction_energy``
  Optional positive reaction-coordinate energy in eV for a saddle launch.
  If omitted, canonical saddle sampling draws it from the flux distribution
  ``E_rxn = -k_B T ln(u)``. The reaction coordinate starts at ``Q_rxn = 0``;
  recrossing during subsequent NVE propagation is allowed.

``stationary_force_tolerance``
  Maximum Cartesian force in eV/A allowed for automatic Hessian sampling. The
  default is ``0.001``.

``temperature``
  Sampling temperature in K for ``canonical``. For each active mode, the
  implementation samples a classical Boltzmann mode energy and optionally adds
  zero-point energy. It is not required by the other explicit sampling modes.

``energy``
  Total stable-mode energy in eV for ``microcanonical``. For ``mode_energy``,
  this is the energy of the mode selected by ``mode``.

``mode``
  Zero-based active stable-mode index used by ``mode_energy``.

``v``
  Non-negative vibrational quantum number used by ``semiclassical``.

``J``
  Non-negative rotational quantum number used by ``semiclassical``.

``seed``
  Non-negative integer random seed. The default is ``1``.

``zpe``
  ``yes`` or ``no``. If ``yes``, add ``0.5 hbar omega`` to each active mode.
  The default is ``yes``.

``phase``
  ``random`` or ``zero``. ``random`` samples each active mode phase uniformly
  from ``[0, 2 pi)`` and is the default. ``zero`` is intended for debugging.

``replicas``
  Number of independent QCT replicas. Harmonic sampling currently supports
  any positive value with one ordinary scalar NEP potential and
  ``pbc=F F F``. The same normal modes and Hessian are shared, while each
  replica's first sampling attempt uses ``seed + replica``. If the sampled
  geometry cannot be corrected to the requested energy on the real potential,
  that replica is deterministically resampled. ``Seed`` in the QCT output is
  the seed of the accepted sampling attempt. The ``phase_point`` path still
  supports ``replicas 1`` only.

Propagation
-----------

QCT propagation uses the same GPU velocity-Verlet NVE integrator as the
:attr:`nve` ensemble. Thermostatted or barostatted QCT propagation is not
enabled by this keyword.

The harmonic initializer runs before the first force evaluation of a run, so
the first force is computed from the generated QCT positions. Automatic Hessian
sampling requires the input structure to be a stationary point; it writes
``qct_stationary.xyz``, ``qct_hessian.out``, and ``qct_eigenvector.out`` before
generating the phase point. For a first-order saddle, the imaginary mode is
not sampled as a harmonic oscillator: its coordinate is zero and its momentum
is launched through the dividing surface. A later crossing back through the
dividing surface does not terminate or invalidate the trajectory.

Initial-condition output
------------------------

For harmonic initialization, GPUMD writes
:ref:`qct_initial.out <qct_initial_out>` with the sampled energy, phase, and
normal coordinate/momentum of every mode. It also writes
:ref:`qct_initial.xyz <qct_initial_xyz>` with the exact Cartesian positions and
velocities before the first force evaluation and
:ref:`qct_initial_summary.csv <qct_initial_summary>` with the accepted energy
correction for each replica. These files use overwrite mode.

For a single replica, the standard trajectory output remains available::

    dump_xyz -1 0 100 trajectory.xyz mass velocity

For a native batch run, use the QCT-specific output::

    dump_qct 100

This writes one frame per replica with ``Replica``, ``Step``, and ``Seed``
metadata, plus per-replica energies in ``qct_thermo.csv``. Other measurement
keywords are currently rejected for batch runs because their reductions do
not yet have per-replica semantics. The QCT analysis and multi-replica
aggregation tools are documented in ``tools/qct/README.md``.

.. _lsc_ivr:

LSC-IVR post-processing
-----------------------

When ``wigner`` sampling is used, the trajectory and summary files are
post-processed by ``tools/qct/lsc_ivr.py`` to compute quantum-corrected
time-correlation functions::

    python3 tools/qct/lsc_ivr.py \
      --trajectory qct_trajectory.xyz \
      --summary qct_initial_summary.csv \
      --config lsc_ivr.json \
      --output qct_lsc_correlation.csv \
      --fft qct_lsc_spectrum.csv \
      --time-step 0.1 \
      --dump-interval 10

The LSC-IVR correlation function is::

    C_AB(t) = sum_i w_i * A(0)_i * B(t)_i  /  sum_i w_i

where ``w_i`` is the Wigner anharmonic reweighting weight from
``qct_initial_summary.csv``.  The tool prefers the ``log_wigner_weight``
column for numerical stability, falling back to ``wigner_weight`` or
defaulting to 1.0 if neither is present.

The JSON configuration file defines the operators ``A`` and ``B``::

    {
      "operator_A": {"name": "position", "params": {"atom": 0, "axis": 0}},
      "operator_B": {"name": "position", "params": {"atom": 0, "axis": 0}}
    }

Available operators:

==================== ============================== ============================================
Name                 Parameters                     Description
==================== ============================== ============================================
``position``         ``atom``, ``axis``             Cartesian position component
``velocity``         ``atom``, ``axis``             Cartesian velocity component
``com_position``    ``atom_indices`` (optional)    Center-of-mass position magnitude
``com_velocity``    ``atom_indices`` (optional)    Center-of-mass speed
``bond_length``      ``atom1``, ``atom2``           Distance between two atoms
``kinetic_energy``   ``atom_indices`` (optional)    Total kinetic energy (eV)
``point_charge_dipole`` ``axis``, ``charges``       Dipole moment component from point charges
==================== ============================== ============================================

The ``--fft`` option produces a spectral density CSV with columns
``frequency_THz``, ``wavenumber_cm_inv``, and ``intensity``.

The standard error of the correlation uses the importance-sampling
(ratio estimator) variance::

    Var[C_hat(t)] = (1/N) * sum_i [w_i^2 * (f_i - C_hat)^2] / (sum_i w_i)^2

where ``f_i = A(0)_i * B(t)_i`` and ``C_hat`` is the estimated mean.

A complete user guide with worked examples (OH radical, ethanol,
ground-state Wigner), troubleshooting, and physical constants is provided
in ``docs/lsc_ivr.md``.
