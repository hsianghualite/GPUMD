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

For a first-order transition-state structure, the same interface can launch a
trajectory from the dividing surface::

    ensemble qct harmonic temperature 300 stationary_point saddle reaction_direction random seed 12345 zpe yes

For reproducibility or debugging, GPUMD can also consume a precomputed binary
normal-mode output directly::

    ensemble qct harmonic eigenvector eigenvector.out exclude_lowest 6 min_frequency 0.001 temperature 300 seed 12345 zpe yes phase random replicas 1

Parameters
----------

``phase_point``
  Use the positions and velocities already present in ``model.xyz``. No native
  QCT initial-condition sampling is performed.

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
  bond with a deterministic orientation.

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
  option is applied. The default is ``random``.

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
velocities before the first force evaluation. Both files use overwrite mode.

For a single replica, the standard trajectory output remains available::

    dump_xyz -1 0 100 trajectory.xyz mass velocity

For a native batch run, use the QCT-specific output::

    dump_qct 100

This writes one frame per replica with ``Replica``, ``Step``, and ``Seed``
metadata, plus per-replica energies in ``qct_thermo.csv``. Other measurement
keywords are currently rejected for batch runs because their reductions do
not yet have per-replica semantics. The QCT analysis and multi-replica
aggregation tools are documented in ``tools/qct/README.md``.
