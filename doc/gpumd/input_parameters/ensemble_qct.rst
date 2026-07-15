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

Generate a harmonic QCT initial condition and then propagate it with NVE
dynamics::

    ensemble qct harmonic modes qct_modes.in temperature 300 seed 12345 zpe yes phase random replicas 1

Parameters
----------

``phase_point``
  Use the positions and velocities already present in ``model.xyz``. No native
  QCT initial-condition sampling is performed.

``harmonic``
  Generate a QCT phase point from harmonic normal modes in ``qct_modes.in``.

``modes``
  Path to the QCT normal-mode input file. The current implementation expects
  the text format documented in :ref:`qct_modes_in`.

``temperature``
  Sampling temperature in K. For each active mode, the implementation samples a
  classical Boltzmann mode energy and optionally adds zero-point energy.

``seed``
  Non-negative integer random seed. The default is ``1``.

``zpe``
  ``yes`` or ``no``. If ``yes``, add ``0.5 hbar omega`` to each active mode.
  The default is ``yes``.

``phase``
  ``random`` or ``zero``. ``random`` samples each active mode phase uniformly
  from ``[0, 2 pi)`` and is the default. ``zero`` is intended for debugging.

``replicas``
  Number of QCT replicas. The current implementation only supports ``1``; the
  keyword is parsed now to keep the interface compatible with future
  multi-replica support.

Propagation
-----------

QCT propagation uses the same GPU velocity-Verlet NVE integrator as the
:attr:`nve` ensemble. Thermostatted or barostatted QCT propagation is not
enabled by this keyword.

The harmonic initializer runs before the first force evaluation of a run, so
the first force is computed from the generated QCT positions.
