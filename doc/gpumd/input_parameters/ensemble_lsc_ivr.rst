.. _kw_ensemble_lsc_ivr:

:attr:`ensemble` (LSC-IVR)
==========================

The :attr:`lsc_ivr` variant of the :attr:`ensemble` keyword provides a
dedicated entry point for Linearized Semiclassical Initial Value
Representation (:term:`LSC-IVR`) calculations.

LSC-IVR generates quantum-corrected initial conditions via Wigner sampling
and then propagates the system with classical NVE dynamics.  The key
advantage over the :attr:`qct` ensemble's ``wigner`` mode is that
:attr:`lsc_ivr` explicitly supports periodic boundary conditions (PBC),
making it suitable for condensed-phase systems such as crystalline solids
and liquids.

Syntax
------

::

    ensemble lsc_ivr T [key=value ...]

where ``T`` is the sampling temperature in Kelvin (use ``0`` for
ground-state Wigner).  The remaining key-value pairs are identical to
those accepted by ``ensemble qct wigner``.

Example: quantum-corrected thermal conductivity of a Si crystal::

    ensemble lsc_ivr 300 seed 12345 replicas 1 hessian_displacement 0.001 anharmonic_reweighting no

Example: ground-state Wigner for a molecular system without PBC::

    ensemble lsc_ivr 0 seed 42 replicas 64 hessian_displacement 0.001 anharmonic_reweighting yes

Parameters
----------

All parameters accepted by ``ensemble qct wigner`` are also accepted by
``ensemble lsc_ivr``:

``temperature`` (implicit)
  The first positional argument ``T`` sets the Wigner sampling temperature.

``seed``
  Random seed for Wigner sampling (default: 1).

``replicas``
  Number of independent Wigner-sampled replicas (default: 1).
  For periodic systems (``pbc=T T T``), only ``replicas 1`` is currently
  supported because batch expansion would require replicating the supercell.

``hessian_displacement``
  Finite-difference displacement for the automatic Hessian (default: 0.001 Å).

``min_frequency``
  Minimum active-mode frequency in THz (default: 0.001).  Modes below this
  threshold are treated as rigid translations/rotations.

``anharmonic_reweighting``
  ``yes`` (default) or ``no``.  When enabled, each replica receives an
  importance-sampling weight :math:`w_i = \\exp(-\\beta \\Delta V)`.

``exclude_lowest``
  Number of lowest-frequency modes to exclude as rigid (default: 6 for
  molecular systems, automatically adjusted for periodic systems).

Periodic Boundary Conditions
-----------------------------

Unlike ``ensemble qct wigner``, which requires ``pbc=F F F`` when
``replicas > 1``, the :attr:`lsc_ivr` ensemble is designed for periodic
systems from the ground up:

- **``replicas 1`` with PBC**: Fully supported.  The system is propagated
  as a single NVE trajectory with quantum-corrected Wigner initial
  conditions.  This is the recommended workflow for condensed-phase
  thermal conductivity calculations (e.g., with ``compute_hac``).
- **``replicas > 1`` with PBC**: Not yet supported.  Batch expansion
  would require replicating the supercell box, which is not implemented.
  For multiple independent samples of a periodic system, run separate
  GPUMD jobs with different seeds.

Relationship to QCT
-------------------

Internally, ``ensemble lsc_ivr T ...`` is transformed into
``ensemble qct wigner T ...`` and all Wigner-sampling machinery is
reused.  The differences are:

1. **Ensemble type**: ``lsc_ivr`` uses type ``-14`` (vs QCT's ``-13``)
   to distinguish the two in logs and internal dispatch.
2. **PBC handling**: The LSC-IVR ensemble is intended for periodic
   systems, while QCT batch mode remains restricted to isolated molecules.
3. **Documentation**: LSC-IVR is documented separately to clarify its
   intended use case (condensed-phase quantum-corrected dynamics).

Post-processing
----------------

After the run, use ``tools/qct/lsc_ivr.py`` to compute LSC-IVR correlation
functions from the trajectory and ``qct_initial_summary.csv``.  See
:ref:`lsc_ivr` and ``docs/lsc_ivr.md`` for details.
