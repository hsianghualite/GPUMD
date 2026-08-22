.. _hac_out:
.. index::
   single: hac.out (output file)

``hac.out``
===========

This file contains the heat current auto-correlation (:term:`HAC`) function and the running thermal conductivity (:term:`RTC`) from the the :ref:`EMD method for heat transport <green_kubo_method>` method.
It is produced when invoking the :ref:`compute_hac keyword <kw_compute_hac>` in the :ref:`run.in input file <run_in>`.

File format
-----------
This file reads

* column 1: correlation time (in units of ps)
* column 2: :math:`\langle J_x^{\text{in}}(0)J_x^{\text{tot}}(t)\rangle` (in units of eV\ :math:`^3`/amu)
* column 3: :math:`\langle J_x^{\text{out}}(0)J_x^{\text{tot}}(t)\rangle` (in units of eV\ :math:`^3`/amu)
* column 4: :math:`\langle J_y^{\text{in}}(0)J_y^{\text{tot}}(t)\rangle` (in units of eV\ :math:`^3`/amu)
* column 5: :math:`\langle J_y^{\text{out}}(0)J_y^{\text{tot}}(t)\rangle` (in units of eV\ :math:`^3`/amu)
* column 6: :math:`\langle J_z^{\text{tot}}(0)J_z^{\text{tot}}(t)\rangle` (in units of eV\ :math:`^3`/amu)
* column 7: :math:`\kappa_x^{\text{in}}(t)` (in units of W/mK)
* column 8: :math:`\kappa_x^{\text{out}}(t)` (in units of W/mK)
* column 9: :math:`\kappa_y^{\text{in}}(t)` (in units of W/mK)
* column 10: :math:`\kappa_y^{\text{out}}(t)` (in units of W/mK)
* column 11: :math:`\kappa_z^{\text{tot}}(t)` (in units of W/mK)

Note that the :term:`HAC` and the :term:`RTC` are decomposed as described in [Fan2017]_.
This decomposition is useful for 2D materials but not necessary for 3D materials.
For 3D materials, one can sum up some columns to get the conventional data.
For example:

.. math::

   \langle J_x^{\text{tot}}(0)J_x^{\text{tot}}(t) \rangle
   = \langle J_x^{\text{in}}(0)J_x^{\text{tot}}(t) \rangle
   + \langle J_x^{\text{out}}(0)J_x^{\text{tot}}(t) \rangle

and

.. math::
   
   \kappa_x^{\text{tot}}(t) = \kappa_x^{\text{in}}(t) + \kappa_x^{\text{out}}(t).

Note that the cross term introduced in [Fan2017]_ has been evenly attributed to the in-plane and out-of-plane components.
This has been justified in [Fan2019]_.

For a native QCT/LSC-IVR batch with ``replicas > 1``, GPUMD first computes one
HAC curve per replica and then combines those curves with the normalized
Wigner weights. It does not correlate the summed heat currents of different
replicas. The weighted batch audit is written to ``hac_replica.out`` and the
weights used for the reduction are written to ``hac_reweighting.csv``.

``hac_replica.out`` has 12 whitespace-separated columns: the zero-based
replica index followed by the same time, HAC, and RTC columns as ``hac.out``.
``hac_reweighting.csv`` contains ``replica``, ``seed``,
``log_wigner_weight``, and ``normalized_weight``. For ``replicas = 1``, the
single-trajectory ``hac.out`` remains a conditional HAC curve; its weight must
be normalized together with other independent trajectories, for example by
``tools/qct/run_multigpu.py``. Multiplying a single curve by its raw weight is
not a normalized reweighted estimator.

Only the potential part of the heat current is included.
If the convective part of the heat current is important in your system, you can use the :ref:`compute keyword <kw_compute>` to calculate and output the heat current data and post-process it by yourself.
