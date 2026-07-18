.. _qct_modes_in:
.. index::
   single: gpumd input files; qct_modes.in

``qct_modes.in``
================

This file provides harmonic normal modes for native QCT initial-condition
sampling. It is the editable text input option; GPUMD can alternatively read
its binary ``eigenvector.out`` via the :ref:`QCT ensemble keyword
<kw_ensemble_qct>`. Cartesian positions always come from the current
``model.xyz``.

File Format
-----------

The file is a text file. Blank lines and comments starting with ``#`` are
ignored.

Required header fields::

    QCT_MODES v1
    num_atoms N
    num_modes M
    frequency_unit THz
    coordinate_unit Angstrom
    mass_unit amu
    eigenvector_type mass_weighted
    normalization sum_e2_1
    reference_position yes

The ``reference_position`` line is optional. If present, its value must be
``yes``.

The atoms block records the atom order and masses associated with the modes::

    atoms
    # index symbol mass x0 y0 z0
    0 C 12.011 0.000000 0.000000 0.000000
    1 C 12.011 1.420000 0.000000 0.000000
    end_atoms

The atom order, symbols, masses, and ``num_atoms`` must match ``model.xyz``.

The modes block gives one mode at a time::

    modes
    mode 0 frequency 0.000000 active no
    0  0.70710678 0.00000000 0.00000000
    1  0.70710678 0.00000000 0.00000000
    end_mode

    mode 1 frequency 12.345678 active yes
    0  0.70710678 0.00000000 0.00000000
    1 -0.70710678 0.00000000 0.00000000
    end_mode
    end_modes

Only modes marked ``active yes`` are sampled. Use ``active no`` for
translational, rotational, imaginary, near-zero, or intentionally excluded
modes.

Conventions
-----------

Frequencies are ordinary frequencies in THz, not squared frequencies. Values
generated from ``eigenvector.out`` are therefore ``sqrt(omega2)``. The QCT
initializer converts them internally to angular frequencies by multiplying by
``2 pi``.

Eigenvectors are mass-weighted normal-mode eigenvectors normalized as::

    sum_i,alpha e_i,alpha,k^2 = 1

For each active mode, the generated Cartesian displacement and velocity are::

    delta r_i,alpha = e_i,alpha,k Q_k / sqrt(m_i)
    v_i,alpha       = e_i,alpha,k P_k / sqrt(m_i)

where ``Q_k`` and ``P_k`` are the sampled mass-weighted normal coordinate and
momentum.
