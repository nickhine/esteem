.. _openqp:

OpenQP Wrapper
==============

.. currentmodule:: esteem.wrappers.openqp

The OpenQP wrapper runs electronic structure calculations with
`OpenQP <https://github.com/Open-Quantum-Platform/openqp>`__, with a particular
focus on Mixed-Reference Spin-Flip TDDFT (MRSF-TDDFT). MRSF-TDDFT treats the
ground state and the excited states on an equal footing, so it is well suited
to generating training data for excited-state potential energy surfaces.

Each calculation writes an OpenQP input file, runs the ``openqp`` command on
it, and reads the results from the JSON file that OpenQP writes. The command
can be set with ``setup(openqp_cmd=...)`` or with the ``ESTEEM_OPENQP_COMMAND``
environment variable.

Supported:

* ground-state RHF/RKS energies and forces
* MRSF-TDDFT energies and forces for one or several states, from a triplet
  ROHF reference. If ``mrsf_ground`` is True (the default), target 0 is the
  MRSF S0 state.
* implicit solvent (PCM, using the ddX backend), for energies only
* gas-phase geometry optimisation of a single state
* vertical MRSF excitation energies and oscillator strengths

Not yet supported: molecular dynamics, dispersion corrections, and forces in
implicit solvent.

OpenQPWrapper
-------------

.. autoclass:: OpenQPWrapper
   :members:
