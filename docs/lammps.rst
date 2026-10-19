.. _lammps:

LAMMPS Wrapper
==============

.. currentmodule:: esteem.wrappers.lammps

The LAMMPS wrapper runs classical molecular dynamics with
`LAMMPS <https://www.lammps.org>`__, through ASE's LAMMPS calculator, using
force fields converted from Amber format.

.. warning::

   This wrapper is incomplete. Its ``minimise``, ``heatup``, ``densityequil``,
   ``equil`` and ``snapshots`` methods do not yet match how the :ref:`solvate`
   task calls them, and most of them only evaluate an energy rather than
   running the MD phase. Use the :ref:`Amber <amber>` wrapper for the Solvate
   task.

:Website: https://www.lammps.org
:Source code: https://github.com/lammps/lammps
:Citation: A.P. Thompson *et al.*, LAMMPS - a flexible simulation tool for particle-based materials
   modeling at the atomic, meso, and continuum scales, *Comput. Phys. Commun.* **271**, 108171 (2022).
   `doi:10.1016/j.cpc.2021.108171 <https://doi.org/10.1016/j.cpc.2021.108171>`__

API
---

.. automodule:: esteem.wrappers.lammps
   :members:
