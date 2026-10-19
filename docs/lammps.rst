.. _lammps:

LAMMPS Wrapper
==============

.. currentmodule:: esteem.wrappers.lammps

The LAMMPS wrapper runs classical molecular dynamics with
`LAMMPS <https://www.lammps.org>`__, through ASE's LAMMPS calculator. It
provides the Molecular Dynamics functions of the :ref:`solvate` task
(``minimise``, ``heatup``, ``densityequil``, ``equil`` and ``snapshots``) as an
alternative to :ref:`Amber <amber>`, using force fields converted from Amber
format, with the same default settings as the Amber wrapper (2 fs timestep,
Langevin thermostat, 9 Å cutoff).

:Website: https://www.lammps.org
:Source code: https://github.com/lammps/lammps
:Citation: A.P. Thompson *et al.*, LAMMPS - a flexible simulation tool for particle-based materials
   modeling at the atomic, meso, and continuum scales, *Comput. Phys. Commun.* **271**, 108171 (2022).
   `doi:10.1016/j.cpc.2021.108171 <https://doi.org/10.1016/j.cpc.2021.108171>`__

API
---

.. automodule:: esteem.wrappers.lammps
   :members:
