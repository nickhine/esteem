.. _amber:

Amber Wrapper
=============

.. currentmodule:: esteem.wrappers.amber

The Amber wrapper runs classical molecular dynamics with
`Amber <https://ambermd.org>`__, and is the standard Molecular Dynamics wrapper
for the :ref:`solvate` task. It parameterises the solute and solvent molecules
with the General Amber Force Field, using ``antechamber`` and ``parmchk`` (or
``acpype``) from AmberTools to assign atom types and partial charges, builds
the solvated box with ``tleap``, and runs the minimisation, heating, density
equilibration, equilibration and snapshot phases with ``sander``
(``sander.MPI`` if it is available). Restraints can be added, for example to
keep counterions away from the solute.

All the tools used are included in the free AmberTools package.

:Website: https://ambermd.org
:Citation: D.A. Case *et al.*, AmberTools, *J. Chem. Inf. Model.* **63**, 6183 (2023).
   `doi:10.1021/acs.jcim.3c01153 <https://doi.org/10.1021/acs.jcim.3c01153>`__

   D.A. Case *et al.*, Recent Developments in Amber Biomolecular Simulations,
   *J. Chem. Inf. Model.* **65**, 7835 (2025).
   `doi:10.1021/acs.jcim.5c01063 <https://doi.org/10.1021/acs.jcim.5c01063>`__

AmberWrapper
------------

.. autoclass:: AmberWrapper
   :members:
