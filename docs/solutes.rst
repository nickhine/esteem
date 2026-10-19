.. _solutes:

Solutes module
==============

The Solutes task prepares optimised structures and gas-phase or implicit-solvent
excitations for each solute molecule. It should also be run on the solvent
molecules. For each molecule it runs:

#. geometry optimisation in the gas phase. Starting geometries are read from
   ``xyz/``, or downloaded with :meth:`SolutesTask.get_xyz_files`.
#. optionally, a rotamer search to find the lowest-energy conformer, and
   rotation of the molecule into the xy-plane.
#. excited state (TDDFT) calculations in the gas phase, if ``nroots > 0``.
#. optionally, vibrational frequencies.
#. if ``solvent`` is set, re-optimisation in implicit solvent, followed by
   excitations and frequencies in implicit solvent.

The optimised implicit-solvent geometries (``is_opt_{solvent}/``) are the
starting point for the :ref:`solvate` task. The implicit-solvent excitations
(``is_tddft_{solvent}/``) can be used to calculate a spectral warp in the
:ref:`spectra` task. The task requires an electronic structure wrapper such as
:ref:`NWChem <nwchem>`, ORCA or :ref:`OpenQP <openqp>`.

.. currentmodule:: esteem.tasks.solutes

SolutesTask
-----------

.. autoclass:: SolutesTask
   :members:

Task attributes
---------------

.. task-attributes:: esteem.tasks.solutes.SolutesTask

   wrapper: The electronic structure wrapper that runs the geometry optimisations, singlepoints, TDDFT and frequency calculations, for example ``ORCAWrapper()``.
   script_settings: Settings for the job scripts written for the Solutes and Solvents tasks (see :ref:`parallel-execution`).

Standalone module routines
--------------------------

.. automodule:: esteem.tasks.solutes
   :exclude-members: SolutesTask
   :members:


