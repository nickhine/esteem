.. _solvate:

Solvate module
==============

The Solvate task surrounds a solute molecule with explicit solvent in a periodic
box and runs classical molecular dynamics to produce well-equilibrated
snapshots of the solvated system. It runs in five phases:

#. **Setup**: solvates the solute and assigns force field parameters.
#. **Heating**: ramps the temperature from zero to the target temperature,
   with SHAKE constraints on bonds to hydrogen.
#. **Density equilibration**: constant-pressure MD to reach the correct
   density.
#. **Equilibration**: constraints are removed and the system is equilibrated
   in the target ensemble.
#. **Snapshots**: ``nsnaps`` geometries are stored, each separated by
   ``nsteps`` MD steps, so that they are decorrelated.

Any counterions are restrained from coming too close to the solute. The output
is a trajectory of snapshots, ``{solute}_{solvent}_solv.traj``, which is the
input to the :ref:`clusters` task. The task requires a Molecular Dynamics
wrapper such as :ref:`Amber <amber>` or :ref:`LAMMPS <lammps>`, or an ML
wrapper used as a force field.

.. currentmodule:: esteem.tasks.solvate

SolvateTask
------------

.. autoclass:: SolvateTask
   :members:

Task attributes
---------------

.. task-attributes:: esteem.tasks.solvate.SolvateTask

   wrapper: The Molecular Dynamics wrapper that runs the MD phases, for example ``AmberWrapper()`` or ``MACEWrapper()``.
   script_settings: Settings for the job script written in each solute-solvent MD directory (see :ref:`parallel-execution`).
   setup_cell: Optional function that builds the solvated model in place of the default Amber setup. It is called from inside the wrapper's output directory.

Standalone module routines
--------------------------

.. automodule:: esteem.tasks.solvate
   :exclude-members: SolvateTask
   :members:


