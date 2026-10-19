.. _qmd_trajectories:

QMD_Trajectories module
=======================

The QMD_Trajectories task generates trajectories with ab initio (quantum)
molecular dynamics on a single molecule or cluster, in the ground state or an
excited state (set with ``target``). After ``nequil`` discarded equilibration
segments, it records ``nsnap`` snapshots, each separated by ``md_steps`` steps,
under a thermostat at ``temp``. Several independent trajectories can be run at
once (``ntraj``).

It can also recalculate energies and forces, at a new level of theory or for a
different state, for every frame of an existing trajectory, rather than running
new dynamics. Its main use is to generate training and validation data for the
:ref:`ml_training` task. It requires an electronic structure wrapper that
provides ``run_md``.

.. currentmodule:: esteem.tasks.qmd_trajectories

QMDTrajTask
-----------

.. autoclass:: QMDTrajTask
   :members:

Task attributes
---------------

.. task-attributes:: esteem.tasks.qmd_trajectories.QMDTrajTask

   wrapper: The electronic structure wrapper that runs the MD and singlepoint calculations, for example ``ORCAWrapper()``.
   script_settings: Settings for the job scripts written for this task (see :ref:`parallel-execution`).

Standalone module routines
--------------------------

.. automodule:: esteem.tasks.qmd_trajectories
   :exclude-members: QMDTrajTask
   :members:


