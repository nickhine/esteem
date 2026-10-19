.. _ml_trajectories:

ML_Trajectories module
======================

The ML_Trajectories task runs molecular dynamics with a trained ML potential
in place of ab initio forces. This makes it possible to run long trajectories
of large systems at a fraction of the cost. Starting from an initial geometry
(often a snapshot from :ref:`solvate`), it equilibrates and then records
``nsnap`` snapshots, each separated by ``md_steps`` steps, for each of ``ntraj``
independent trajectories. It can also:

* restart a trajectory from a new initial snapshot if it blows up, and continue
  an unfinished run (``continuation``).
* stop a trajectory when a committee of models disagrees by more than
  ``md_stop_stde_thresh`` (useful for active learning).
* carve each snapshot down to a cluster around the solute
  (``carve_trajectory_radius``) and recalculate it with a second wrapper.
* post-process the dipole moment along the trajectory into an infrared
  spectrum (``freq``).

The snapshots can be passed to the :ref:`clusters` task for excited-state
calculations, and then to the :ref:`spectra` task.

.. currentmodule:: esteem.tasks.ml_trajectories

MLTrajTask
----------

.. autoclass:: MLTrajTask
   :members:

Task Arguments
------------------

.. argparse::
   :module: esteem.tasks.ml_trajectories
   :func: get_parser
   :nodescription:

Standalone module routines
--------------------------

.. automodule:: esteem.tasks.ml_trajectories
   :exclude-members: MLTrajTask
   :members:


