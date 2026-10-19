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

Task attributes
---------------

.. task-attributes:: esteem.tasks.ml_trajectories.MLTrajTask

   wrapper: The Machine Learning wrapper whose model drives the MD, for example ``MACEWrapper()``.
   script_settings: Settings for the job scripts written for this task (see :ref:`parallel-execution`).
   snap_wrapper: The wrapper used to evaluate each snapshot, for example the whole committee of models. If None, ``wrapper`` is used. Also used to recalculate carved clusters.
   snap_calc_params: Calculator parameters for ``snap_wrapper`` (keys such as ``'target'``, ``'calc_seed'``, ``'calc_suffix'`` (a dict for a committee), ``'calc_dir_suffix'`` and ``'calc_prefix'``). If None, the parameters of the MD calculator are used.
   snap_target: The state or states evaluated at each snapshot (set by :mod:`esteem.active_learning`).
   which_targets: The state of each trajectory in ``which_trajs``. An entry may be a list such as ``[1, 0]``: the first state drives the MD, and all are evaluated at each snapshot.
   ref_mol_dir: Directory of reference solute and solvent results, used when recalculating carved clusters (``{target}`` is replaced by the state). Required if ``recalculate_carved_traj`` is True.

Standalone module routines
--------------------------

.. automodule:: esteem.tasks.ml_trajectories
   :exclude-members: MLTrajTask
   :members:


