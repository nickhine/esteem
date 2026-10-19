.. _clusters:

Clusters module
===============

The Clusters task turns snapshots of a solvated system into cluster models and
calculates their energies or excitations. For each snapshot of an MD
trajectory, which may come from the :ref:`solvate`, :ref:`qmd_trajectories`
or :ref:`ml_trajectories` task, it:

#. deletes counterions, after checking that none are inside the cluster.
#. "carves" a sphere around the solute by deleting every solvent molecule
   that has no atom within ``radius`` of any solute atom. Solvent molecules
   within ``kernel`` of the solute are tagged.
#. rotates and centres the cluster in the cell, so that the solute has a
   consistent orientation across snapshots.
#. runs a singlepoint or excited-state calculation on the cluster with the
   supplied wrapper, with optional implicit solvent around the cluster.

Each snapshot is an independent calculation, so the task is normally run as
an array job with one snapshot per task ID (see :ref:`parallel-execution`).
The results are the input to the :ref:`spectra` task. If you run singlepoints
on a range of geometries, they can also be used as training data for the
:ref:`ml_training` task.

.. currentmodule:: esteem.tasks.clusters

ClustersTask
------------

.. autoclass:: ClustersTask
   :members:

Task attributes
---------------

.. task-attributes:: esteem.tasks.clusters.ClustersTask

   wrapper: The wrapper that runs the calculation on each cluster, for example ``ORCAWrapper()``. Setting it also sets ``output`` to the wrapper's output name.
   script_settings: Settings for the job scripts written for each clusters directory (see :ref:`parallel-execution`).
   target: The electronic states to calculate: a list of state indices, a dict of ML model heads, or an int ``n`` meaning states ``n`` to ``nroots``. If not set, states 0 to ``nroots`` are calculated. Set this explicitly: some code paths fail if it is missing.
   exc_dir_suffix: If set, used instead of ``exc_suffix`` in the name of the calculation directory ``{solute}_{solvent}_{exc_dir_suffix}`` (set by :mod:`esteem.active_learning`).

Standalone module routines
--------------------------

.. automodule:: esteem.tasks.clusters
   :exclude-members: ClustersTask
   :members:


