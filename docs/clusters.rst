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

Task Arguments
------------------

.. argparse::
   :module: esteem.tasks.clusters
   :func: get_parser
   :nodescription:

Standalone module routines
--------------------------

.. automodule:: esteem.tasks.clusters
   :exclude-members: ClustersTask
   :members:


