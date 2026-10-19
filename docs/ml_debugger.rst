.. _ml_debugger:

ML_Debugger module
==================

.. currentmodule:: esteem.tasks.ml_debugger

The ML Debugger is a diagnostic tool, not a workflow task: it has no
command-line arguments and is not run by the drivers. It is used interactively
to find where a molecular dynamics trajectory generated with an ML potential
goes wrong.

:meth:`MLDebugger.debug_cluster_trajectory` steps through a trajectory and
applies a user-supplied test function to each solvent molecule in each frame.
At the first frame where the test fails (for example, a molecule that has
broken apart), it returns the frames before and after that point. These frames
can be cut down to a cluster around the bad molecule, which is moved to the
start of the atom list. The resulting frames can be inspected, or recalculated
with an ab initio method and added to the training set.

MLDebugger
----------

.. autoclass:: MLDebugger
   :members:
