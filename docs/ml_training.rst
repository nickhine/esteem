.. _ml_training:

ML_Training module
==================

The ML_Training task trains a machine-learned potential energy surface on
trajectories of reference calculations, such as the output of the
:ref:`qmd_trajectories` or :ref:`clusters` tasks. It merges the selected
trajectories (``which_trajs``) into a single training file and either splits
off a fraction for validation or uses a separate set of validation
trajectories. It then calls the ``train`` function of the ML wrapper.

``target`` can be a single electronic state or a dictionary of states, for
models with one head per state. A ``diff`` entry trains on the energy
difference between two states. The trained model is used by the
:ref:`ml_testing` and :ref:`ml_trajectories` tasks. The task requires a Machine
Learning wrapper such as :ref:`MACE <mace>`.

.. currentmodule:: esteem.tasks.ml_training

MLTrainingTask
--------------

.. autoclass:: MLTrainingTask
   :members:

Task Arguments
------------------

.. argparse::
   :module: esteem.tasks.ml_training
   :func: get_parser
   :nodescription:

Standalone module routines
--------------------------

.. automodule:: esteem.tasks.ml_training
   :exclude-members: MLTrainingTask
   :members:


