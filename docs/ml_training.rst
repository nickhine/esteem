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

Task attributes
---------------

.. task-attributes:: esteem.tasks.ml_training.MLTrainingTask

   wrapper: The Machine Learning wrapper that trains the model, for example ``MACEWrapper()``.
   script_settings: Settings for the job scripts written for this task (see :ref:`parallel-execution`).
   train_params: Extra keyword arguments passed to the wrapper's ``train`` method.
   which_targets: The state (for example ``'gs'``) of each trajectory in ``which_trajs``. Required when ``which_trajs`` is set.
   which_targets_valid: As ``which_targets``, for the trajectories in ``which_trajs_valid``.
   rand_seed: Random seed for the random split into training and validation data (123 if not set).

Standalone module routines
--------------------------

.. automodule:: esteem.tasks.ml_training
   :exclude-members: MLTrainingTask
   :members:


