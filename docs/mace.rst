.. _mace:

MACE Wrapper
============

.. currentmodule:: esteem.wrappers.mace

The MACE wrapper trains and runs `MACE <https://github.com/ACEsuit/mace>`__
equivariant message-passing neural network potentials. It is a Machine Learning
wrapper, so it provides the ``train``, ``load``, ``geom_opt`` and ``run_md``
functions used by the :ref:`ml_training`, :ref:`ml_testing` and
:ref:`ml_trajectories` tasks. It also provides the Molecular Dynamics functions
(``heatup``, ``densityequil``, ``equil``, ``snapshots``) so that a trained
model can stand in for a force field in the :ref:`solvate` task.

Training data are read from ESTEEM trajectory files, converted to extended XYZ
and passed to MACE's own training script. You can change any of MACE's training
arguments by modifying the ``train_args`` attribute of the wrapper before
training. A run that already has a finished model is skipped, so you have to
delete the model before you can retrain it.

When ``target`` is a dictionary (for example ``{0: 'gs', 1: 'es1'}``), a single
multi-head model is trained with one head per electronic state. The head is
chosen at evaluation time. You can also set ``foundation_model`` to evaluate a
pre-trained MACE foundation model instead of a model trained by ESTEEM.

X-MACE
------

With ``MACEWrapper(xmace=True)``, the wrapper uses the ``xmace`` package
instead of ``mace``. X-MACE trains one ``ExcitedMACE`` model for several
electronic states at once. The ``target`` dictionary gives the states in model
order, and its values name the states that are selected at evaluation time.

MACEWrapper
-----------

.. autoclass:: MACEWrapper
   :members:
