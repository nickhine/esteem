.. _ml_testing:

ML_Testing module
=================

The ML_Testing task checks the accuracy of a trained ML potential. It merges a
set of reference trajectories that were not used in training, and evaluates the
energies and forces of every frame with the trained calculator. It then compares
these with the reference values. It writes a trajectory of the ML results and
plots the predicted energies against the reference energies, coloured by the
RMS force deviation, to ``plotfile``. Run it after :ref:`ml_training` and
before you rely on the model in :ref:`ml_trajectories`.

.. currentmodule:: esteem.tasks.ml_testing

MLTestingTask
-------------

.. autoclass:: MLTestingTask
   :members:

Task Arguments
------------------

.. argparse::
   :module: esteem.tasks.ml_testing
   :func: get_parser
   :nodescription:

Standalone module routines
--------------------------

.. automodule:: esteem.tasks.ml_testing
   :exclude-members: MLTestTask
   :members:


