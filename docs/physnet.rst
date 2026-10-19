.. _physnet:

PhysNet Wrapper
===============

.. currentmodule:: esteem.wrappers.physnet

The PhysNet wrapper trains and runs `PhysNet <https://github.com/MMunibas/PhysNet>`__
neural network potentials. PhysNet is a message-passing neural network which
predicts energies, forces, dipole moments and partial charges, with optional
long-range electrostatic and dispersion corrections. It is a Machine Learning
wrapper, so it provides the ``train``, ``load``, ``geom_opt`` and ``run_md``
functions used by the :ref:`ml_training`, :ref:`ml_testing` and
:ref:`ml_trajectories` tasks. Training data are converted from ESTEEM
trajectory files to PhysNet's ``.npz`` format, and the training settings can be
changed through the ``train_args`` attribute of the wrapper.

:Source code: https://github.com/MMunibas/PhysNet
:Citation: O.T. Unke and M. Meuwly, PhysNet: A Neural Network for Predicting Energies, Forces,
   Dipole Moments, and Partial Charges, *J. Chem. Theory Comput.* **15**, 3678 (2019).
   `doi:10.1021/acs.jctc.9b00181 <https://doi.org/10.1021/acs.jctc.9b00181>`__

PhysNetWrapper
--------------

.. autoclass:: PhysNetWrapper
   :members:
