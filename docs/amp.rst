.. _amp:

AMP Wrapper
===========

.. currentmodule:: esteem.wrappers.amp

The AMP wrapper trains and runs neural network potentials with
`Amp <https://amp.readthedocs.io>`__, the Atomistic Machine-learning Package,
through its ASE calculator. It is a Machine Learning wrapper, so it provides the
``train``, ``load``, ``geom_opt`` and ``run_md`` functions used by the
:ref:`ml_training`, :ref:`ml_testing` and :ref:`ml_trajectories` tasks. It was
the first ML wrapper in ESTEEM: for new work, the :ref:`MACE <mace>` wrapper is
likely to be a better choice.

:Website: https://amp.readthedocs.io
:Source code: https://bitbucket.org/andrewpeterson/amp
:Citation: A. Khorshidi and A.A. Peterson, Amp: A modular approach to machine learning in atomistic
   simulations, *Comput. Phys. Commun.* **207**, 310 (2016).
   `doi:10.1016/j.cpc.2016.05.010 <https://doi.org/10.1016/j.cpc.2016.05.010>`__

API
---

.. automodule:: esteem.wrappers.amp
   :members:
