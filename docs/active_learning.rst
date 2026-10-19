.. _active_learning:

Active learning
===============

.. currentmodule:: esteem.active_learning

An active-learning workflow (see the overview in :doc:`introduction`) repeats the same few tasks
for every electronic state, committee member and generation: :ref:`ml_training`,
:ref:`ml_trajectories`, :ref:`clusters` (to select and label new snapshots) and
:ref:`ml_testing`. Rather than setting up each of these tasks by hand, you create one
*prototype* of each task with the settings that are common to all of them, and the
``create_*`` functions of :mod:`esteem.active_learning` make a copy of it for every
combination, with the names, directories and trajectory links filled in.

Each function returns a dict of tasks keyed by task name, which is the ``target`` used on
the command line, so the dicts can be passed straight to ``drivers.main``.

Example
-------

This outline follows the structure of ``examples/cate_ml.py``. It sets up three generations
of a committee of three MACE models for the ground and first excited states, with
uncertainty-biased selection of new snapshots (the ``u`` at the end of each calculator
name):

.. code-block:: python

   from esteem.active_learning import (create_mltrain_tasks, create_mltraj_tasks,
                                       create_clusters_tasks, create_mltest_tasks)

   targets = {0: 'gs', 1: 'es1'}
   train_calcs = ["MACEac0u", "MACEac1u", "MACEac2u"]   # generations 0, 1 and 2
   rand_seed = {'a': 123, 'b': 456, 'c': 789}           # committee of 3 models
   truth = 'orca'                                       # label of the DFT data

   # mltrain_task, mltraj_task, clusters_task and mltest_task are prototype tasks,
   # with their wrappers and common settings already set up

   all_mltrain_tasks = create_mltrain_tasks(
       mltrain_task, train_calcs=train_calcs, seeds=[seed], targets=targets,
       rand_seed=rand_seed, meth='', truth=truth,
       traj_suffixes=[truth], dir_suffixes={truth: 'init'},
       ntraj={('gs', truth): 1, ('es1', truth): 0}, iter_dir_suffixes=['mlclus'])

   all_mltraj_tasks = create_mltraj_tasks(
       mltraj_task, train_calcs=train_calcs, targets=targets, rand_seed=rand_seed,
       meth='', traj_suffix='mldyn', md_wrapper=mltraj_task.wrapper,
       snap_wrapper=mltraj_task.snap_wrapper)

   all_clusters_tasks = create_clusters_tasks(
       clusters_task, train_calcs=train_calcs, seed=seed, traj_suffix='mlclus',
       md_suffix='mldyn', md_dir_suffix='mldyn', targets=targets,
       rand_seed=rand_seed, meth='', truth=truth)

   all_mltest_tasks = create_mltest_tasks(
       mltest_task, train_calcs=train_calcs, seeds=[seed], targets=targets,
       rand_seed=rand_seed, truth=truth, meth='', traj_suffixes=[truth],
       dir_suffixes={truth: 'init'}, iter_dir_suffixes=['mlclus'],
       ntraj={('gs', truth): 1, ('es1', truth): 0})

Each generation then runs in order, for example for generation 0 of the ground state::

   $ python script.py train <seed> gs_MACEac0ua     # train committee member a (and b, c)
   $ python script.py traj <seed> gs_MACEac0uax3    # MD driven by member a (and b, c), scored by all 3
   $ python script.py mlclus <seed> gs_MACEac0u_U   # select and label new snapshots
   $ python script.py test <seed> gs_MACEac0ua      # test on the new snapshots

after which the generation 1 models (``MACEac1u``) are trained on all the data so far.

.. note::

   The ``create_*`` functions change the prototype task as they go, so the prototype is
   left with the settings of the last task created. Use a fresh prototype, or reset the
   settings you rely on, before passing it to another ``create_*`` call.

Module reference
----------------

.. automodule:: esteem.active_learning
   :members:
