.. _parallel-execution:

Parallel Execution
==================

Most HPC clusters run a job queue, usually based on either SLURM or PBS. ESTEEM can write job scripts for either of these, but I use SLURM on most of the machines I have access to so this is likely to be better-tested.

All the job scripts created by ESTEEM write to a log, with the name ``<seed>_<target>_<task>_<array_id>.log`` (without the ``<target>`` part if there is only one set of arguments, and without ``_<array_id>`` if it is not an array job), for example ``cate_solutes_1.log`` for array task 1 of the solutes task from the ``cate`` workflow script. This output is mirrored to the SLURM/PBS output file.

After running the ``scripts`` task for your workflow, you will see a different script for each combination of task and ``target`` (ie argument set). For the Solutes task, the script is capable of running the task for all of the solute molecules. If you submit the job script with no array job specification, the solutes are run sequentially. The Solvate and Clusters tasks instead work in two stages, described below.

Identifying array task IDs
--------------------------

Let's say we have 2 different solute molecules, and 3 possible solvents:

.. code-block:: python

   all_solutes = {'cate': 'catechol', 'tert': '4-tert-butylcatechol'}
   all_solvents = {'watr': 'water', 'cycl': 'cyclohexane', 'meth': 'methanol'}

These become zero-indexed lists when the script is run, so within the ``solutes`` task, array task ID 0 refers to ``cate``, and task ID 1 refers to ``tert``.

If you also add the solvents to ``all_solutes`` (as the example scripts do, so that the Solutes task optimises them too), they follow on in the same order: here task ID 2 would be ``watr``, and so on.

In ``solvate``, one first runs the task from the base directory of the workflow. This is a quick setup stage which creates a directory ``{solute}_{solvent}_{md_suffix}`` (for example ``cate_watr_md``) for each combination of solute and solvent, copies in the optimised geometries, and writes a job script there named ``{solute}_{solvent}_{md_suffix}_solvate_sub``. You then submit each of those job scripts from its own directory, without an array specification. If you do run the setup stage as an array job, the task IDs number the combinations of solute and solvent, looping over solutes for each solvent in turn: task ID 0 is ``cate_watr``, task ID 1 is ``tert_watr``, 2 is ``cate_cycl`` etc, up to 5, ``tert_meth``.

``clusters`` works in the same way: one first runs the top-level setup task, then the actual job script for a calculation is submitted from a directory that the setup task has created for you, which will have the name ``{solute}_{solvent}_{exc_suffix}``. Therefore the array task IDs refer to which cluster index is to be calculated. These should range from ``0`` to ``nsnaps - 1`` (where ``nsnaps`` is the number of snapshots from the Solvate task) as they are zero-indexed. If you have, say, 200 snapshots, you might wish at first just to run every 10th snapshot.

Array job submission
--------------------

Array jobs, which are supported by most modern queuing systems, are very useful for many of the tasks in ESTEEM.

The syntax for array job for a range of job numbers in SLURM is (here we are running the ``solvate`` task for all 6 of our combinations of solute and solvent)::

  $ sbatch --array=0-5 cate_solvate_sub

To run a maximum of 3 tasks at a time, you can append ``%3`` to the range::

  $ sbatch --array=0-5%3 cate_solvate_sub

To run every 10th entry among all the extracted ``clusters`` jobs between 0 and 90, we can append ``:10`` to the range (from within the ``cate_cycl_exc`` directory)::

  $ sbatch --array=0-90:10 cate_cycl_exc_sub

Most of the tasks in ESTEEM are set up to automatically resume a half-finished run, so if some jobs have timed out and failed, you may want to restart a specific selection. Let's say we wanted to restart jobs 3, 5 and 7 of the ``solutes`` task::

  $ sbatch --array=3,5,7 cate_solutes_sub

Running several array entries per job
-------------------------------------

Some script settings run more than one array entry in each job, which is useful when each calculation is short compared to the time spent waiting in the queue. Each job then runs consecutive array task IDs, starting from its own ID:

* The default settings for the ORCA and OpenQP wrappers on the machines known to ``get_default_script_settings`` run 5 consecutive IDs one after the other in each job. Submit every 5th ID, for example ``--array=0-195:5`` for 200 snapshots.
* Settings with the suffix ``_100`` (such as ``avon_1node_100`` and ``sulis_1node_100``) use GNU parallel to run 100 consecutive IDs in each job, several at a time. Submit every 100th ID, for example ``--array=0-900:100``.

Predefined script settings
--------------------------

If you prefer, you can write your own HPC job script and invoke your workflow script, which should then launch parallel jobs inheriting the number of nodes, number of processes per node and number of cores per process from the job environment.

However, ESTEEM also comes with some predefined parallelisation setups that can be supplied to the drivers from your workflow script, as the ``script_settings`` of each task. These are defined in the ``parallel`` module, and represent the machines I have access to. I am happy to add new definitions to this list for regular users of the code.

``parallel.get_default_script_settings(wrapper)`` chooses one of these from the hostname of the machine and the type of wrapper, and also sets up some wrappers (such as the location of the ONETEP executable) for that machine. It raises an error on any machine it does not recognise, so elsewhere you need to supply your own script settings.

For example, if I wanted to launch on the ``nanosim`` queue on the University of Warwick's SCRTP Cluster of Workstations, I could use the ``parallel.nanosim_1node`` predefined settings dictionary. The definition is in ``parallel.py``:

.. code-block:: python

  nanosim_1node = {'partition': 'nanosim',
                   'nodes': 1,
                   'ntask': 24,
                   'ncpu': 1,
                   'time': '48:00:00',
                   'mem': '61679mb',
                   'declarations': '',
                   'endmatter': ''}

If I wanted to tweak something (let's say the run time and the account name), I could do:

.. code-block:: python

  from copy import deepcopy
  from esteem import parallel
  script_settings = deepcopy(parallel.nanosim_1node)
  script_settings['time'] = '24:00:00'
  script_settings['account'] = 'mstdjh'
  clusters_task.script_settings = script_settings

Existing definitions inside ``parallel.py`` include ``nanosim_1node``, ``nanosim_1core``, ``nanosimd_1node``, ``nanosimd_8mpi``, ``archer2_1node``, ``archer2_4node``, ``archer2_8node``, ``archer2_16node``, ``avon_1node_gnu``, ``avon_1node_intel``, ``avon_1gpu``, ``avon_1node_100``, ``sulis_1node``, ``sulis_1gpu``, ``sulis_3gpu`` and ``sulis_1node_100``. Please feel free to modify these or write your own. The keys understood by the ``make_sbatch`` routine include ``account``, ``partition``, ``qos``, ``gres``, ``nodes``, ``ntask``, ``ncpu``, ``time``, ``mem``, ``declarations`` (lines added to the script before the main command, such as ``module load`` commands), ``endmatter`` (lines added after it), ``command``, ``execpath`` and ``logdir``.

Parallel module reference
-------------------------

.. automodule:: esteem.parallel
   :members: make_sbatch, make_pbs, get_default_script_settings, get_array_task_id
