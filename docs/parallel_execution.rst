.. _parallel-execution:

Parallel Execution
==================

Most HPC clusters run a job queue, usually based on either SLURM or PBS. ESTEEM can write job scripts for either of these, but I use SLURM on most of the machines I have access to so this is likely to be better-tested.

All the job scripts created by ESTEEM write to a log, with the name ``<seed>_<task>_<array_id>.log``, for example ``cate_solvate_3.log`` for the 3rd job of the solvate task from the ``cate`` workflow script. This output is mirrored to the SLURM/PBS output file.

After running the ``scripts`` task for your workflow, you will see a different script for each combination of task and ``target`` (ie argument set). Each of these scripts is capable of running that task for all of the solute molecules, or all of the combinations of solute and solvent (as appropriate to the task). If you submit the job script with no array job specification, all the tasks will run sequentially, looping over solutes (and solvents, if applicable to the task).

Identifying array task IDs
--------------------------

Let's say we have 2 different solute molecules, and 3 possible solutes:

.. code-block:: python

   all_solutes = {'cate': 'catechol', 'tert': '4-tert-butylcatechol'}
   all_solvents = {'watr': 'water', 'cycl': 'cyclohexane', 'meth': 'methanol'}

These become zero-indexed lists when the script is run, so within the ``solutes`` task, array task ID 0 refers to ``cate``, and task ID 1 refers to ``tert``.

In ``solvate``, the number of possible task IDs is the product of the solutes and solvents, i.e. 6 here. Task ID 0 is ``cate_watr``, task ID 1 is ``cate_cycl``, 3 is ``tert_watr`` etc, up to 5, ``tert_meth``.

In ``clusters``, one first runs a top-level setup job, then the actual job script for a calculation is submitted from a directory that the top-level job has set up for you, which will have the name '{solute}_{solvent}_{exc_suffix}'. Therefore the array task ID's refer to which cluster index is to be calculated. These should range from ``0`` to ``solvate_args.nsnaps - 1`` as they are zero-indexed. If you have, say, 200 snapshots, you might wish at first just to run every 10th snapshot.

Array job submission
--------------------

Array jobs, which are supported by most modern queuing systems, are very useful for many of the tasks in ESTEEM.

The syntax for array job for a range of job numbers in SLURM is (here we are running the ``solvate`` task for all 6 of our combinations of solute and solvent)::

  $ sbatch --array=0-5 cate_solvate_sub

To run a maximum of 3 tasks at a time, you can append ``%3`` to the range::

  $ sbatch --array=0-5%3 cate_solvate_sub

To run every 10th entry among all the extracted ``clusters`` jobs between 0 and 90, we can append ``:10`` to the range::

  $ sbatch --array=0-90:10 cate_clusters_sub

Most of the tasks in ESTEEM are set up to automatically resume a half-finished run, so if some jobs have timed out and failed, you may want to restart a specific selection. Let's say we wanted to restart jobs 3,5 and 7 of the ``solutes`` task::

  $ sbatch --array=3,5,7 cate_solutes_sub

Predefined script settings
--------------------------

If you prefer, you can write your own HPC job script and invoke your workflow script, which should then launch parallel jobs inheriting the number of nodes, number of processes per node and number of cores per process from the job environment.

However, ESTEEM also comes with some predefined parallelisation setups that can can be supplied to the drivers from your workflow script. At present these represent the machines I have access to as of 2020. I am happy to add new definitions to this list for regular users of the code.

For example, if I wanted to launch on the ``nanosim`` queue on the University of Warwick's SCRTP Cluster of Workstations, I could use the ``drivers.nanosim_1node`` predefined settings dictionary. The definition is in ``drivers.py``:

.. code-block:: python

  nanosim_1node = {'account': 'phspvr',
                   'partition': 'nanosim',
                   'nodes': 1,
                   'ntask': 24,
                   'ncpu': 1,
                   'time': '48:00:00',
                   'mem': '2679mb'}

If I wanted to tweak something (let's say the run time and the account name), I could do:

.. code-block:: python

  script_settings = deepcopy(nanosim_1node)
  script_settings['time'] = '24:00:00'
  script_settings['account']: 'mstdjh'

Existing definitions insider drivers.py include ``athena_1node``, ``athena_4node``, ``athena_10node``, ``nanosim_1node``, ``nanosim_1core``, ``archer2_1node``. Please feel free to modify these or write your own. A list of arguments understood by the ``make_sbatch`` routine can be found in its entry in the ``drivers`` documentation.
