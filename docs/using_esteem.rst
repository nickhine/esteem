.. _using-esteem:

Using ESTEEM
============

ESTEEM is primarily intended to be invoked via a script or notebook that defines a set of tasks then calls the main driver to run one of these tasks. The same script, with different arguments, can be used for all parts of a workflow. It is also possible to invoke some tasks directly from the command line providing all inputs as arguments. Both are useful under different circumstances: the command-line tools allow you to explore various options to discover what works, whereas a script allows you to create a documented, reproducible, adjustable workflow.

The overall model is: :ref:`solutes` -> :ref:`solvate` -> :ref:`clusters` -> :ref:`spectra`.

In workflows utilising an ML surrogate model, there is an extra set of tasks between clusters and spectra. Training data are generated with :ref:`clusters` or :ref:`qmd_trajectories`, then: :ref:`ml_training` -> :ref:`ml_testing` -> :ref:`ml_trajectories` -> :ref:`clusters` -> :ref:`spectra`.

The use of spectral warping will require the :ref:`solutes` or :ref:`clusters` to have been run twice, with different levels of theory (eg different XC functionals).

Command-line tools
------------------

Command-line use is helpful for setup and testing: naturally you must not run any large calculations on the login node of your HPC cluster! The task modules of ESTEEM can be invoked with commands such as this::

  $ python -m esteem.tasks.solutes --namefile my_solutes --basis 6-311++G** --func PBE0

where ``my_solutes`` refers to a file which contains a list of the solutes to process, one per line, in the form ``shortname: IUPAC name``. ``solutes`` can be replaced by any of the other tasks, namely ``solvate``, ``clusters``, ``spectra`` (as well as ``qmd_trajectories``, ``ml_training``, ``ml_testing``, ``ml_trajectories`` for the Machine-Learning part of the package).

Note that when run this way, each task uses a fixed default wrapper (for example NWChem for ``solutes`` and ``clusters``, and Amber for ``solvate``). To use any other wrapper, write a workflow script as described below.

Help on the arguments for each of the commands can be found in the documentation or by invoking the command-line help text::

  $ python -m esteem.tasks.solvate --help

which gives a listing of all the available input variables and their default values.

It would be perfectly possible to run a whole set of calculations this way, but the reality is that explicit solvent calculations are both computationally demanding, and demanding of human attention and intervention. If you find you need to repeat some part of the calculation, it would be fiddly to reproduce the whole chain of steps correctly.

In complex cases, there may be many combinations of solute and solvent to investigate, or you may need to perform a range of calculations for different combinations of XC functional, basis, various radii and other sizes, MD temperature etc etc. This is where writing a workflow script becomes highly preferable, and is the main use case of ESTEEM for generating publishable results.

Scripting with ESTEEM
---------------------

Creating a workflow script or notebook to use ESTEEM in a semi-automated fashion only requires a small amount of coding. Even so, it may be preferable to start from one of the examples and modify it appropriately. The basic idea is that a single python script sets up all the calculation parameters, then picks which job to perform based on command line arguments. This script is also capable of writing HPC job submission scripts for SLURM or PBS which invoke specific tasks in parallel. Array jobs are particularly useful in this context, as will be discussed later.

Let's say we want to define the solute to be catechol and set up calculations in explicit cyclohexane and methanol, and put the workflow in a script called ``cate.py``. The necessary components of a workflow script or notebook are as follows:

First we define a dictionary whose keys are short-form names of the solutes you want to use. It is advisable to keep the names in here short, as otherwise filenames become very long. The entries for each key are long-form names for labelling plots and other outputs. If you wish the molecular structures to be downloaded from an existing database (see ``get_xyz_files`` in :ref:`solutes`), these can be full IUPAC names:

.. code-block:: python

   all_solutes = {'cate': 'catechol'}

Next we define a similar dictionary for the solvent molecules. The solvents also need optimised structures from the Solutes task, so we add them to the list of solutes:

.. code-block:: python

   all_solvents = {'cycl': 'cyclohexane', 'meth': 'methanol'}
   all_solutes.update(all_solvents)

We import the ``drivers`` and ``parallel`` modules of ESTEEM, and create a task object for each of the main tasks. Each task object starts with default values for all of its arguments:

.. code-block:: python

   from esteem import drivers, parallel
   from esteem.tasks.solutes import SolutesTask
   from esteem.tasks.solvate import SolvateTask
   from esteem.tasks.clusters import ClustersTask
   from esteem.tasks.spectra import SpectraTask

   solutes_task = SolutesTask()
   solvate_task = SolvateTask()
   clusters_task = ClustersTask()
   spectra_task = SpectraTask()

We can now modify the attributes of each task according to our needs. Their attributes are the same as the command line arguments of the task modules as discussed above. For example, here we set the basis and functional for the Solutes task, choose a small box for the Solvate task, and a small cluster radius for the Clusters task:

.. code-block:: python

   solutes_task.basis = '6-311++G**'
   solutes_task.func = 'PBE0'
   solutes_task.directory = 'gs_PBE0'
   solvate_task.boxsize = 15
   solvate_task.md_geom_prefix = 'gs_PBE0/'
   clusters_task.radius = 3

Each task needs a wrapper, which determines which code is used to run its calculations (see :ref:`wrappers`). Here we are using ORCA for DFT and TDDFT, and Amber for MD:

.. code-block:: python

   from esteem.wrappers.orca import ORCAWrapper
   from esteem.wrappers.amber import AmberWrapper

   solutes_task.wrapper = ORCAWrapper()
   solutes_task.wrapper.setup(nprocs=8)
   solvate_task.wrapper = AmberWrapper()
   clusters_task.wrapper = ORCAWrapper()
   clusters_task.wrapper.setup(nprocs=8)

Each task also needs script settings, which define the parallelisation of the job scripts that ESTEEM writes for it. ``parallel.get_default_script_settings`` chooses settings suitable for the wrapper on the machines in use by my group, and raises an error on any other machine, where you will need to write your own settings dictionary. Templates for these machines are provided in ``parallel``, and can be copied and modified as required (see :ref:`parallel-execution`):

.. code-block:: python

   solutes_task.script_settings = parallel.get_default_script_settings(solutes_task.wrapper)
   solvate_task.script_settings = parallel.get_default_script_settings(solvate_task.wrapper)
   clusters_task.script_settings = parallel.get_default_script_settings(clusters_task.wrapper)
   # make any changes you need to the script settings here

Finally, we invoke the top-level driver of ESTEEM, passing it the solutes, solvents and tasks we have just set up:

.. code-block:: python

   drivers.main(all_solutes, all_solvents,
                all_solutes_tasks=solutes_task,
                all_solvate_tasks=solvate_task,
                all_clusters_tasks=clusters_task,
                all_spectra_tasks=spectra_task,
                make_script=parallel.make_sbatch)

The ML tasks are passed in the same way, with the keyword arguments ``all_qmd_tasks``, ``all_mltrain_tasks``, ``all_mltest_tasks`` and ``all_mltraj_tasks``. Any task you do not need can be left out.

This top-level driver then works out, from the command-line arguments with which the script was invoked, what part of the calculation to run. The script is invoked as ``python <script>.py <task> <seed> [<target>]``, where ``seed`` is a name for the workflow (usually the same as the script name) that is used in the names of job scripts and log files. One of the most useful tasks is ``scripts``, which writes HPC submission scripts for all the other tasks.

Let's say I named my top-level script ``cate.py``. I would write scripts with::

   $ python cate.py scripts cate

Then submit the resulting scripts to my queuing system, for example::

   $ sbatch cate_solutes_sub

Internally, this job script will invoke the top-level script again, with a specific task::

   $ python cate.py solutes cate

which runs the Solutes task for all of the solutes, or for just one of them if it is run as an array job (see :ref:`parallel-execution`).

After the invocation of the top-level driver, you might want to have your script quit python::

   exit()

so that in the same file you can define other post-processing functions or interactive cells in a notebook which require access to the workflow settings, but which you do not want to run every time the script is invoked.

Providing multiple sets of arguments
------------------------------------

In the example above we provided a single task object for each of the tasks. However, we might want to run with several sets of options, for example to investigate convergence with respect to basis size, check different XC functionals, run MD at different temperatures, or converge the cluster excitations with respect to cluster radius.

To enable this sort of study, any of the task arguments to ``drivers.main`` can be a python dictionary of task objects, and the command-line argument ``target`` decides which entry in that dictionary to use. The ``scripts`` task will write a job script for each possible ``target`` in the dictionary for each task, named ``<seed>_<target>_<task>_sub``.

It is recommended to set up a "master" task first, copy it and change what you need. The routine deepcopy is useful for this:

.. code-block:: python

    from copy import deepcopy
    all_clusters_tasks = {}
    for rad in [0, 3, 6, 9]:
        target = f'solvR{rad}'
        clusters_task.radius = rad
        clusters_task.exc_suffix = target
        all_clusters_tasks[target] = deepcopy(clusters_task)

Then you simply pass ``all_clusters_tasks`` rather than ``clusters_task`` in the call to ``drivers.main()``, and run a specific target with, for example::

   $ python cate.py clusters cate solvR6

Example Scripts
---------------

The ``examples`` directory of the package contains several example scripts of increasing complexity to illustrate the functionality of the code:

   * ``cate_demo.py`` - a simple script, similar to the one described above, to run solvated catechol in cyclohexane and methanol with ORCA and Amber.
   * ``cate_dft.py`` - a more complex script that runs the Solutes task with PBE and PBE0 for the ground and first two excited states, followed by clusters calculations for two solvent shell radii, and spectra with a spectral warp from PBE to PBE0.
   * ``cate_ml.py`` - an active-learning workflow that trains, tests and uses MACE potentials for ML trajectories, adding new cluster calculations to the training data at each generation, then generates spectra from the ML trajectories.

The page on :ref:`examples` gives a walkthrough of how to submit these to a computing cluster with SLURM.


Locating binaries for packages
------------------------------

By default, ESTEEM assumes that any dependency codes have been loaded via a module system, and are available by their standard executable names from the command line, for example "nwchem", "onetep", "sander" to invoke NWChem, ONETEP and Amber's sander tool, respectively.

If this is not the case, for example if you have installed the codes yourself, you can adjust the names of the executables with the ``setup`` method of the wrapper::

   >>> from esteem.wrappers.onetep import OnetepWrapper
   >>> wrapper = OnetepWrapper()
   >>> wrapper.setup(onetep_cmd='~/onetep/bin/onetep.archer2', mpirun='srun',
   ...               set_pseudo_path='~/NCP17_PBE_OTF/', set_pseudo_suffix="_NCP17_PBE_OTF.usp")

More details can be found in the documentation pages for each wrapper.
