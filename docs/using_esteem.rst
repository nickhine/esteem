.. _using-esteem:

Using ESTEEM
============

ESTEEM is primarily intended to be invoked via a script or notebook that defines a set of tasks then imports the driver modules to run one of these tasks. The same script, with different arguments, can be used for all parts of a workflow. It is also possible to invoke some tasks directly from the command line providing all inputs as arguments. Both are useful under different circumstances: the command-line tools allow you to explore various options to discover what works, whereas a script allows you to create a documented, reproducible, adjustable workflow.

The overall model is: :ref:`solutes` -> :ref:`solvate` -> :ref:`clusters` -> :ref:`spectra`.

In workflows utilising an ML-based NN surrogate model, there is an extra set of tasks between clusters and spectra: :ref:`ml_train` -> :ref:`ml_trajectories` -> :ref:`clusters`

The use of spectral warping will require the :ref:`solutes` or :ref:`clusters` to have been run twice, with different levels of theory (eg different XC functionals).

Command-line tools
------------------

Command-line use is helpful for setup and testing: naturally you must not run any large calculations on the login node of your HPC cluster! The scripts of ESTEEM can be invoked with commands such as this::

  $ python ~/esteem/esteem/tasks/solutes.py --namefile my_solutes --basis 6-311++G** --func PBE0

where ``my_solutes`` refers to a file which contains a list of the solutes to process. ``solvate`` can be replaced by any of the other tasks, namely ``solvate``, ``clusters``, ``spectra`` (as well as ``qmd_trajectories``, ``ml_training``, ``ml_testing``, ``ml_trajectories`` for the Machine-Learning part of the package). If you intend to use this a lot, it might be simpler to make everything in ``~/esteem/esteem/tasks/`` executable with ``chmod +x ~/esteem/esteem/tasks/*.py`` and add this directory to your ``$PATH``. That way you can simplify the above to::

  $ solutes.py --namefile my_solutes --basis 6-311++G** --func PBE0

Help on the arguments for each of the commands can be found in the documentation or by invoking the command-line help text::

  $ python ~/esteem/esteem/tasks/solvate.py --help

which gives a listing of all the available input variables and their default values.

Refer to the bash script in ``/examples`` to see an example of how to use the command line tools to perform steps of a calculation. It would be pperfectly possible to run a whole set of calculations this way, but the reality is that explicit solvent calculations are both computationally demanding, and demanding of human attention and intervention. If you find you need to repeat some part of the calculation, it would be fiddly to reproduce the whole chain of steps correctly.

In complex cases, there may be many combinations of solute and solvent to investigate, or you may need to perform a range of calculations for different combinations of XC functional, basis, various radii and other sizes, MD temperature etc etc. This is where writing a workflow script becomes highly preferable, and is the main use case of ESTEEM for generating publishable results.

Scripting with ESTEEM
---------------------

Creating a workflow script or notebook to use ESTEEM in a semi-automated fashion only requires a small amount of coding. Even so, it may be preferable to start from one of the examples and modify it appropriately. The basic idea is that a single python script sets up all the calculation parameters, then picks which job to perform based on command line arguments. This script is also capable of writing HPC job submission scripts for SLURM or PBS which invoke specific tasks in parallel. Array jobs are particularly useful in this context, as will be discussed later.

Let's say we want to define the solute to be catechol and set up calculations in explicit cyclohexane and methanol, and put the workflow in a script called ``cate.py``. The necessary components of a workflow script or notebook are as follows:

First we define a dictionary whose keys are short-form names of the solutes you want to use. It is advisable to keep the names in here short, as otherwise filenames become very long. The entries for each key are long-form names for labelling plots and other outputs. If you wish the molecular structures to be downloaded from an existing database (see ``get_xyz_files`` in ``solutes``), these can be full IUPAC names:

.. code-block:: python

   all_solutes = {'cate': 'catechol'}

Next we define a similar dictionary for the solvent molecules:

.. code-block:: python

   all_solvents = {'cycl': 'cyclohexane', 'meth': 'methanol'}

We import the top level 'drivers' module of ESTEEM:

.. code-block:: python

   from esteem import drivers

We generate a set of default arguments for each of the main tasks of the code, in ``args`` objects:

.. code-block:: python

    solutes_args, solvate_args, clusters_args, spectra_args = drivers.get_default_args()

We can now modify the members of each of these ``args`` according to our needs. Their members are the same as the command line arguments of the scripts as discussed above. For example, here we set the basis and functional for the Solutes task, choose a small box for the solvate task, and a small cluster radius for the clusters task:

.. code-block:: python

   solutes_args.basis = '6-311++G**'
   solutes_args.func = 'PBE0'
   solvate_args.boxsize = 15
   clusters_args.radius = 3

We initialise any wrapper settings by calling setup routines for the wrappers we need (these may become classes in the near future). Here we are using Amber for MD, and NWChem for DFT and TDDFT:

.. code-block:: python

   from esteem.wrappers import amber
   solvate_wrapper = amber.AmberWrapper()
   from esteem.wrappers import nwchem
   nwchem.nwchem_setup()
   solutes_wrapper = nwchem.NWChemWrapper()
   from esteem.wrappers import onetep
   clusters_wrapper = onetep.OnetepWrapper()

We set up choices enabling the code to write job scripts defining appropriate parallelisation (templates for machines in use by my group are provided in drivers, and can be copied and modified as required):

.. code-block:: python

   make_script = drivers.make_sbatch
   solutes_script_settings = drivers.nanosim_1node
   solvate_script_settings = drivers.nanosim_1node
   clusters_script_settings = drivers.nanosim_1node
   # make any changes you need to the script settings here
 
Finally, we invoke the top-level driver of ESTEEM, which unfortunately has a rather long argument list concatenating all the things we have just set up:
 
.. code-block:: python
 
   drivers.main(all_solutes,all_solvents,
                solutes_args,solvate_args,clusters_args,spectra_args,
                solutes_wrapper,solvate_wrapper,clusters_wrapper,
                make_script,solutes_script_settings,solvate_script_settings,clusters_script_settings)

This top-level driver then works out, from the rest of the command-line arguments with which the script was invoked, what part of the calculation to invoke. For example, a script to invoke the solutes task for catechol (shortname ``cate``)  One of the most useful tasks is 'scripts', which writes HPC submission scripts for all the other tasks.

Let's say I named my top-level script 'cate.py', I would write scripts with::
   $ python cate.py scripts

Then submit the resulting scripts to my queuing system, for example::
   $ sbatch cate_solutes_sub

Internally, this job script will invoke the top-level script again, with a specific task::
   $ python cate.py solutes cate

which runs the Solutes task.

After the invocation of the top-level driver, you might want to have the your script quit python::
   exit()

so that in the same file you can define other post-processing functions or interactive cells in a notebook which require access to the workflow settings, but which you do not want to run every time the script is invoked.

Providing multiple sets of arguments
------------------------------------

In the example above we provided a single set of arguments for each of the tasks. However, we might want to run with several sets of options, for example to investigate convergence with respect to basis size, check different XC functionals, run MD at different temperatures, or converge the cluster excitations with respect to cluster radius.

To enable this sort of study, if any of the argument lists to Drivers.main are python dictionaries, then the command-line argument ``target`` decides which entry in that dictionary to use. The scripts task will write a job script for each possible ``target`` in the list appropriate to each task.

It is recommended to make a set of "master" arguments first, copy that list and change what you need. The routine deepcopy is useful for this:

.. code-block:: python

    from copy import deepcopy
    all_clusters_args = {}
    for rad in [0,3,6,9]:
        target = f'solvR{rad}'
        all_clusters_args[target] = deepcopy(clusters_args)
        all_clusters_args[target].radius = rad
 
Then you simply pass 'all_clusters_args' rather than 'clusters_args' in the call to drivers.main()

Example Scripts
---------------

The package contains several example scripts of increasing complexity to illustrate the functionality of the code:
   * ``cate.py`` - a simple script as described above to run solvated catechol in cyclohexane and methanol
   * ``cate_full.py`` - a more complex script that includes PBE and PBE0 solutes calculations, followed by a test of different solvent shell radii, and a spectral warp of the cluster results from PBE to PBE0.

The page on :ref:`examples` gives a walkthrough of how to submit these to a computing cluster with SLURM.


Locating binaries for packages
------------------------------

By default, ESTEEM assumes that any dependency codes have been loaded via a module system, and are available by their standard executable names from the command line, for example "nwchem", "onetep", "sander" to invoke NWChem, ONETEP and Amber's sander tool, respectively. 

If this is not the case, for example if you have installed the codes yourself, you can adjust names of the executables when you setup the calculator::
   >>> onetep.onetep_setup(onetep_cmd='~/onetep/bin/onetep.archer2',mpirun='srun',set_pseudo_path='~/NCP17_PBE_OTF/',set_pseudo_suffix="_NCP17_PBE_OTF.usp")

More details can be found in the documentation pages for each wrapper.
