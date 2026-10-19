.. _examples:

Examples of using ESTEEM
========================

This page walks through running a complete explicit solvent workflow with the example script ``examples/cate_demo.py``, which calculates the spectrum of catechol in cyclohexane and in methanol. It uses ORCA for DFT and TDDFT, and Amber for MD, with small settings so that it runs quickly:

.. literalinclude:: ../examples/cate_demo.py
      :language: python

The script is built in the same way as the one described in :ref:`using-esteem`. I will assume we are using SLURM rather than PBS throughout this example. If you are not on one of the machines known to ``parallel.get_default_script_settings``, replace those calls with your own script settings first (see :ref:`parallel-execution`).

First, we generate job submission scripts::

   $ python cate_demo.py scripts cate_demo

``scripts`` is the 'task', and ``cate_demo`` is the 'seed' for the calculation (which should match the name of the script).

This will produce job submission scripts for each task: ``cate_demo_solutes_sub``, ``cate_demo_solvents_sub``, ``cate_demo_solvate_sub``, ``cate_demo_clusters_sub`` and ``cate_demo_spectra_sub``.

Solutes Task
------------

The script adds the solvents to ``all_solutes``, so the Solutes task optimises the solvent molecules as well as catechol, and we only need to submit the Solutes job (the Solvents job is for workflows that keep the solvents out of ``all_solutes``). There are three molecules, so we can run them as three separate jobs with an array task::

   $ sbatch --array=0-2 cate_demo_solutes_sub

Array task IDs 0, 1 and 2 refer to ``cate``, ``cycl`` and ``meth``, in the order of ``all_solutes``.

These will create a new directory ``gs_PBE`` (set by ``solutes_task.directory``) and run ORCA geometry optimisation and TDDFT calculations in there, first in gas phase then in implicit solvent. The starting structures are downloaded from PubChem by name. If your cluster cannot reach PubChem, you will need to supply initial guess structures in ``xyz/`` for all three molecules. The outputs of the geometry optimisations will be in ``gs_PBE/geom`` and the resulting structures in ``gs_PBE/opt``. The calculations are then repeated in each implicit solvent, producing the final geometries in ``gs_PBE/is_opt_cycl`` and ``gs_PBE/is_opt_meth``, and implicit solvent excitations in ``gs_PBE/is_tddft_cycl`` and ``gs_PBE/is_tddft_meth``.

Solvate Task
------------

We now want to run the MD. Like the Clusters task below, the Solvate task runs in two stages. First, run it from the base directory to set up a directory and job script for each solute-solvent pair. This only copies files, so it is quick enough to run interactively::

  $ python cate_demo.py solvate cate_demo

This creates four directories: ``cate_cycl_md`` and ``cate_meth_md`` for catechol in each solvent, and ``cycl_cycl_md`` and ``meth_meth_md`` for boxes of pure solvent (because the solvents are also in ``all_solutes``). Each contains the optimised geometries from ``gs_PBE`` (``solvate_task.md_geom_prefix``) and a job script. Submit the job script in each directory you need, for example::

  $ cd cate_cycl_md
  $ sbatch cate_cycl_md_solvate_sub
  $ cd ../cate_meth_md
  $ sbatch cate_meth_md_solvate_sub

There are many output files in each directory, for the different steps of setup and different MD runs, but the most important ones are the 'trajectory' of snapshots: ``cate_cycl_solv.traj`` and ``cate_meth_solv.traj``, which get used by the next step. Here each contains 20 snapshots (``solvate_task.nsnaps``).

Clusters Task
-------------

The Clusters task is next. As for the Solvate task, we run its setup stage from the base directory::

  $ python cate_demo.py clusters cate_demo

This is a short task that sets up directories: ``cate_cycl_exc`` and ``cate_meth_exc`` in this case (and ``cycl_cycl_exc`` and ``meth_meth_exc`` for the pure solvents, for which no excitations are calculated). In each it will put a job script. Change into the first directory and list the contents::

  $ cd cate_cycl_exc
  $ ls

You should see four files: ``cate.xyz``, ``cycl.xyz``, ``cate_cycl_solv.traj`` (a link to the MD trajectory) and ``cate_cycl_exc_sub``. The latter is the job script which can be used to launch a calculation for each of the snapshot clusters. The array task ID is the snapshot index, so to run all 20 snapshots::

  $ sbatch --array=0-19 cate_cycl_exc_sub

On some machines, the default script settings for ORCA run 5 consecutive snapshots in each job, starting from the array task ID. In that case, submit every 5th ID instead::

  $ sbatch --array=0-19:5 cate_cycl_exc_sub

The result will be an ORCA calculation for each cluster, in the ``orca`` subdirectory. You can check their progress with ``squeue`` and by looking at the log files. You may want to inspect a few of the outputs to check the behaviour is as expected.

Spectra Task
------------

This is where the raw results from the Clusters task get turned into spectra for plotting purposes. In ``cate_demo.py``, ``spectra_task.files`` lists the ORCA outputs in ``orca/``, so we run this task from inside a clusters directory. It is quick, so you can run it interactively to generate a ``.png`` file on your compute cluster::

  $ cd cate_cycl_exc
  $ python ../cate_demo.py spectra cate_demo

This writes the spectrum to ``test.png`` (``spectra_task.output``). Alternatively, you can transfer the output files from the clusters run back to another machine for interactive analysis in a notebook, by calling routines such as ``drivers.spectra_driver()``.

More complex examples
---------------------

``examples/cate_dft.py`` extends this workflow with several levels of theory and target states for the Solutes task, two cluster radii for the Clusters task, and a spectral warp from PBE to PBE0 in the Spectra task. ``examples/cate_ml.py`` adds the ML tasks, training MACE potentials by active learning and using them to generate the trajectories from which spectra are calculated.
