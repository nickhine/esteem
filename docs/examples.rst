.. _examples:

Examples of using ESTEEM
========================

In :ref:`using-esteem` we wrote the following short script, ``cate.py``, to run catechol in water:

.. literalinclude:: ../examples/cate.py
      :language: python

The following section addresses how we would use that script to run a complete explicit solvent workflow.

First, we generate job submission scripts::

   $ python cate.py scripts cate

``scripts`` is the 'task', and ``cate`` is the 'seed' for the calculation (which should match the name of the script).

This will produce five job submission scripts: ``cate_solutes_sub``, ``cate_solvents_sub``, ``cate_solvate_sub``, ``cate_clusters_sub`` and ``cate_spectra_sub``.

Solutes and Solvents Tasks
--------------------------

Let's run the first two at the same time. I will assume we are using SLURM rather than PBS throughout this example::

   $ sbatch cate_solutes_sub
   $ sbatch cate_solvents_sub

As directed in the script above, this will launch the Solutes task for the list of solutes (1 entry: ``cate``) and the list solvents (2 entries: ``cycl`` and ``meth``).

These will create a new directory ``PBE_6-31G`` and run NWChem geometry optimisation and TDDFT calculations in there, first in gas phase then in implicit solvent. If your cluster does not allow ``wget`` to access the Chemical Structure Resolver, you will need to supply initial guess structures in ``PBE_6-31G/xyz`` for all three molecules. The output file for the geometry optimisations will be ``PBE_6-31G/geom/cate/geom.nwo`` and the resulting structure will be in ``PBE_6-31G/opt``. Calculations will be repeated in water using the COSMO implicit solvent model to produce the final geometry ``PBE_6-31G/is_opt_watr``.

Solvate Task
------------

We now want to run the MD task. If we just ran::

  $ sbatch cate_solvate_sub
  
This would launch the Solvate task twice, one after the other, for the two different solvents. A more likely scenario is that we want to run these as two separate jobs, so we can use an array task::

  $ sbatch --array=0-1 cate_solvate_sub
  
Which launches two jobs, one for catechol in cyclohexane, one for catechol in ethanol. The results will go in separate directories, ``cate_cycl_md`` and ``cate_meth_md``.
There are many output files in each directory, for the different steps of setup and different MD runs, but the most important ones are the 'trajectory' of snapshots: ``cate_cycl_solv.traj`` and ``cate_meth_solv.traj`` which get used by the next step.

Clusters Task
-------------

The clusters task is next. We can run its setup task from the command line::

  $ python cate.py clusters cate
  
This is a short task that sets up directories - it will set up ``cate_cycl_exc`` and ``cate_meth_exc`` in this case. In each it will put a script. Change into the first directory and list the contents::

  $ cd cate_cycl_exc
  $ ls

You should see four files: ``cate.xyz``, ``cycl.xyz``, ``cate_cycl_solv.traj`` and ``cate_cycl_exc_sub``. The latter is the job script which can be used to launch a calculation for each of the snapshot clusters.

We would launch 10 calculations on equally-spaced snapshots from 0 to 90 (ie 0,10,20,30,...,90) as follows:

  $ sbatch --array=0-90:10 cate_cycl_exc_sub
  
The result will be a ONETEP calculation for each cluster. You can check their progress with ``squeue`` and ``tail *.out``.

Once they have finished, they will produce files with names such as ``cate_cycl_solv000.out`` (and any other output files produced by the code) you can run the Spectra task to generate plots.
You may want to inspect a few of these to check the behaviour is as expected.

Spectra Task
------------

This is where the raw results from the Clusters task get turned into spectra for plotting purposes. You can run this task interactively at the command line to generate ``.png`` files on your compute cluster::

  $ python cate.py spectra cate
  
or, if you prefer, you can transfer all the ``.out`` files from the clusters run back to another machine for interactive analysis in a notebook, by calling routines such as ``spectra_driver()``

