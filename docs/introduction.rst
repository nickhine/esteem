Introduction and Background
===========================

ESTEEM is a code for performing theoretical spectroscopy calculations to understand excited state and optical properties of molecular systems embedded in complex environments such as an explicit representation of a solvent.

An excellent introduction to the use of "explicit" representation of a solvent environment in the context of theoretical spectroscopy calculations can be found in Tim Zuehlsdorff and Christine Isborn's 2019 review:

Modeling absorption spectra of molecules in solution, Int J. Quant. Chem. 119, e25719
https://onlinelibrary.wiley.com/doi/full/10.1002/qua.25719

ESTEEM is an attempt to harness the flexibility provided by ASE, and the ease with which one can create "calculators" to perform atomistic simulation tasks, and use it to provide an API for explicit solvent calculations, including setting them up, running them on HPC systems, and analysing and plotting the results. 

Explicit Solvent Calculations
-----------------------------

The main explicit solvent code works on the basis of a set of predefined tasks representing steps in the workflow of a calculation of a solute in explicit solvent. These tasks are:

   1. Calculate the structure and electronic excitations of a solute molecule in gas-phase and/or implicit solvent.
   2. Calculate the structure (and excitations, if required) of the solvent molecules as above.
   3. Solvate the solute molecule, i.e. surround it by a realistic, well-equilibrated representation of the solvent in a large periodic box.
   4. Take a series of snapshots of the geometry of the solvated system during a long Molecular Dynamics trajectory, separated by a time interval long enough to decorrelate the snapshot geometries.
   5. Extract clusters from the snapshots of the solvated system, centered on the solute molecule, containing the solute molecule and all solvent molecules within a certain distance of the solute.
   6. Calculate the relevant electronic eigenstates, eg the ground and one or more excited states, for each extracted cluster. This may involve perform a Theoretical Spectroscopy (eg TDDFT) calculation on each extracted cluster. At this stage, possible uses split into either direct simulation of the spectra with ab initio methods, or the training and use of a surrogate model based on Machine Learning (for which see below).
   7. If required, solute spectra generated as in step 1 may be used to determine a "spectral warp". This is a mapping the result of one spectroscopy calculation (usually with a computationally-inexpensive level of theory) onto another (a state-of-the-art calculation with a high level of theory).
   8. Produce a final predicted spectrum for the solvated system, by averaging over the extracted clusters or by processing a molecular dynamics trajectory. The spectral warp can be applied to each snapshot. This spectrum can be used to predict the colour of the molecule in solvent, or to identify what is present in a mixture.

In ESTEEM, steps 1. and 2. are performed by the :mod:`esteem.tasks.solutes` module, steps 3. and 4. are performed by the 'Solvate' module, steps 5. and 6. are performed by the 'Clusters' module, and steps 7. and 8. are performed by the 'Spectra' module.

A workflow for ESTEEM usually takes the form of a single python script which both defines the calculation parameters and calls driver routines which perform the above tasks.

Machine Learning Potential Energy Surfaces
------------------------------------------

The ESTEEM code also has functionality for machine learning of potential energy surfaces of molecules. The processes involved here are as follows:

  1. Generate training data, eg by following steps 1.-6. of the list above.
  2. Train a Neural Network surrogate model of the ab initio electronic structure, using the :mod:`esteem.tasks.ml_training` module.
  3. Test the Neural Network surrogate model on alternative training data, using the :mod:`esteem.tasks.ml_testing` module
  4. Generate molecular dynamics trajectories on the potential energy surfaces of the NN surrogate model, likely using snapshots from step 4 of the list above as starting points, and store snapshots from these.
  5. Extract Clusters from the snapshots (again using the :mod:`esteem.tasks.clusters` module) and perform ground and excited state calculations on them.
  6. Generate spectra as in steps 7 and 8 above.

Active Learning
---------------

Training data generated in advance rarely covers all the configurations that an ML potential will
visit in its own molecular dynamics. ESTEEM can therefore build up the training set by *active
learning*, so that expensive ab initio calculations are spent only on the configurations where the
model is least reliable. A typical workflow runs as follows:

  1. Generate initial geometries, for example by MD with a MACE foundation model such as
     MACE-POLAR-1, or by classical MD.
  2. Select a random subset of these snapshots, and calculate their energies and forces with DFT
     and TDDFT (with the :mod:`esteem.tasks.clusters` module), for each electronic state of interest.
  3. Train a *committee* of ML models (for example 3 MACE models with different random seeds) for
     each state, using the :mod:`esteem.tasks.ml_training` module.
  4. Run ML molecular dynamics driven by each committee member (with the
     :mod:`esteem.tasks.ml_trajectories` module), with the whole committee evaluating each snapshot.
     The spread of the committee's predictions measures the uncertainty of the model, and a
     trajectory is stopped if the committee disagrees too strongly.
  5. Select new snapshots from these trajectories, biased towards high committee uncertainty, and
     calculate their DFT and TDDFT energies and forces.
  6. Test the current models on these new snapshots, which they have not seen before (with the
     :mod:`esteem.tasks.ml_testing` module), then train the next generation of models on all the
     data so far.

Steps 4 to 6 are repeated for several generations, until the models are accurate enough. The final
models can then be used for production MD, for example ground-state MD to sample the ensemble
followed by excited-state trajectories started from it.

Each generation of an active-learning cycle needs many similar tasks, for each state, committee
member and generation. The :mod:`esteem.active_learning` module creates all of these from a single
prototype of each task, and the example script ``examples/cate_ml.py`` shows how to set this up.
