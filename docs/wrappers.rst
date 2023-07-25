.. _wrappers:

Wrappers available in ESTEEM
============================

.. toctree::
   :maxdepth: 1
   :caption: Available Wrappers:

   amber.rst
   lammps.rst
   nwchem.rst
   onetep.rst
   orca.rst
   physnet.rst
   amp.rst
   molspeckpy.rst
   ezfcf.rst

A wrapper in ESTEEM is an intermediate layer between a task and ASE calculator.
It allows the task to call upon a number of different electronic structure codes,
without needing to store details of how to invoke each of them within the module
for that task.

The idea is that all the standard calculator setup for a particular theoretical
spectroscopy task can be hidden inside the wrapper, and the functions of the wrapper,
which might include "singlepoint" energy evaluations, geometry optimisation,
or calculation of electronic excitations, can then be passed into the task
as arguments.

The wrappers fall into three categories at present: electronic structure wrappers
(currently NWChem and ONETEP), Molecular Dynamics wrappers (currently Amber
and LAMMPS) and Machine Learning wrappers (currently PhysNet and AMP).
Details of these wrappers can be found on their individual pages.

If you have a new code you would like ESTEEM to be able to use, then adding a new
wrapper for it should be fairly easy, as long as there is already an ASE calculator
for your code. The wrapper is simply a set of default settings for a typical
run - effectively a mapping of arguments passed to the task, to input varibles
passed to the calculator. The arguments might mean very different things to
different calculators: for example the "basis" argument when passed to NWChem is
a single string representing a basis set such as "6-31G*", whereas when passed
to ONETEP it is a tuple specifying cutoff energy, NGWF radius, and conduction
band energy range.

Molecular dynamics wrappers
---------------------------

Need to specify functions for: ``singlepoint``, ``minimise``, ``heatup``, ``densityeq``, ``equil``, ``snapshots``.

Electronic Structure wrappers
-----------------------------

Need to specify functions for: ``singlepoint``, ``geom_opt``, ``excit``, ``freq``.
They may optionally also provide ``run_qmd``, ``read_excit`` and ``read_freq``.

Machine Learning wrappers
-------------------------

Need to specify functions for: ``train``, ``load``, ``geom_opt``, ``run_mlmd``.

Spectrum Generation wrappers
----------------------------

Need to specify functions for ``write_excitations``, ``write_input_file``, ``results_exist`` and ``run``.

Other Wrappers
==============

Writing a wrapper is a pretty easy task for anyone with ASE and general python skills.
Any electronic structure, MD or Machine-Learning package that already has an ASE
interface can probably be given an appropriate wrapper within a day or so's work.

I am very happy to accept contributions of new wrappers to the code. Please have a look at
the code most similar package to what you want to add and adapt it as required.

Future Plans
------------

Upon request, if presented with a use case, I could probably find time to add wrappers for
any of the following codes, some of which I am reasonably familiar with already:

CASTEP, QuantumEspresso, SchNarc, CP2K, BigDFT, Gaussian.

Get in touch with me at n.d.m.hine@warwick.ac.uk to discuss the possibilities.
