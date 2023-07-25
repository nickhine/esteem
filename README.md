ESTEEM: Explicit Solvent Toolkit for Electronic Excitations in Molecules
========================================================================

## Contents of package:

A python package that uses the Atomic Simulation Environment python
packages, and a range of optional calculators that ASE can make
available, to perform calculations of excited states of explicitly-solvated
solute molecules.

## Dependencies:

* Python 3
* ASE (tested on 3.19.0)

## Codes which can be interfaced with (all optional):
* NWChem >=6.6 (tested on 6.6, 6.8, 7.0)
* ORCA >= 5.0.2
* AMBER (tested on AmberTools 16.0 and 20.0)
* LAMMPS
* ONETEP (tested on 5.3.x and 6.0)
* PhysNet
* MACE
* AMP
* SpecPyCode
* EZFCF

## License:

This package is distributed under the MIT License.

## Setup:

Install the package via pip:

pip install esteem

Write a script that imports esteem tasks and wrappers (or adapt something from
/examples)

Run the script, specifying the task, seedname and task_target

python my_script.py <task> <seedname> <task_target>

eg

python cate.py solutes cate gs_PBE

The script should invoke the main drivers routine once the tasks are set up.

Some wrappers require only another python package: if this is correctly
installed the code should work. Other wrappers require a binary, and
in most cases this will need to be specified - see the instructions on 
each wrapper for more details.

