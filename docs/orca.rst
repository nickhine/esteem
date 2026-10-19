.. _orca:

ORCA Wrapper
============

.. currentmodule:: esteem.wrappers.orca

The ORCA wrapper runs DFT and TDDFT calculations with
`ORCA <https://www.faccts.de/orca/>`__, a quantum chemistry package that is
free for academic use. It works through ASE's ORCA calculator, and requires
ORCA 5.0.2 or later. It provides singlepoint energies and forces, geometry
optimisation, TDDFT excitations and ab initio molecular dynamics, with implicit solvent through the CPCM model. It can be
used for the :ref:`solutes`, :ref:`clusters` and :ref:`qmd_trajectories`
tasks.

Use ``setup`` to set the number of processes and memory per process, and set
``orcaprofile`` to an ASE ``OrcaProfile`` giving the full path of the ORCA
executable (ORCA needs to be run with its full path for parallel runs).

:Website: https://www.faccts.de/orca/
:Citation: F. Neese, Software Update: The ORCA Program System—Version 6.0,
   *WIREs Comput. Mol. Sci.* **15**, e70019 (2025).
   `doi:10.1002/wcms.70019 <https://doi.org/10.1002/wcms.70019>`__

   F. Neese, The ORCA program system, *WIREs Comput. Mol. Sci.* **2**, 73 (2012).
   `doi:10.1002/wcms.81 <https://doi.org/10.1002/wcms.81>`__

   If you use ORCA 5, cite instead: F. Neese, Software update: The ORCA program
   system—Version 5.0, *WIREs Comput. Mol. Sci.* **12**, e1606 (2022).
   `doi:10.1002/wcms.1606 <https://doi.org/10.1002/wcms.1606>`__

   ORCA also lists the papers for the methods you have used at the end of each output file.

ORCAWrapper
-----------

.. autoclass:: ORCAWrapper
   :members:
