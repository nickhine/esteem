.. _nwchem:

NWChem Wrapper
==============

.. currentmodule:: esteem.wrappers.nwchem

The NWChem wrapper runs DFT and TDDFT calculations with
`NWChem <https://nwchemgit.github.io>`__, an open-source quantum chemistry
package, through ASE's NWChem calculator. It provides singlepoint energies and
forces, geometry optimisation, vibrational frequencies, TDDFT excitations and
ab initio molecular dynamics, with implicit solvent through the COSMO model. It
can be used for the :ref:`solutes`, :ref:`clusters` and
:ref:`qmd_trajectories` tasks.

:Website: https://nwchemgit.github.io
:Source code: https://github.com/nwchemgit/nwchem
:Citation: E. Aprà *et al.*, NWChem: Past, present, and future, *J. Chem. Phys.* **152**, 184102 (2020).
   `doi:10.1063/5.0004997 <https://doi.org/10.1063/5.0004997>`__

NWChemWrapper
-------------

.. autoclass:: NWChemWrapper
   :members:
