.. _onetep:

ONETEP Wrapper
==============

.. currentmodule:: esteem.wrappers.onetep

The ONETEP wrapper runs DFT and TDDFT calculations with
`ONETEP <https://onetep.org>`__, a linear-scaling DFT code whose cost grows
only linearly with the number of atoms. This makes it well suited to
calculating the excitations of large explicit-solvent clusters, of hundreds or
thousands of atoms, in the :ref:`clusters` task. It provides singlepoint
energies, geometry optimisation and linear-response TDDFT excitations, with
implicit solvent around the cluster through ONETEP's minimal-parameter solvent
model.

The ``basis`` argument has a different meaning for ONETEP than for Gaussian
basis set codes: it specifies the cutoff energy, NGWF radius and conduction
NGWF energy range. Use ``setup`` to set the ONETEP executable and the location
of the pseudopotentials.

:Website: https://onetep.org
:Documentation: https://docs.onetep.org
:Citation: J.C.A. Prentice *et al.*, The ONETEP linear-scaling density functional theory program,
   *J. Chem. Phys.* **152**, 174111 (2020).
   `doi:10.1063/5.0004445 <https://doi.org/10.1063/5.0004445>`__

   For TDDFT calculations, also cite: T.J. Zuehlsdorff, N.D.M. Hine, J.S. Spencer, N.M. Harrison,
   D.J. Riley and P.D. Haynes, Linear-scaling time-dependent density-functional theory in the linear
   response formalism, *J. Chem. Phys.* **139**, 064104 (2013).
   `doi:10.1063/1.4817330 <https://doi.org/10.1063/1.4817330>`__

OnetepWrapper
-------------

.. autoclass:: OnetepWrapper
   :members:
