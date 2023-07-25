.. ESTEEM documentation master file, created by
   sphinx-quickstart on Tue Aug 18 23:47:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ESTEEM: Explicit Solvent Toolkit for Electronic Excitations of Molecules
========================================================================

ESTEEM is an open source package for calculations of excited state and spectroscopic properties of molecules in solvent environments, developed by `Nicholas Hine <https://warwick.ac.uk/fac/sci/physics/staff/academic/nicholashine/>`_ at the University of Warwick.

It is built upon the python framework provided by the `Atomic Simulation Environment <https://wiki.fysik.dtu.dk/ase/index.html>`_ and can call upon a number of different electronic structure and molecular dynamics codes, and machine-learning techniques.

The project is still under active development but is feature-complete and has some level of documentation (improving all the time). The code’s `git repository <https://bitbucket.org/ndmhine/esteem>`_ on bitbucket contains the latest development version and can be used to report any issues.

The Explicit Solvent methodology that has been adapted to a workflow in ESTEEM is based on the ideas described in the papers below.
If you find this project useful, I would appreciate if you cite these works:

   M.A.P. Turner, M.D. Horbury, V.G. Stavros, N.D.M. Hine, Determination of secondary species in solution through pump-selective transient absorption spectroscopy and explicit-solvent TDDFT, `J. Phys. Chem. A 123, 873 (2019). <https://pubs.acs.org/doi/10.1021/acs.jpca.8b11013>`_
   
   T.J. Zuehlsdorff, P.D. Haynes, M.C. Payne, and N.D.M. Hine, Predicting solvatochromic shifts and colours of a solvated organic dye: The example of Nile Red, `J. Chem. Phys. 146, 124504 (2017) <https://aip.scitation.org/doi/10.1063/1.4979196>`_

ESTEEM is installable via pip: installation should be as simple as::

   $ pip3 install esteem

Acknowledgements: Some parts of ESTEEM are based on python scripting by members of my research group including Tim Zuehlsdorff, David Turban, Matt Turner, Carlo Maino and Panos Kourtis.

**Manual**:

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   introduction.rst
   installation.rst
   using_esteem.rst
   parallel_execution.rst
   examples.rst
   drivers.rst
   solutes.rst
   solvate.rst
   clusters.rst
   spectra.rst
   trajectories.rst
   qmd_trajectories.rst
   ml_training.rst
   ml_testing.rst
   ml_trajectories.rst
   wrappers.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
