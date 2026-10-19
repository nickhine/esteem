.. _ezfcf:

ezFCF Wrapper
=============

.. currentmodule:: esteem.wrappers.ezfcf

The ezFCF wrapper calculates vibronic spectra with
`ezFCF <https://iopenshell.usc.edu/downloads/>`__ (formerly ezSpectrum), which
computes Franck-Condon factors between the vibrational states of two
electronic states, within either the parallel-mode approximation or including
Duschinsky rotations. It is a spectrum generation wrapper for the
:ref:`spectra` task. The wrapper writes an ezFCF XML input file from the
geometries and harmonic vibrational frequencies of the initial and target
states (for example from :ref:`solutes` calculations with ``vibrations``
enabled), runs ezFCF and reads in the resulting stick spectrum.

:Website: https://iopenshell.usc.edu/downloads/
:Citation: S. Gozem and A.I. Krylov, The ezSpectra suite: An easy-to-use toolkit for spectroscopy
   modeling, *WIREs Comput. Mol. Sci.* **12**, e1546 (2022).
   `doi:10.1002/wcms.1546 <https://doi.org/10.1002/wcms.1546>`__

   P. Wojcik, S. Gozem, V. Mozhayskiy, M. Mukherjee, C. Baker and A.I. Krylov, ezFCF,
   https://iopenshell.usc.edu/downloads

ezFCFWrapper
------------

.. autoclass:: ezFCFWrapper
   :members:
