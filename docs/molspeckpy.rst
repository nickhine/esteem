.. _molspeckpy:

MolSpeckPy Wrapper
==================

.. currentmodule:: esteem.wrappers.specpycode

The MolSpeckPy wrapper calculates optical spectra with
`MolSpeckPy <https://github.com/tjz21/Spectroscopy_python_code>`__, a Python
code for simulating linear and nonlinear spectroscopy. It is a spectrum
generation wrapper for the :ref:`spectra` task. Rather than broadening the
excitations of each snapshot independently, MolSpeckPy uses the fluctuations
of the excitation energies along an MD trajectory to calculate the spectrum
with approaches such as the cumulant expansion, which captures vibronic
effects and the dynamics of the environment. The wrapper writes the
excitation energies from ESTEEM trajectories and a MolSpeckPy input file, runs
MolSpeckPy (from the directory given by ``exec_path``) and reads in the
resulting spectrum.

:Source code: https://github.com/tjz21/Spectroscopy_python_code
:Citation: T.J. Zuehlsdorff, A. Montoya-Castillo, J.A. Napoli, T.E. Markland and C.M. Isborn,
   Optical spectra in the condensed phase: Capturing anharmonic and vibronic features using dynamic
   and static approaches, *J. Chem. Phys.* **151**, 074111 (2019).
   `doi:10.1063/1.5114818 <https://doi.org/10.1063/1.5114818>`__

   The MolSpeckPy README lists further papers to cite for specific methods.

SpecPyCodeWrapper
-----------------

.. autoclass:: SpecPyCodeWrapper
   :members:
