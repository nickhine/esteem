.. _spectra:

Spectra module
==============

The Spectra task turns the excitation energies and oscillator strengths of the
individual calculations into a UV/visible absorption spectrum. It reads
excitations from calculation output files or trajectories, which normally come
from the :ref:`clusters` task. It then broadens each stick spectrum with
Gaussians of width ``broad`` and averages over snapshots to give the spectrum
of the solvated system. It can also:

* apply a *spectral warp*: a shift and/or scaling of the excitation energies
  that maps a cheap level of theory onto a more accurate one. The warp
  parameters are fitted by comparing two :ref:`solutes` calculations of the
  same molecule (``warp_origin_files`` and ``warp_dest_files``).
* compute the perceived RGB colour of the solution from the spectrum, under a
  given illuminant.
* plot the result and write it to ``output``.

Vibronic spectra (via a spectrum generation wrapper such as
:ref:`MolSpeckPy <molspeckpy>`) are also supported.

.. currentmodule:: esteem.tasks.spectra

SpectraTask
-----------

.. autoclass:: SpectraTask
   :members:

Task Arguments
------------------

.. argparse::
   :module: esteem.tasks.spectra
   :func: get_parser
   :nodescription:

Standalone module routines
--------------------------

.. automodule:: esteem.tasks.spectra
   :exclude-members: SpectraTask
   :members:

