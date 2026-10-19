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

Task attributes
---------------

.. task-attributes:: esteem.tasks.spectra.SpectraTask

   wrapper: The wrapper used to process the excitations. An electronic structure wrapper (for example ``ORCAWrapper()``) reads the output files; a spectrum generation wrapper such as :ref:`MolSpeckPy <molspeckpy>` or :ref:`ezFCF <ezfcf>` calculates the lineshape.
   script_settings: Settings for the job scripts written for this task (see :ref:`parallel-execution`).

Standalone module routines
--------------------------

.. automodule:: esteem.tasks.spectra
   :exclude-members: SpectraTask
   :members:

