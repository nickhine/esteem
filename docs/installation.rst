Installation
============

ESTEEM is a python-based project, originally written within Jupyter notebooks, and is tightly integrated with the Atomic Simulation Environment (ASE). This integration means that a range of electronic structure, molecular dynamics and machine-learning packages with calculators available in ASE can be used by ESTEEM to perform its tasks. ASE functionality is accessed via Wrapper modules which create default settings for a package. These packages are each individually optional, though there is little that can be done without at least one MD code and one electronic structure code.

Apart from the calculators, the code has the following dependencies, which follow those of the current release of ASE (3.29.0):

    * Python, version 3.10 or above
    * ASE, version 3.23.0 or above (the ORCA wrapper uses the calculator profiles introduced in 3.23.0)
    * NumPy, version 1.21.6 or above
    * SciPy, version 1.8.1 or above
    * Matplotlib, version 3.5.2 or above, for graphical outputs

NumPy, SciPy and Matplotlib are installed automatically when ASE is installed with pip. The development version of ASE requires Python 3.11 or above, NumPy 1.24, SciPy 1.10 and Matplotlib 3.7.

Installing ESTEEM
-----------------

**Install ASE.**

ESTEEM does not currently install ASE for you, so install it first. It is highly advisable to use an up-to-date version of ASE. Version 3.23.0 is the oldest that will work with all of ESTEEM's wrappers, and the calculator interfaces of ASE change from version to version, so the latest release is the best choice::

   $ pip install --upgrade ase

There are more detailed instructions at the `ASE <https://wiki.fysik.dtu.dk/ase>`_ website.
The following commands can be used to verify that ASE and other dependencies can be loaded successfully::

   $ python3
   >>> import ase
   >>> import numpy
   >>> import scipy
   >>> import matplotlib

**Install ESTEEM from PyPI.**

The simplest way to install ESTEEM is with pip::

   $ pip install esteem

Note that the release on PyPI may lag behind the latest version of the code. To get the latest features, install from the git repository as described below.

**Install ESTEEM from the git repository.**

The latest version of the code is on `GitHub <https://github.com/nickhine/esteem>`_. The ``main`` branch contains the latest release, and the ``develop`` branch contains the latest development version. Clone the repository, starting in whatever path you wish to install to::

   $ git clone https://github.com/nickhine/esteem.git

This will create a directory called ``esteem``. Then install it with pip in "editable" mode, so that Python uses the code in this directory directly::

   $ cd esteem
   $ pip install -e .

Any updates to the code can then be downloaded with::

   $ git pull

with no need to reinstall.

**Check the installation.**

To check that ESTEEM can be found, start python and type the commands below, checking that the version and location listed are what you expect::

   >>> import esteem.drivers
   >>> print(esteem.__version__)
   >>> print(esteem.drivers.__file__)

**Install the codes you need.**

Finally, install the electronic structure, molecular dynamics or machine-learning codes that you intend to use. Some wrappers require only another python package (for example ``mace-torch`` for the MACE wrapper): if this is correctly installed the wrapper should work. Other wrappers run an external program, which needs to be available on your ``PATH`` or specified when you set up the wrapper. See :ref:`using-esteem` and the page for each wrapper in :ref:`wrappers` for more details.
