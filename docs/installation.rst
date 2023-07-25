Installation
============

ESTEEM is a python-based project, originally written within Jupyter notebooks, and is tightly integrated with the Atomic Simulation Environment (ASE). This integration means that a range of electronic structure, molecular dynamics and machine-learning packages with calculators available in ASE can be used by ESTEEM to perform its tasks. ASE functionality is accessed via Wrapper modules which create default settings for a package. These packages are each individually optional, though there is little that can be done without at least one MD code and one electronic structure code.

Apart from the calculators, the code has the following dependencies:

    * Python, version 3.6 or above
    * ASE
    * NumPy
    * Matplotlib for graphical outputs

Compiling the Python files from the ipynb notebooks requires IPython (though this may be dropped in future).

Manual installation
-------------------

Currently the best way to install the code is to clone the git repository. In future I hope to make a PyPI package so that "pip install esteem" will work.

**Install ASE.**

It is highly advisable to use an up-to-date version of ASE. Many calculator bugs in relevant calculators have been fixed in versions 3.18-3.20 so older versions may well fail. This should be done before installing ESTEEM as currently ESTEEM will attempt to apply a tiny bugfix to ASE.

There are instructions at the `ASE <https://wiki.fysik.dtu.dk/ase>`_ website for installing ASE.
The following commands can be used to verify that ASE and other dependencies can be loaded successfully::

   $ python3
   >>> import ase
   >>> import numpy
   >>> import scipy
   >>> import matplotlib

**Clone the Bitbucket repository.**

ESTEEM is still evolving, so releases are not yet numbered. Versions can be identified by their commit string.

The latest version of the code can be downloaded via `the project's bitbucket page <https://bitbucket.org/ndmhine/esteem/>`_., but to ensure updates can be applied easily, it is better to clone the repository with git. Check out the code with::

   $ git clone https://ndmhine@bitbucket.org/ndmhine/esteem.git

Starting in whatever path you wish to install to. This will create a directory called 'esteem'.

**Convert notebooks into pure python**

After you have downloaded the code, the fastest way to do this conversion is via the 'setup.py' script::

    $ python setup.py

If that works, you should have a set of .py files in the esteem directory.

If the code changes, downloading updates can be achieved with::

    $ git pull

after which you need to compile the notebooks again with the setup.py script.

**Set the environment**

You need to let Python know where to find the ESTEEM modules.
The following line can be added to your '.bashrc' (or anywhere else that gets run automatically), with the appropriate path substituted for '~/' if required::

   $ export PYTHONPATH=~/esteem:$PYTHONPATH

To check this works, start python and type the below command, checking that the location listed 
by the second command is what you expect::

   >>> import esteem.drivers
   >>> print(esteem.drivers.__file__)

