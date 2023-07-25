.. _drivers:

Task drivers in ESTEEM
======================

The task drivers in ESTEEM provide a means of looping over multiple sets of
arguments for a given task, such that a set of calculations can be performed
consistently over, for example, many combinations of molecule and solvent,
or many different temperatures, or a range of cluster sizes.

There is also a main driver, ``drivers.main()``, which determines which of
the other drivers should be run, based on the command-line arguments. Invoking
this main driver is the standard way of using ESTEEM in scripts.

However, for more advanced functionality, you can call the other drivers manually
from your scripts. This page documents the main driver and the individual
task drivers.


.. automodule:: esteem.drivers
   :members:
