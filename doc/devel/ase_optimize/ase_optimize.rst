.. _optimizer_tests:

===============
Optimizer tests
===============
This page shows benchmarks of optimizations done with our different optimizers.
Note that the iteration number (steps) is not the same as the number of force
evaluations. This is because some of the optimizers uses internal line searches
or similar.

The most important performance characteristics of an optimizer is the
total optimization time.
Different optimizers may perform the same number of steps, but along a different
path, so the time spent on calculation of energy/forces may be different
due to different convergence of the self-consistent field.


Test systems
============

.. csv-table::
   :file: systems.csv


EMT calculations
================

.. csv-table::
   :file: emt.csv


GPAW-LCAO calculations
======================

.. csv-table::
   :file: lcao.csv
