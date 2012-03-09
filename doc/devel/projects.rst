================
Ongoing Projects
================

.. contents::


New PAW setups
==============

:Who:
    Marcin and Jens Jørgen.

The goal is to produce new PAW setup for all elements.  Compared to our
old collection of PAW setups we will focus on:

* higher accuracy - more semi-core states might be added
* reduced eggbox error
* faster convergence of energy with number of grid-points - if possible
* tesing the setups more carfully agains a bigger set of all-electron results

The code is in :svn:`~gpaw/gpaw/atom/generator2.py` and it is based on
a new and more robust atomic solver: :svn:`~gpaw/gpaw/atom/aeatom.py`.


Improving the RMM-DIIS eigensolver
==================================

Currently, our :ref:`RMM-DIIS eigensolver <RMM-DIIS>` will always take
two steps for each state.  In an attempt to make the eigensolver
faster and more robust, we should investigate the effect of taking a
variable number of steps depending on the change in the eigenstate
error and occupations number as described in [Kresse]_.



Improved density mixing
=======================

For density mixing in real-space, we use a special metric (described
:ref:`here <densitymix>`) for measuring input to output density
changes with more weight on long wavelength changes.  We will try to
do the density mixing in reciprocal space, where the metric is easier
to express and a wave length dependent preconditioning can also easily
be applied.


Plane wave basis
================

:Who:
    Jens Jørgen

A plane wave implementation is already in trunk.  It's based on FFTW
and does the projector wave function overlaps in reciprocal space with
BLAS's ZGEMM.

Missing features:

* Meta-GGA
* Dipole layer correction
* Parallelization over states works only when number of states is
  divisible by number of processors
* and maybe more ...


Calculation of stress tensor
============================

:Who:
    Ask and Ivano

The plan is to implement this in the plane-wave part of the code,
where all the terms are relatively simple.  So far, we have coded the
kinetic energy contribution.


.. [Kresse] G. Kresse, J. Furthmüller:
   Phys. Rev. B 54, 11169 - 11186 (1996)
   "Efficient iterative schemes for ab initio total-energy calculations
   using a plane-wave basis set"
