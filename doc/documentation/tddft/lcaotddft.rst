.. _lcaotddft:

================================
Time propagation TDDFT with LCAO
================================

This page documents the use of time-propagation TDDFT in :ref:`LCAO
mode <lcao>`.  Work in progress!!!!!!!!!!

Usage
=====

Create LCAOTDDFT object like a GPAW calculator::

 >>> from gpaw.lcaotddft import LCAOTDDFT
 >>> td_calc = LCAOTDDFT(setups={'Na':'1'}, basis='1.dzp', xc='oldLDA', h=0.3, nbands=1,
                         convergence={'density':1e-7},
                         poissonsolver=PoissonSolver(eps=1e-20, remove_moment=1+3+5))

Important points are:

 * Use always compatible setups and basis sets.
 * Grid spacing is only used to calculate the Hamiltonian matrix and therefore larger grid than usual can be used.
 * Completely unoccupied bands should be left out of the calculation, since they are not needed.
 * Convergence of density should be few orders of magnitude more accurate than in ground state calculations.
 * Convergence of poisson solver should be at least 1e-14, but 1e-20 does not hurt (this is the quadratic error).
 * One should use multipole corrected poisson solver in any TDDFT run. See PoissonSolver documentation about flags.

Perform a regular ground state calculation, the get the ground state wave functions::

 >>> atoms.set_calculator(td_calc)
 >>> atoms.get_potential_energy()

If you wish to save here, write the wave functions also::

 >>> td_calc.write('Na2.gpw', mode='all')

The calculation proceeds as in grid mode. We kick the system to x-direction and propagate with 10as time steps for 500 steps::

 >>> td_calc.kick([1e-5, 0.0, 0.0])
 >>> td_calc.propagate(10, 500, out='Na2.dm')

The spectrum is obtained in same manner, as in grid propagation mode.

Simple run script
=================

.. literalinclude:: lcaotddft.py


General notes about basis sets
==============================

In time-propagation LCAO-TDDFT, the basis sets are in even more crucial role than in a ground state LCAO 
calculation. It is required, that basis set can represent both the occupied (holes) and relevant 
unoccupied states (electrons) adequately. Custom basis sets for the time propagation should be generated 
according to ones need, and then benchmarked. For already benchmarked basis sets, click here.

In other words, ALWAYS, ALWAYS, benchmark results with respect to grid real time propagation code on a 
largest system possible. For example, one can create a prototype system, which consists of similar atom 
species with similar roles than in the parent system, but small enough to calculate with grid propagation 
mode. Example will be given in advanged tutorial.

Parallelization
===============

TODO.

=======================================================
Advanced tutorial - Plasmon resonance of Silver cluster
=======================================================

One should think what type of transitions is there of interest, and make sure that the basis set can 
represent such Kohn-Sham electron and hole wave functions. The first transitions in silver
cluster will be `5s \rightarrow 5p` like. We require 5p orbitals in the basis set, and thus, we must
generate a custom basis set.


==========================================
Advanced tutorial - large organic molecule
==========================================

General notes
=============

On large organic molecules, on large conjugated systems, there will`\pi \rightarrow \pi^*`, `\sigma 
\rightarrow \sigma^*`. These states consists of only the valence orbitals of carbon, and they are likely 
by quite similar few eV's below and above the fermi lavel. These is thus a reason to believe that these 
states are well described with hydrogen 1s and carbon 2s and 2p valence orbitals around the fermi level.

Here, we will calculate a small and a large organic molecule with lcao-tddft.

Induced density
===============

Here we will obtain the induced density.

Kohn-Sham decomposition of the transition density matrix
========================================================

Here we will analyse the origin of the transitions.
