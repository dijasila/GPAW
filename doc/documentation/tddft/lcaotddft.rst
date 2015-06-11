.. _lcaotddft:

================================
Time propagation TDDFT with LCAO
================================

This page documents the use of time-propagation TDDFT in :ref:`LCAO
mode <lcao>`.  Work in progress!!!!!!!!!!

TODO: add reference to the implementation articles

TODO: after which version the code works?

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

PoissonSolver
=============

The Poisson solver with default parameters uses zero boundary conditions on the cell boundaries.
To get converged Hartree potential, one often needs large vacuum sizes. However, in LCAO approach
large vacuum size is often unnecessary. Thus, to avoid using large vacuum sizes but get converged
potential, one can use two approaches 1) use multipole moment corrections or 2) solve Poisson 
equation on a extended grid. These two approaches are implemented in ExtendedPoissonSolver.

Multipole moment corrections
----------------------------

The boundary conditions can be improved by adding multipole moment corrections to the density so that
the corresponding multipoles of the density vanish. The potential of these corrections is added to
the obtained potential. For a reference of the method, see XXX.

This can be accomplished by following solver::

  from gpaw.poisson_extended import ExtendedPoissonSolver
  poissonsolver = ExtendedPoissonSolver(eps=eps,
                                        moment_corrections=4)

This corrects the 4 first multipole moments, i.e., s, p_x, p_y, and p_z type multipoles. The range of
multipoles can be changed by using moment_corrections=9 when in addition the multipoles d_xx, d_xy,
d_yy, d_yz, and d_zz are included.

This setting has been observed to work well for spherical-like metallic nanoparticles, but more complex
geometries require inclusion of high multipoles or multicenter multipole approach. For this, consider
the advanced syntax of the moment_corrections. The previous code snippet is equivalent to::

  from gpaw.poisson_extended import ExtendedPoissonSolver
  poissonsolver = ExtendedPoissonSolver(eps=eps,
                                        moment_corrections=[{'moms': range(4), 'center': None}])

Here moment_corrections is a list of dictionaries with following keywords: moms specifies the
considered multipole moments, e.g., range(4) equals to s, p_x, p_y, and p_z multipoles, and center
specifies the center of the added corrections in atomic units (None corresponds to the center of the
cell).

As an example, consider metallic nanoparticle dimer where the nanoparticle centers are at (x1, y1, z1) Å and
(x2, y2, z2) Å. In this the following settings for the poisson solver may be tried out::

  import numpy as np
  from ase.units import Bohr
  from gpaw.poisson_extended import ExtendedPoissonSolver
  moms = range(4)
  center1 = np.array([x1, y1, z1]) / Bohr
  center2 = np.array([x2, y2, z2]) / Bohr
  poissonsolver = ExtendedPoissonSolver(eps=eps,
                                        moment_corrections=[{'moms': moms, 'center': center1},
					                    {'moms': moms, 'center': center2}])

In general case with multiple centers, the calculation cell is divided into non-overlapping regions
determined by the given centers so that each point of space is associated to the closest center.
See `Voronoi diagrams <http://en.wikipedia.org/wiki/Voronoi_diagram>`_ for analogous illustration of
the partitioning of a plane.


Extended Poisson grid
---------------------

The multipole correction scheme is challenging for complex system geometries. For these cases, the
size of the grid used for solving the Poisson equation can be increased. The extended grid can be
defined as follows::

  from gpaw.poisson_extended import ExtendedPoissonSolver
  poissonsolver = ExtendedPoissonSolver(eps=eps,
                                        extendedgpts=(256, 256, 256))

This solves the Poisson equation on an extended fine grid of size (256, 256, 256). Important notes:

* **extendedgpts refers to the size of the fine grid**
* **use sizes that are divisible by high powers of 2 to accelerate the multigrid scheme**

As a consequence of the different grid that is used in the Hartree potential evaluation, the
Poisson solver neglects the given initial potential in function solve(). However, in most of
the cases an analogous behaviour is obtained by setting extendedhistory=True::

  from gpaw.poisson_extended import ExtendedPoissonSolver
  poissonsolver = ExtendedPoissonSolver(eps=eps,
                                        extendedgpts=(256, 256, 256),
					extendedhistory=True)

This means that the Poisson solver uses the previously calculated potential as an initial guess
in the potential calculation. The default value extendedhistory=False leads to significant
performance decrease in the potential evaluation during the SCF cycle and the time-propagation TDDFT.



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


References
==========

.. [#Kuisma2015]
   M. Kuisma, A. Sakko, T. P. Rossi, A. H. Larsen, J. Enkovaara, L. Lehtovaara, and T. T. Rantala, 
   Localized surface plasmon resonance in silver nanoparticles: Atomistic first-principles time-dependent
   density functional theory calculations,
   *Phys. Rev. B* **69**, 245419 (2004). `doi:10.1103/PhysRevB.91.115431 <http://dx.doi.org/10.1103/PhysRevB.91.115431>`_
