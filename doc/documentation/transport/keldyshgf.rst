.. _keldyshgf:

.. default-role:: math

=======================
Keldysh Green functions
=======================

The Keldysh Green function (KGF) code allows for calculations
of non-equilibrium transport calculations where electron
exchange and correlation effect are threated using many body pertubation
theory such as Hartree-Fock, second Born and the GW approximation.
It is recommended that you go through the ASE/GPAW electron transport exercice
to get familiar with the general transport setup and definitions used 
in ase and gpaw and the KGF code.

-------------------------
Download and Installation
-------------------------

The KGF code is currently beeing merged into the development version of
GPAW and is expected to be part of the GPAW package in the near future.
The latest revision can be obtained from svn::

  $ svn checkout https://svn.fysik.dtu.dk/projects/KeldyshGF/trunk/KeldyshGF KeldyshGF
 

Installation is completed by adding the path of KeldyshGF to the PYTHONPATH 
environment variable.

-----------------------
Doing a KGF calculation
-----------------------


Pariser-Parr-Model Hamiltonian
------------------------------

To do an electron transport calculation using a model Hamiltonian
the parameters of both the non-interacting
part as well as the interacting part of the Hamiltonian need to 
be explicitly specified. The non-interacting part 
h_ij describe kinetic energy and electron-electron interaction 
part in the PPP approximation 
is on the form V_ij = v_iijj, where the v_ijkl's are two electron
Coulomb integrals.
To get started consider a simple four site interacting
model. The four the x's in the figure below represent the sites where 
electron-electron interactions are included.
The o's (dashes) represents non-interacting sites.::

   Left lead   Molecule   Right Lead
   --------------------------------- 
   o o o o x| x  x  x  x |x o o o o
   ---------------------------------
           0  1  2  3  4  5

The numbers refers to indix numbers in the Green functions 
- the Green function will be a 6x6 matrix where the subspace corresponding
to the molecule will be the central 4x4 matrix.
Leads are treated as simple nearest neighbour tight-binding chains with
a principal layer size of one.

The following parameters will be used to simulate a physisorbed molecule:

=================  =========    ==============================
parameter	   value         description
=================  =========    ==============================
``t_ll``           -20.0        intra lead hopping
``t_lm``           -1.0         lead-molecule hopping
``t_mm``           -2.4         intra molecule hopping
``V``                           electron-electron interaction 
=================  =========    ==============================

where V is the matrix:: 
     
      V = [[  0.     7.45   4.54   3.18   2.42   0.  ]
           [  7.45  11.26   7.45   4.54   3.18   2.42]
           [  4.54   7.45  11.26   7.45   4.54   3.18]
           [  3.18   4.54   7.45  11.26   7.45   4.54]
           [  2.42   3.18   4.54   7.45  11.26   7.45]
           [  0.     2.42   3.18   4.54   7.45   0.  ]]

In Python code the input parameters can generated like this:

.. literalinclude:: parms.py

We begin by performing an equilibrium calculation (zero bias).
An equilibrium involces setting the relevant Green's functions and
self-energies. All Green's functions are represented on energy grid 
which should have a grid spacing fine enough to resolve all spectreal
feautres. In practise this accomplished by choosing an energy grid spacing
about half the size of the infinitesimal eta appearing in the 
Green's functions (which is given a finite value
in numerical calculations). 
 
In Python code an equilibrium non-interacting calculatoins followed
by a Hartree-Fock calculations and a GW calculation look like this:

.. literalinclude:: GW_eq.py


Self-consistency
-----------------
The self-consistent solution is obtained by mixing Green's function
using a pulay mixing scheme, which is controllod by three parameters
(convergence criteria , mixing weight : float, history length : int).
The Hartree-Fock and GW self-consistent iterations is written to the
logfiles files HF.log and GW.log, respectively. is written to a NetCDFFile

.. default-role::
