.. _overview:

========
Overview
========

This document describes the most important objects used for a DFT calculation.
More information can be found in the :epydoc:`API <gpaw>` or in the code.


PAW
===

This object is the central object for a GPAW calculation::

                    +---------------+
                    |ForceCalculator|      +-----------+
                    +---------------+  --->|Hamiltonian|
        +------+             ^        /    +-----------+
        |Setups|<--------    |    ----      +---------------+
        +------+         \   |   /     ---->|InputParameters|
     +-----+              +-----+     /     +---------------+     
     |Atoms|<-------------| PAW |-----      
     +-----+              +-----+     \          
                         /   |   \     \            +-----------+
      +-------------+   /    |    ---   ----------->|Occupations|
      |WaveFunctions|<--     v       \              +-----------+
      +-------------+     +-------+   \   +-------+    
                          |Density|    -->|SCFLoop|    
                          +-------+       +-------+

The implementation is in :svn:`gpaw/paw.py`.  The
:class:`~gpaw.paw.PAW` class doesn't do any part of the actual
calculation - it only handles the logic of parsing the input
parameters and setting up the neccesary objects for doing the actual
work (see figure above).


A PAW instance has the following attributes: :attr:`atoms`,
:attr:`input_parameters setups`, :attr:`wfs`, :attr:`density`,
:attr:`hamiltonian`, :attr:`scf`, :attr:`forces`, :attr:`timer`,
:attr:`occupations`, :attr:`initialized` and :attr:`observers`.



GPAW
====

The :class:`~gpaw.aseinterface.GPAW` class
(:svn:`gpaw/aseinterface.py`), which implements the :ase:`ASE calculator
interface <ase/calculators/calculators.html#calculator-interface>`,
inherits from the PAW class, and does unit conversion between GPAW's
internal atomic units (`\hbar=e=m=1`) and :ase:`ASE's units <ase/units.html>`
(Angstrom and eV)::

        gpaw          |    ase
  
  (Hartree and Bohr)  |  (eV and Angstrom)
               
     +-----+          |
     | PAW |
     +-----+          |
        ^
       /_\            |
        |
        |             |
     +------+           calc +-------+
     | GPAW |<---------------| Atoms |
     +------+                +-------+
                      |


Generating a GPAW instance from scratch
---------------------------------------

When a GPAW instance is created from scratch::

  calc = GPAW(xc='LDA', nbands=7)

the GPAW object is almost empty.  In order to start a calculation, one
will have to call the :meth:`~gpaw.PAW.calculate` method::

  calc.calculate(converge=True)

This will trigger:

1) A call to the :meth:`~gpaw.PAW.initialize` method, which will set
   up the objects needed for a calculation:
   :class:`gpaw.`:class:`gpaw.density.Density`,
   :class:`gpaw.hamiltonian.Hamiltonian`, :class:`gpaw.wfs.WaveFunctions`,
   :class:`gpaw.setup.Setups` and a few more (see figure above).

2) A call to the :meth:`~gpaw.PAW.set_positions` method, which will:

   a) Pass on the atomic positions to the wave functions, hamiltonian
      and density objects (call their ``set_positions()`` methods).
   
   b) Make sure the wave functions are initialized.

   c) Reset the :class:`~gpaw.scf.SCFLoop` and
      :class:`~gpaw.forces.ForceCalculator` objects.


Generating a GPAW instance from a restart file
----------------------------------------------

When a GPAW instance is created like this::

  calc = GPAW('restart.gpw')

the :meth:`~gpaw.PAW.initialize` method is called first, and then the
parts read from the file can be placed inside the objects where they
belong: the effective pseudo potential and the total energy are put in
the hamiltonian, the pseudo density is put in density object and so
on.

...


WaveFunctions
=============

We currently have two implementations of the 

::

     +--------------+     +-----------+
     |GridDescriptor|     |Eigensolver|
     +--------------+     +-----------+
                 ^           ^
                 |gd         |
                  \          |
   +------+        +-------------+ kpt_u   +------+
   |Setups|<-------|WaveFunctions|-------->|KPoint|+
   +------+        +-------------+         +------+|+
                          ^                 +------+|
                         /_\                 +------+
                          |
                          |
               -----------^--------------------
              |                                |
     +-----------------+            +-----------------+
     |LCAOWaveFunctions|            |GridWaveFunctions|
     +-----------------+            +-----------------+
           |        |              /    |           |
           v        |tci          |     |kin        |pt
   +--------------+ |             v     |           v
   |BasisFunctions| |        +-------+  |         +----------+
   +--------------+ |        |Overlap|  |         |Projectors|
                    v        +-------+  |         +----------+
     +------------------+               v                             
     |TwoCenterIntegrals|     +---------------------+         
     +------------------+     |KineticEnergyOperator|         
                              +---------------------+         




.. _overview_array_naming:

Naming convention for arrays
============================

A few examples:

 =========== =================== ===========================================
 name        shape    
 =========== =================== ===========================================
 ``spos_c``  ``(3,)``            **S**\ caled **pos**\ ition vector
 ``nt_sG``   ``(2, 24, 24, 24)`` Pseudo-density array
                                 :math:`\tilde{n}_\sigma(\vec{r})`
                                 (``t`` means *tilde*):
                                 two spins, 24*24*24 grid points.
 ``cell_cv`` ``(3, 3)``          Unit cell vectors.
 =========== =================== ===========================================

 =======  ==================================================
 index    description
 =======  ==================================================
 ``a``    Atom number
 ``c``    Unit cell axis-index (0, 1, 2)
 ``v``    *xyz*-index (0, 1, 2)                                    
 ``k``    **k**-point index                                   
 ``s``    Spin index (:math:`\sigma`)                           
 ``u``    Combined spin and **k**-point index 
 ``G``    Three indices into the coarse 3D grid                     
 ``g``    Three indices into the fine 3D grid                     
 ``n``    Principal quantum number *or* band number        
 ``l``    Angular momentum quantum number (s, p, d, ...)
 ``m``    Magnetic quantum number (0, 1, ..., 2*l - 1)         
 ``L``    ``l`` and ``m`` (``L = l**2 + m``)                                
 ``j``    Valence orbital number (``n`` and ``l``)               
 ``i``    Valence orbital number (``n``, ``l`` and ``m``)            
 ``q``    ``j1`` and ``j2`` pair                                 
 ``p``    ``i1`` and ``i2`` pair
 ``r``    CPU-rank
 =======  ==================================================


Array names and their definition
--------------------------------

 ================  ==================================================
 name in the code  definition
 ================  ==================================================
 nucleus.P_uni     eq. (6) in [1]_ and eq. (6.7) in [2]_
 nucleus.D_sp      eq. (5) in [1]_ and eq. (6.18) in [2]_
 nucleus.H_sp      eq. (6.82) in [2]_
 setup.Delta_pL    eq. (15) in [1]_
 setup.M_pp        eq. (C2,C3) in [1]_ and eq. (6.48c) in [2]_
 ================  ==================================================
 

Parallelization over spins, k-points and domains
================================================

When using parallization over spins, **k**-points and domains,
three different MPI communicators are used:

* *mpi.world*
   Communicator containing all processors. 
* *domain_comm*
   One *domain_comm* communicator contains the whole real space 
   domain for a selection of the spin/k-point pairs.
* *kpt_comm* 
   One *kpt_comm* communicator contains all k-points and spin 
   for a part of the real space domain.

For the case of a :math:`\Gamma`-point calculation all parallel communication
is done in the one *domain_comm* communicator, which are in this case 
equal to *mpi.world*. 

.. [1] J J. Mortensen and L. B. Hansen and K. W. Jacobsen,
       Phys. Rev. B 71 (2005) 035109.
.. [2] C. Rostgaard, Masters thesis, CAMP, dep. of physics, Denmark, 2006.
       This document can be found at the :ref:`exx` page.
