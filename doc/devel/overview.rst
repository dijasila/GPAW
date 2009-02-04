.. _overview:

========
Overview
========

.. default-role:: math


This document describes the most important objects used for a DFT calculation.
More information can be found in the :epydoc:`API <gpaw>` or in the code.


PAW
===

This object is the central object for a GPAW calculation::

                    +---------------+
                    |ForceCalculator|      +-----------+
                    +---------------+  --->|Hamiltonian|
                             ^        /    +-----------+
                             |    ----      +---------------+
                             |   /     ---->|InputParameters|
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
:attr:`input_parameters`, :attr:`wfs`, :attr:`density`,
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
will have to call the :meth:`~gpaw.paw.PAW.calculate` method::

  calc.calculate(converge=True)

This will trigger:

1) A call to the :meth:`~gpaw.paw.PAW.initialize` method, which will
   set up the objects needed for a calculation:
   :class:`~gpaw.density.Density`,
   :class:`~gpaw.hamiltonian.Hamiltonian`,
   :class:`~gpaw.wavefunctions.WaveFunctions`,
   :class:`~gpaw.setup.Setups` and a few more (see figure above).

2) A call to the :meth:`~gpaw.paw.PAW.set_positions` method, which will
   initialize everything that depends on the atomic positions:

   a) Pass on the atomic positions to the wave functions, hamiltonian
      and density objects (call their ``set_positions()`` methods).
   
   b) Make sure the wave functions are initialized.

   c) Reset the :class:`~gpaw.scf.SCFLoop` and
      :class:`~gpaw.forces.ForceCalculator` objects.




Generating a GPAW instance from a restart file
----------------------------------------------

When a GPAW instance is created like this::

  calc = GPAW('restart.gpw')

the :meth:`~gpaw.paw.PAW.initialize` method is called first, so that the
parts read from the file can be placed inside the objects where they
belong: the effective pseudo potential and the total energy are put in
the hamiltonian, the pseudo density is put in the density object and so
on.

After a restart, everything *should* be as before the restart file was
written.  However, there are a few exceptions:

* The wave functions are only read when needed ... XXX

* Atom centered functions (`\tilde{p}_i^a`, `\bar{v}^a`,
  `\tilde{n}_c^a` and `\hat{g}_{\ell m}^a`) are not
  initialized. ... XXX




WaveFunctions
=============

We currently have two representations for the wave functions: uniform
3-d grids and expansions in atom centered basis functions as
implemented in the two classes
:class:`~gpaw.wavefunctions.GridWaveFunctions` and
:class:`~gpaw.wavefunctions.LCAOWaveFunctions`.  Both inherit from the
:class:`~gpaw.wavefunctions.WaveFunctions` class, so the wave
functions object will always have a
:class:`~gpaw.grid_descriptor.GridDescriptor`, an
:class:`~gpaw.eigensolvers.eigensolver.Eigensolver`, a
:class:`~gpaw.setup.Setups` object and a list of :class:`~gpaw.kpoint.KPoint`
objects.

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
               --------------------------------
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

Attributes of the wave function object: :attr:`gd`, :attr:`nspins`,
:attr:`nbands`, :attr:`mynbands`, :attr:`dtype`, :attr:`world`,
:attr:`kpt_comm`, :attr:`band_comm`, :attr:`gamma`, :attr:`bzk_kc`,
:attr:`ibzk_kc`, :attr:`weight_k`, :attr:`symmetry`, :attr:`kpt_comm`,
:attr:`rank_a`, :attr:`nibzkpts`, :attr:`kpt_u`, :attr:`ibzk_qc`,
:attr:`eigensolver` and :attr:`timer`.
        


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


Commonly used indices:

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


.. list-table::

   * - name in the code
     - definition
   * - wfs.kpt_u[u].P_ani
     - `\langle\tilde{p}_i^a|\tilde{\psi}_{\sigma\mathbf{k}n} \rangle`
   * - density.D_asp
     - `D_{s i_1i_2}^a`
   * - hamiltonian.dH_sp
     - `\Delta H_{s i_1i_2}^a`
   * - setup.Delta_pL
     - `\Delta_{Li_1i_2}^a`
   * - setup.M_pp
     - `\Delta C_{i_1i_2i_3i_4}^a` eq. (C2) in [1]_ or eq. (47) in [2]_
   * - wfs.kpt_u[u].psit_nG
     - `\tilde{\psi}_{\sigma\mathbf{k}n}(\mathbf{r})`
   * - setup.pt_j
     - `\tilde{p}_j^a(r)`
   * - wfs.pt
     - `\tilde{p}_i^a(\mathbf{r}-\mathbf{R}^a)`




Parallelization over spins, k-points domains and states
=======================================================

XXX

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
.. [2] C. Rostgaard, `The Projector Augmented Wave Method <../paw_note.pdf>`_.



.. default-role::
