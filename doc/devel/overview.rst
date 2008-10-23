.. _overview:

========
Overview
========


------------
Introduction
------------

This document describes the most important objects used for a DFT calculation.
More information can be found in the :epydoc:`API <gpaw>` or in the code.


.. _overview_array_naming:

----------------------------
Naming convention for arrays
----------------------------

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

--------------------------------
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
 
------------------------------------------------
Parallelization over spins, k-points and domains
------------------------------------------------

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
