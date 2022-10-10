.. _abinitiomd:

=====================================
ab initio molecular dynamics (DFT/MD)
=====================================

Ab initio molecular dynamics uses DFT to calculate the forces between the atoms 
at each time step. While computationally expensive, prohibiting simulations longer
than a few ps, DFT/MD can be directly applied to any system that DFT can describe.

-----------------------------------------
Transmission of hydrogen through graphene
-----------------------------------------

Neutral atom transmission through a graphene target can be simulated with DFT/MD.
However, note that TDDFT/MD is required to account for electronic stopping 
(Ref. [#Brand2019]_).

The following script simulates the impact of a hydrogen atom with an initial
velocity corresponding to a kinetic energy of 40 keV, transmitting through the
center of a hexagon in a graphene target.

In a realistic calculation, one might have to change the default convergence 
parameters depending on the projectile used, and to verify the convergence of 
the results with respect to the timestep and *k*-points. Here, slightly less 
strict criteria are used. The impact point in this case is the center of a 
carbon hexagon, but this can be modified by changing the x-y position of the 
H atom (``projpos``).

.. literalinclude:: graphene_h_md.py


----------
References
----------

.. [#Brand2019] C. Brand, M. Debiossac, T. Susi, F. Aguillon, J. Kotakoski,
                P. Roncin, M. Arndt, "Coherent diffraction of hydrogen 
                through the 246 pm lattice of graphene",
                *New J. Phys.* **21**, 033004 (2019).
