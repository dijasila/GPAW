======
Muon SIte
======


Positive muons implanted in metals tend to stop at interstitial sites that correspond to the maxima of the Coulomb potential in the material. In turns the Coulomb potential is approximated by the Hartree pseudo-potential obtained from the GPAW calculation. A good guess is therefore given by the minima of this potential.
.
In this tutorial we obtain the guess in the case of MnSi. The results can be compared with A. Amato et al. [Amato14]_, who find a muon site at fractional cell coordinates (0.532,0.532,0.532).


MnSi calculation
================

Let's perform the calculation in ASE, starting from the space group of MnSi, 198, and the known Mn and Si coordinates.

.. literalinclude:: mnsi.py

The ASE code outputs a cube file with the volumetric data of the potential
that can be visualized.


Getting the minimum
===================

One way of identifying the maximum is by the use of an isosurface (or 3d contour surface) with a visualization program, like eg. VESTA.

This allows also secondary (local) minima to be identified.

In comparing with [Amato14]_ keep in mind that the present examples has a very low threshold and  

 

-------------

.. [Amato14]  A. Amato et al.
   Phys. Rev. B. 89, 184425 (2014)
   *Understanding  the μSR spectra of MnSi without magnetic polarons*
   DOI: 10.1103/PhysRevB.1.4555
