======
Muon SIte
======

Positive muons implanted in metals tend to stop at interstitial sites that correspond to the maxima of the Coulomb potential energy for electrons in the material. In turns the Coulomb potential is approximated by the Hartree pseudo-potential obtained from the GPAW calculation. A good guess is therefore given by the maxima of this potential.

In this tutorial we obtain the guess in the case of MnSi. The results can be compared with A. Amato et al. [Amato14]_, who find a muon site at fractional cell coordinates (0.532,0.532,0.532) by DFT calculations and by the analysis of experiments.


MnSi calculation
================

Let's perform the calculation in ASE, starting from the space group of MnSi, 198, and the known Mn and Si coordinates.

.. literalinclude:: mnsi.py

The ASE code outputs a cube file with volumetric data of the potential that can be visualized.


Getting the minimum
===================

One way of identifying the maximum is by the use of an isosurface (or 3d contour surface) at a slightly lower value than the maximum, by meaans of an external visualization program, like eg. VESTA.

This allows also secondary (local) minima to be identified.

A simplified procedure to identify the global maximum is the following

.. literalinclude:: mnsicontour.py

The figure below shows the contour plot of the potential in a plane containing the maximum

.. image:: mnsicontour.png

In comparing with [Amato14]_ keep in mind that the present examples has a very reduced number of k points and a low plane wave cutoff energy, just enough to show the right extrema in the shortest CPU time. 
 

-------------

.. [Amato14]  A. Amato et al.
   Phys. Rev. B. 89, 184425 (2014)
   *Understanding  the μSR spectra of MnSi without magnetic polarons*
   DOI: 10.1103/PhysRevB.1.4555
