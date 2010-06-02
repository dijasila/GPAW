"""Density functional perturbation theory (DFPT) package.

DFPT
====

In DFPT the first-order variation in the density due to a static perturbation
in the external potential is calculated self-consistently from a set of coupled
equations for
  1. the variation of the effective potential,
  2. variation of the wave functions, and
  3. the resulting variation of the density.

The variation of the wave-functions can be determined from a set of linear
equations, the so-called Sternheimer equation. Since only their projection onto
the unoccupied state manifold is required to determine the corresponding
density variation, only the occupied wave functions/states of the unperturbed
system are needed. This is in contrast to the standard textbook expression for
the first-order variation of the wave function which involves the full spectrum
of the unpertured system.

The first-order variation of the density with respect to different
perturbations can be used to obtain various physical quantities of the
system. This package includes calculators for:

    1. Phonons in periodic systems
        - perturbation is a lattice distortion with a given q-vector (work in
          progress).
    2. Born effective charges
        - perturbation is a constant electric field (to be done).
    3. Dielectric constant
        - perturbation is a constant electric field (to be done).


References
----------
  - Louie et al., Phys. Rev. B 76, 165108 (2007)
  - Baroni et al., Rev. Mod. Phys. 73, 515 (2001)
  - Gonze, Phys. Rev. B 55, 10337 (1997)
  - Gonze et al., Phys. Rev. B 55, 10355 (1997)
  - Savrasov, Phys. Rev. B 54, 16470 (1996)
  - Savrasov et al., Phys. Rev. B 54, 16487 (1996)

Implementaion notes
===================
1. When using the ``derivative`` method of the ``lfc`` class to calculate
   integrals between derivatives of localized functions wrt atomic
   displacements and functions defined on the grid, the signs of the calculated
   integrals come out incorrectly (see doc string of the ``derivative``
   method). All files in this ``dfpt`` package follows the prescription:: 

        X_niv = -1 * X_aniv[a] ,

   for correcting the sign error, i.e. correct the sign when extracting from
   the dictionary.

2. More to come

"""

# __version__ = "0.1"

# Import things specified in the __all__ attributes
from gpaw.dfpt.phononcalculator import *
from gpaw.dfpt.responsecalculator import *
from gpaw.dfpt.phononperturbation import *

# Set the behavior of from gpaw.dfpt import *
import gpaw.dfpt.phononcalculator
import gpaw.dfpt.responsecalculator
import gpaw.dfpt.phononperturbation

__all__ = []
__all__.extend(phononcalculator.__all__)
__all__.extend(responsecalculator.__all__)
__all__.extend(phononperturbation.__all__)
