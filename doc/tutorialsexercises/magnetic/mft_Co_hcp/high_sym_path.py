"""Compute magnon dispersion on band path between all the high-symmetry
points of the hcp lattice. The resulting magnon energies are saved."""

# General modules
import numpy as np

# CAMD modules
from gpaw import GPAW
from gpaw.response.mft import IsotropicExchangeCalculator, \
    compute_magnon_energy_FM

# ----- Setup exchange calculator ----- #

# Settings specific to exchange calculator
ecut = 400  # Energy cutoff in eV for response calc (number of G-vectors)
# How to map onto discrete lattice of magnetic moments
sitePos_mv = 'atoms'  # Magnetic sites at all atoms
shapes_m = ['sphere', 'sphere']  # Use spheres to define magnetic sites
rcBest = 1.3  # Pick optimal rc value

# Load converged ground state
calc = GPAW(f'converged_gs_calc.gpw', parallel={'domain': 1})
Co = calc.get_atoms()
mm = calc.get_magnetic_moments()

# Get number of converged bands
nbands_response = calc.parameters['convergence']['bands']

# Get size of k-space simulation grid
k = calc.parameters['kpts']['size'][0]

# Setup exchange calculator
exchCalc = IsotropicExchangeCalculator(calc,
                                       sitePos_mv,
                                       shapes_m=shapes_m,
                                       ecut=ecut,
                                       nbands=nbands_response)