"""Compute magnon dispersion on band path between all the high-symmetry
points of the hcp lattice. The resulting magnon energies are saved."""

# General modules
import numpy as np

# CAMD modules
from gpaw import GPAW
from gpaw.response.mft import IsotropicExchangeCalculator, \
    compute_magnon_energy_FM