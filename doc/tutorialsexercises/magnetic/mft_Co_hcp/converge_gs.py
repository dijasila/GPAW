"""Converge ground state of Co(hcp) and save with all plane-wave components

A number of unoccupied bands are converged for the sake of subsequent
response calculations.

Note : k must be a multiple of 6 in order to do computations at all the
high-symmetry points of the hcp lattice.

Note : Co(hcp) has 2 atoms in the unit cell, each with 9 valence electrons
which fully or partially occupy 6 bands per atom. Hence there are 12 occupied
bands, so nbands_gs >= nbands_response > 12 is required"""

# General modules
import numpy as np

# CAMD modules
from gpaw import GPAW, PW, FermiDirac
from ase.build import bulk

# ----- Select ground state simulation settings ----- #
xc = 'LDA'              # Exchange-correlation potential
k = 12                  # kpts in each direction of simulation grid
pw = 1200               # Plane wave energy cutoff in eV
nbands_gs = 21          # Bands for ground state calculation
nbands_response = 18    # Bands for response calculation
conv = {'density': 1.e-8,
        'forces': 1.e-8,
        'bands': nbands_response}  # Converge bands used in response calc
