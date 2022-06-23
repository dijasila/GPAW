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

# ----- Ground state simulation ----- #

# Setup material
Co = bulk('Co','hcp',a=2.5071, c=4.0695)
mm = np.array([1.67, 1.67])
Co.set_initial_magnetic_moments(magmoms=mm)

# Prepare ground state calculator
kpts = {'size': (k, k, k), 'gamma': True}
calc = GPAW(xc='LDA',
            mode=PW(pw),
            kpts=kpts,
            nbands=nbands_gs,
            convergence=conv,
            occupations=FermiDirac(0.01),
            symmetry={'point_group': False},
            parallel={'domain': 1},
            spinpol=True,
            )

# Converge ground state
Co.set_calculator(calc)
Co.get_potential_energy()

# Save converged ground state with all plane-wave components
calc.write(f'converged_gs_calc.gpw', mode='all')