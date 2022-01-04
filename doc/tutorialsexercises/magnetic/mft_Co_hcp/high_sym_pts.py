"""Compute magnon energy of Co(hcp) at high symmetry points
of the Brillouin zone. Test different parameters of the
integration regions, which define magnetic sites.

Note : change 'sphere' to 'cylinder' in shapes_m to use a cylinder rather
than a sphere as integration region. Then rc is cylinder radius and height
equals diameter.

Note : specify sitePos_mv as an (N_sites, 3) array to set magnetic sites
manually. If N_sites != 2, then change length of shapes_m accordingly.
"""

# General modules
import numpy as np
import json

# CAMD modules
from gpaw import GPAW
from gpaw.response.mft import IsotropicExchangeCalculator, \
    compute_magnon_energy_FM
from ase.dft.kpoints import get_special_points

# ----- Setup exchange calculator ----- #

# Settings specific to exchange calculator
ecut = 400  # Energy cutoff in eV for response calc (number of G-vectors)
# How to map onto discrete lattice of magnetic moments
sitePos_mv = 'atoms'  # Magnetic sites at all atoms
shapes_m = ['sphere', 'sphere']  # Use spheres to define magnetic sites


# Pick radii of integration spheres to test (rc values)
rc_r = np.linspace(0.4, 1.9, 31)
rc_rm = np.array([rc_r, rc_r]).T

# Load converged ground state
calc = GPAW(f'converged_gs_calc.gpw', parallel={'domain': 1})
Co = calc.get_atoms()
mm = calc.get_magnetic_moments()

# Get number of converged bands
nbands_response = calc.parameters['convergence']['bands']

# Setup exchange calculator
exchCalc = IsotropicExchangeCalculator(calc,
                                       sitePos_mv,
                                       shapes_m=shapes_m,
                                       ecut=ecut,
                                       nbands=nbands_response)