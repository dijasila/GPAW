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