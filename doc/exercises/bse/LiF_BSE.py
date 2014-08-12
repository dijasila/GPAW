import numpy as np
from gpaw.response.bse import BSE
from ase.parallel import paropen
import sys

####################################################################################################
# BSE CALCULATION AND BSE SPECTRUM
####################################################################################################

w_grid = np.linspace(0.,15.,1001)

bse = BSE('LiF_fulldiag.gpw',
w=w_grid,
q=np.array([0.0001, 0., 0.]),
optical_limit=True,
ecut=30,
nbands=60,
eta=0.1,
kernel_file = 'LiF_W_qGG',   # It stores the four-points kernel used for building the two-particles Hamiltonian
txt='LiF_BSE_out.txt')

df_BSE = bse.get_dielectric_function()  # It returns the dielectric function calculated at the BSE level
