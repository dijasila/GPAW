import _pickle as pickle

import numpy as np

from ase.io import read
from ase.utils import opencew
from ase.dft import monkhorst_pack

from gpaw import GPAW

from gpaw1.lcao.fourier import Fourier
from gpaw1.defects.defect import Defect


# Use an odd number divicible by 3 in order to include K,K' points
nkpts = 45
bands = slice(0, 8)
nbands = bands.stop - bands.start

# Atoms
atoms = read('graphene.traj')

# k-point grid
kpts_kc = monkhorst_pack((nkpts, nkpts, 1))

############################################
#              Calculate wfs
############################################
# Use lcao/fourier module to interpolate onto arbitrary k-points
calc = GPAW('calc_lcao.gpw')
calc_ft = Fourier(calc)
eps_kn, coef_knM = calc_ft.diagonalize(kpts_kc, return_wfs=True)

# Slice out bands
c_knM = coef_knM[:, bands]
e_kn = eps_kn[:, bands]

# Dump bands and states to file for later use
c_nkM = c_knM.transpose(1, 0, 2)
e_nk = e_kn.T
base = 'graphene_lcao_BZ_%ux%u_bands_%u-%u' % \
       (nkpts, nkpts, bands.start, bands.stop)
fd = open('%s.pckl' % base, 'wb')
pickle.dump([kpts_kc, e_nk, c_nkM], fd)
fd.close()


############################################
#       Calculate matrix elements
############################################
basis_lcao = 'DZP'
N = 5
cutoff = 4.00
fname = 'vacancy_%ux%u' % (N, N)

calc_def = Defect(atoms, calc=None, supercell=(N, N, 1), defect=None,
                  name=fname, pckldir='.')
# Load supercell and apply cutoff
calc_def.load_supercell_matrix('%s(dzp)' % basis_lcao.lower(), cutoff=cutoff,
                               pos0=atoms.positions[0])

# Calculate matrix elements: hermitian=True: take advantage of the fact that
# the matrix is hermitian -> only upper triangle is calculated
V_kknn = calc_def.bloch_matrix(kpts_kc, c_knM, hermitian=True)

# Rearrange matrix elements
V_nknk = np.ascontiguousarray(V_kknn.transpose(2, 0, 3, 1))
 
# Check for hermitian
# hermitian = np.allclose(V_nknk, V_nknk.conj().transpose(2,3,0,1))
# assert hermitian, "Matrix not hermitian"
assert V_nknk.flags['C_CONTIGUOUS']

# Dump matrix elements to file -- use numpy format as matrices can get very
# large
base = 'vacancy_V_BZ_%ux%u_bands_%u-%u' % \
       (nkpts, nkpts, bands.start, bands.stop)
fd = open('./pickle/%s.npy' % base, 'wb')
np.save(fd, V_nknk, allow_pickle=False)
fd.close()
