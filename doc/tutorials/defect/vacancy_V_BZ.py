import _pickle as pickle

import numpy as np

from ase.io import read
from ase.utils import opencew
from ase.dft import monkhorst_pack

from gpaw import GPAW

from gpaw1.lcao.fourier import Fourier
from gpaw1.defects.defect import Defect


# Use a number divicible by 3 in order to include K,K' points
nkpts = 15
bands = slice(0, 8)
nbands = bands.stop - bands.start

# Atoms
atoms = read('graphene.traj')

# k-point grid
kpts_kc = monkhorst_pack((nkpts, nkpts, 1))

# Calculate wfs
# Use lcao/fourier module to interpolate onto arbitrary k-points
calc = GPAW('calc_lcao.gpw')
calc_ft = Fourier(calc)
e_nk, c_knM = calc_ft.diagonalize(kpts_kc, return_wfs=True)

# Slice out LCAO coefficient for specified bands
c_kn = np.ascontiguousarray(c_knM[:, bands])

############################################
#            Defect calculation
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

# Calculate matrix elements: hermitian=True -> only upper triangle wrt the
# k-point index is calculated
V_kknn = calc_def.bloch_matrix(kpts_kc, c_kn, hermitian=True)

# Reorder matrix elements
V_nknk = np.ascontiguousarray(V_kknn.transpose(2, 0, 3, 1))
 
# Check for hermitian
hermitian = np.allclose(V_nknk, V_nknk.conj().transpose(2,3,0,1))
assert hermitian, "Matrix not hermitian"
assert V_nknk.flags['C_CONTIGUOUS']

# Dump matrix elements to file -- use numpy format as matrices can be very
# large 
base = 'vacancy_V_BZ_%ux%u_bands_%u-%u' % \
       (nkpts, nkpts, bands.start, bands.stop)
fd = open('./pickle/%s.npy' % base, 'wb')
np.save(fd, V_nknk, allow_pickle=False)
fd.close()

# Dump to unperturbed bands to file
base = 'graphene_bands_BZ_%ux%u_bands_%u-%u' % \
       (nkpts, nkpts, bands.start, bands.stop)
fd = open('%s.pckl' % base, 'wb')
pickle.dump([kpts_kc, e_nk], fd)
fd.close()
