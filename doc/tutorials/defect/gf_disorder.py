from math import pi
import _pickle as pickle

import numpy as np

from ase.io import read
from ase.dft.kpoints import ibz_points

from gpaw1.defects.tmatrix import TMatrix
from gpaw1.defects.bzgrid_extract_path import extract_path

# Parameters
nkpts = 45
bands = slice(0, 8)

# Disorder concentrations
cdis_c = 1e-2 * np.array([0.1, 1.0, 5.0])
# Self-energy approximation: FB=T-matrix
sigma = 'FB'
# Numerical broadening (eV)
eta = 0.25
# Solver for T-matrix equation (linear or inversion)
solver = 'linear'

# Energy axis
Emax = +2.5  # Range: E_Dirac +/- Emax
de = 50e-3   # Spacing

# Atoms
atoms = read('graphene.traj')
cell_cv = atoms.cell

############################################
#     Load wfs/bands and matrix elements
############################################
# wfs and energies
fd = open('./graphene_lcao_BZ_%ux%u_bands_%u-%u.pckl' % \
         (nkpts, nkpts, bands.start, bands.stop), 'rb')
kpts_kc, eps_nk, c_nkM = pickle.load(fd)
fd.close()

# matrix elements
fd = open('./pickle/vacancy_V_BZ_%ux%u_bands_%u-%u.npy' % \
          (nkpts, nkpts, bands.start, bands.stop), 'rb')
V_nknk = np.load(fd, allow_pickle=False)
fd.close()

# Create energy axis
# Dirac point
E0 = np.max(eps_nk[3-bands.start])
nepoints = int(2*Emax / de) + 1
energy_e = np.linspace(E0 - Emax, E0 + Emax, nepoints)

# T matrix class
tmatrix = TMatrix(kpts_kc, eps_nk, V_nknk, dk=None)

# Calculate GF
G_cekn, T_enk = tmatrix.gf_disavg(energy_e, cdis_c, eta=eta,
                                  sigma=sigma, solver=solver)

# DOS (band resolved)
dos_cen = tmatrix.dos_disavg(GF=G_cekn)

# Dump to file
base = 'dos_vs_energy_%s_BZ_%ux%u_bands_%u-%u_eta_%.1e' % \
       (sigma, nkpts, nkpts, bands.start, bands.stop, eta)
fd = open('%s.pckl' % base, 'wb')
pickle.dump([cdis_c, energy_e, dos_cen, eps_nk], fd)
fd.close()

# Spectral function along path
A_cekn = -1. / pi * G_cekn.imag


# Extract path from BZ grid
path = 'MGKM'
npoints = 500
i_k, k_k, K_k = extract_path(kpts_kc, path, cell_cv, npoints=npoints)

kpath_kc = kpts_kc[i_k]
epspath_nk = eps_nk[:, i_k]

# Slice out path from arrays
Apath_cekn = A_cekn[:, :, i_k]

# Dump spectral to file
base = 'spectral_vs_path_%s_BZ_%ux%u_bands_%u-%u_eta_%.1e' % \
       (sigma, nkpts, nkpts, bands.start, bands.stop, eta)
fd = open('%s.pckl' % base, 'wb')
pickle.dump([cdis_c, energy_e, kpath_kc, Apath_cekn, epspath_nk, k_k, K_k], fd)
fd.close()
