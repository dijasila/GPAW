from math import pi
import pickle

import numpy as np
import numpy.linalg as la

import pylab as plt

from ase.io import read
from ase.dft.kpoints import ibz_points

from gpaw import GPAW

from gpaw1.defects.defect import Defect
from gpaw1.lcao.fourier import Fourier

save = False

# Parameters
nkpts = 100           # k-point grid
ebands = slice(3, 5)  # Electronic bands (valence + conduction)
nbands = ebands.stop - ebands.start

# Atoms
atoms = read('graphene.traj')

# Unit cell (Ang)
cell_cv = atoms.cell
# Cell of reciprocal lattice
reci_vc = 2 * pi * la.inv(cell_cv)

# Square k-point grid that covers the BZ
a_ = la.norm(reci_vc[:, 0])
b = a_ * 0.66
k_v = np.linspace(-b, b, nkpts)
kx_vv, ky_vv = np.meshgrid(k_v, k_v)

# List of k points (in array)
kpts_kv = np.array([kx_vv.ravel(), ky_vv.ravel(),
                    np.zeros_like(kx_vv.ravel())]).transpose()
# Convert to units of reciprocal lattice vectors
kpts_kc = np.dot(kpts_kv, la.inv(reci_vc).T)

# High-symmetry points in the Brillouin zone
points = ibz_points['hexagonal']
K = np.array(points['K'])
M = np.array(points['M'])

# Initial state: displace from K point (degenerate bands!)
k0_c = K + 0.01 * (M-K)
# Add k0 point to list of k points
kpts_kc = np.concatenate([k0_c.reshape(1, 3), kpts_kc])


# Use lcao/fourier module to interpolate onto arbitrary k-points
calc = GPAW('calc_lcao.gpw')
calc_ft = Fourier(calc)
e_nk, c_knM = calc_ft.diagonalize(kpts_kc, return_wfs=True)

# Slice out LCAO coefficient for specified bands
c_kn = np.ascontiguousarray(c_knM[:, ebands])

# Defect calculation
basis = 'DZP'
N = 5
cutoff = 4.0

# Defect calculator
fname = 'vacancy_%ux%u' % (N, N)
calc_def = Defect(atoms, calc=None, supercell=(N, N, 1), defect=None,
                  name=fname, pckldir='.')
# Load supercell matrix
calc_def.load_supercell_matrix('%s(dzp)' % basis.lower(), cutoff=cutoff,
                               pos0=atoms.positions[0])
# Calculate matrix elements
V_kknn = calc_def.bloch_matrix(kpts_kc, c_kn, kpts_from=[0])
# Extract elements and reshape
V0_vvnn = V_kknn[0, 1:].reshape(len(kx_vv), len(ky_vv), nbands, nbands)

# Band element for plotting
ind1 = 1 ; ind2 = 1
Vnn = np.abs(V0_vvnn[:, :, ind1, ind2])

if save:
    base = 'vacancy_V_vs_k'
    f = open('%s.pckl' % base, 'wb')
    pickle.dump([kx_vv, ky_vv, V0_vvnn], f)
    f.close()

# Brillouin zone
K_kc = [[-1./3, 1./3, 0],
        [1./3, 2./3, 0],
        [2./3, 1./3, 0],
        [1./3, -1./3, 0],
        [-1./3, -2./3, 0],
        [-2./3, -1./3, 0],
        [-1./3, 1./3, 0],
        ]
# Convert to cartesian coordinates
K_kv = np.dot(K_kc, reci_vc.T)
k0_v = np.dot(k0_c, reci_vc.T)

# Plot - convert to (2pi/a) coordinates
a = la.norm(cell_cv[0])
kx_vv /= (2*pi / a)
ky_vv /= (2*pi / a)
K_kv /= (2*pi / a)
k0_v /= (2*pi / a)


plt.figure(1, (8, 6))
plt.axes([.18, .18, .76, .76])
# BZ
plt.plot(K_kv[:, 0], K_kv[:, 1], 'k', lw=2)

# Mark initial state
plt.plot([k0_v[0],], [k0_v[1],], 'o', markerfacecolor='w',
         markersize=8, zorder=11)
plt.text(0.975*k0_v[0], 1.025*k0_v[1], r'$\mathbf{k}$', fontsize=30, color='w',
         va='bottom', ha='left')


# Matrix element
p1 = plt.imshow(Vnn, origin='lower', interpolation='nearest',
                extent=(kx_vv[0, 0], kx_vv[0, -1],
                        ky_vv[0, 0], ky_vv[-1, 0]), cmap='jet')

# Color limits
V_max = np.max(Vnn)
V_min = np.min(Vnn)
p1.set_clim([0, V_max])
    
# Colorbar
tick_pos = np.linspace(0, V_max, 4)
c = plt.colorbar(ticks=tick_pos)
c.ax.set_yticklabels(['%2i' % t for t in tick_pos])

# Add contour plot
plt.contour(kx_vv, ky_vv, Vnn, 10, colors='k')

plt.xlim([kx_vv.min(), kx_vv.max()])
plt.ylim([ky_vv.min(), ky_vv.max()])

plt.xticks([-0.6, -0.3, 0, 0.3, 0.6], fontsize=26)
plt.yticks([-0.6, -0.3, 0, 0.3, 0.6], fontsize=26)

plt.xlabel(r"$k_x'$ ($2\pi / a$)", fontsize=32)
plt.ylabel(r"$k_y'$ ($2\pi / a$)", fontsize=32)
c.set_label(r'$\mathrm{eV}$', fontsize=32)
    
for t in c.ax.get_yticklabels():
    t.set_fontsize(28)

plt.savefig('vacancy_V_vs_k.png')
    
