from math import pi
import _pickle as pickle

import numpy as np
import pylab as plt

from ase.io import read
import ase.units as u

# Parameters
nkpts = 45
bands = slice(0, 8)
sigma = 'FB'
eta = 0.25

# Load DOS from file
base = 'dos_vs_energy_%s_BZ_%ux%u_bands_%u-%u_eta_%.1e' % \
       (sigma, nkpts, nkpts, bands.start, bands.stop, eta)
fd = open('%s.pckl' % base, 'rb')
cdis_c, energy_e, dos_cen, eps_nk = pickle.load(fd)
fd.close()

# Dirac point and DOS
E0 = np.max(eps_nk[3-bands.start])
dos_ce = np.sum(dos_cen, axis=-1)

# Pristine DOS
G_enk = (energy_e[:, np.newaxis, np.newaxis] - eps_nk[np.newaxis, :, :] + 1.j*eta)**(-1)
A_enk = -1 / pi * G_enk.imag
dos0_e = 1 / len(eps_nk[0]) * np.sum(A_enk, axis=(1,2))


# Shift energy axis
energy_e -= E0

plt.figure(1, (14, 6))

# Plot DOS
plt.axes([0.11, 0.18, 0.35, 0.76])
plt.plot(energy_e, dos0_e, 'k--', lw=2, zorder=11)
for dos_e in dos_ce:
    plt.plot(energy_e, dos_e, lw=2)

plt.xlim(-2.1, 2.1)
plt.ylim([0, 0.32])

plt.xticks([-2, -1, 0, 1, 2], fontsize=26)
plt.yticks(fontsize=26)

labels = ['pristine', ] + \
         [r'$c_\mathrm{dis}=%.1f$ %s' % (100*cdis, '%') for cdis in cdis_c]

plt.legend(labels, prop={'size': 25}, handlelength=1.2,
           handletextpad=0.4, loc='upper left', frameon=False)

plt.xlabel(r'Energy ($\mathrm{eV}$)', fontsize=32)
plt.ylabel(r'DOS ($\mathrm{eV}^{-1}$)', fontsize=32)


# Spectral function
base = 'spectral_vs_path_%s_BZ_%ux%u_bands_%u-%u_eta_%.1e' % \
       (sigma, nkpts, nkpts, bands.start, bands.stop, eta)
fd = open('%s.pckl' % base, 'rb')
cdis_c, energy_e, kpath_kc, Apath_cekn, epspath_nk, k_k, K_k = pickle.load(fd)
fd.close()

# Shift energies
energy_e -= E0
epspath_nk -= E0

# Spectral function for c_dis=5%
A_ek = np.sum(Apath_cekn[-2], axis=-1)

# Plot
plt.axes([0.60, 0.18, 0.35, 0.76])

# Unperturbed bands
for epspath_k in epspath_nk:
    plt.plot(k_k, epspath_k, 'r--', lw=2, zorder=11)

# Shift k-point axis for pcolormesh
diff = np.diff(k_k)
k_k[0] -= 0.5 * diff[0]
k_k[1:] -= 0.5 * diff
k_k = np.concatenate([k_k, [K_k[-1],]])
# Generate meshgrids
k_ek, e_ek = np.meshgrid(k_k, energy_e)

# Spectral function
plt.pcolormesh(k_ek, e_ek, A_ek, cmap='binary') # , offset_position='data', offsets=diff)

plt.xlim(K_k[0], K_k[-1])
plt.ylim([energy_e[0], energy_e[-1]])

point_names = [r'$M$', r'$\Gamma$', r'$K$', r'$M$']
plt.xticks(K_k, point_names, fontsize=32)
plt.yticks(fontsize=26)

ax = plt.gca()
ax.tick_params(axis='x', direction='out', pad=5)

plt.ylabel(r'Energy ($\mathrm{eV}$)', fontsize=32)

plt.savefig('graphene_vacancy_dos_spectral.png')    
         
plt.show()
