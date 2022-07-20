# creates: Fe_magnons_vs_rc.png
"""Plot the magnon energy as a function of the cutoff radius rc for all
high-symmetry points of Fe (bcc)"""

# General modules
import numpy as np
import matplotlib.pyplot as plt

# Script modules
from gpaw import GPAW
from gpaw.response.heisenberg import calculate_single_site_magnon_energies

# ---------- Inputs ---------- #

# Ground state
gpw = 'Fe_all.gpw'

# High-symmetry points
sp_p = ['N', 'P', 'H']
spq_pc = [np.array(qc) for qc in
          [[0., 0., 0.5], [0.25, 0.25, 0.25], [0.5, -0.5, 0.5]]]

# Load MFT data
q_qc = np.load('Fe_q_qc.npy')
rc_r = np.load('Fe_rc_r.npy')
J_qr = np.load('Fe_J_qr.npy')

# Labels and limits
rlabel = r'$r_{\mathrm{c}}\: [\mathrm{\AA}]$'
mwlabel = r'$\hbar\omega$ [meV]'
rlim = (0.4, 1.85)
mwlim = (200., 600.)

filename = 'Fe_magnons_vs_rc.png'

# ---------- Script ---------- #

# Extract the magnetization of the unit cell
calc = GPAW(gpw, txt=None)
muc = calc.get_magnetic_moment()

# Calculate the magnon energies
E_qr = calculate_single_site_magnon_energies(J_qr, q_qc, muc)

# Plot the magnon energies at the high-symmetry points as a function of rc
for sp, spq_c in zip(sp_p, spq_pc):
    q = 0
    while not np.allclose(q_qc[q], spq_c):
        q += 1
        if q == len(q_qc):
            raise ValueError
    E_r = E_qr[q] * 1.e3  # eV -> meV
    plt.plot(rc_r, E_r, '-x', label=sp)

plt.xlabel(rlabel)
plt.ylabel(mwlabel)
plt.xlim(rlim)
plt.ylim(mwlim)

plt.legend()

plt.savefig(filename, format=filename.split('.')[-1],
            bbox_inches='tight')
