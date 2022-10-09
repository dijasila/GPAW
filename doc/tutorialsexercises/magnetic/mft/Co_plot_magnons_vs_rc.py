# web-page: Co_magnons_vs_rc.png
"""Plot the magnon energy as a function of the cutoff radius rc for all the
high-symmetry points of Co (hcp)"""

# General modules
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

# Script modules
from gpaw import GPAW
from gpaw.response.heisenberg import calculate_fm_magnon_energies

# ---------- Inputs ---------- #

# Ground state
gpw = 'Co.gpw'

# High-symmetry points
sp_p = [r'$\Gamma$', 'M', 'K', 'A']
spq_pc = [np.array(qc) for qc in
          [[0., 0., 0], [0.5, 0., 0.], [1 / 3., 1 / 3., 0.], [0., 0., 0.5]]]

# Load MFT data
q_qc = np.load('Co_q_qc.npy')
rc_r = np.load('Co_rc_r.npy')
# We study the case, where the two sublattices are equivalent
J_qabr = np.load('Co_Jsph2_qabr.npy')

# Labels and limits
rlabel = r'$r_{\mathrm{c}}\: [\mathrm{\AA}]$'
mwlabel = r'$\hbar\omega$ [meV]'
rlim = (0.4, 1.85)
mwlim = (200., 600.)

filename = 'Co_magnons_vs_rc.png'

# ---------- Script ---------- #

# Extract the magnetization of the unit cell
calc = GPAW(gpw, txt=None)
muc = calc.get_magnetic_moment()

# Calculate the magnon energies
# We distribute the unit cell magnetization evenly between the sites
mm_ar = muc / 2. * np.ones(J_qabr.shape[2:], dtype=float)
E_qnr = calculate_fm_magnon_energies(J_qabr, q_qc, mm_ar)

# We separate the acoustic and optical magnon modes by sorting them
E_qnr = np.sort(E_qnr, axis=1)

# Make a subplot for each magnon mode
fig, axes = plt.subplots(1, 2, constrained_layout=True)
colors = rcParams['axes.prop_cycle'].by_key()['color']

# Plot the magnon energies at the high-symmetry points as a function of rc
for p, (sp, spq_c) in enumerate(zip(sp_p, spq_pc)):
    q = 0
    while not np.allclose(q_qc[q], spq_c):
        q += 1
        if q == len(q_qc):
            raise ValueError
    for n in range(2):
        if n == 0 and p == 0:
            continue  # Do not plot the acoustic mode Gamma point
        E_r = E_qnr[q, n] * 1.e3  # eV -> meV
        axes[n].plot(rc_r, E_r, '-x',
                     color=colors[p], label=sp)

for n, (ax, mode) in enumerate(zip(axes, ['Acoustic', 'Optical'])):
    ax.set_title(mode)
    ax.set_xlabel(rlabel)
    ax.set_ylabel(mwlabel)
    ax.set_xlim(rlim)
    ax.set_ylim(mwlim)

    ax.legend()

plt.savefig(filename, format=filename.split('.')[-1],
            bbox_inches='tight')
