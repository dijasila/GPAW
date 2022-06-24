"""Plot the magnon dispersion of Fe(bcc)"""

# General modules
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

# Script modules
from gpaw import GPAW
from gpaw.response.heisenberg import calculate_single_site_magnon_energies

# ---------- Inputs ---------- #

# Ground state
gpw = 'Fe_all.gpw'

# High-symmetry points
sp_p = [r'$\Gamma$', 'N', 'P', 'H']
spq_pc = [np.array(qc) for qc in
          [[0., 0., 0], [0., 0., 0.5], [0.25, 0.25, 0.25], [0.5, -0.5, 0.5]]]

# Load MFT data
q_qc = np.load('Fe_q_qc.npy')
rc_r = np.load('Fe_rc_r.npy')
J_qr = np.load('Fe_J_qr.npy')
Juc_q = np.load('Fe_Juc_q.npy')

# Define range of spherical radii to plot
rmin = 0.9
rmax = 1.5

# Labels and limits
mwlabel = r'$\hbar\omega$ [meV]'
mwlim = (0., 450.)

filename = 'Fe_magnon_dispersion.png'

# ---------- Script ---------- #

# Extract the magnetization of the unit cell
calc = GPAW(gpw, txt=None)
muc = calc.get_magnetic_moment()

# Convert relative q-points into distance along the bandpath in reciprocal
# space.
B_cv = 2.0 * np.pi * calc.atoms.cell.reciprocal()  # Coordinate transform
q_qv = q_qc @ B_cv  # Transform into absolute reciprocal coordinates
pathq_q = [0.]
for q in range(1, len(q_qc)):
    pathq_q.append(pathq_q[-1] + np.linalg.norm(q_qc[q] - q_qc[q - 1]))
pathq_q = np.array(pathq_q)

# Define q-limits of plot
qlim = ((pathq_q[0] - pathq_q[1]) / 2.,
        (1.5 * pathq_q[-1] - 0.5 * pathq_q[-2]))

# Calculate the magnon energies
E_qr = calculate_single_site_magnon_energies(J_qr, q_qc, muc) * 1.e3  # meV
Euc_q = calculate_single_site_magnon_energies(Juc_q, q_qc, muc) * 1.e3

# We define the lower bound on the magnon energy as the minimum within the
# chosen rc range, the "best estimate" as the median, and the upper bound
# as the maximum value
Evalid_qr = E_qr[:, np.logical_and(rc_r >= rmin, rc_r <= rmax)]
Emin_q = np.min(Evalid_qr, axis=1)
E_q = np.median(Evalid_qr, axis=1)
Emax_q = np.max(Evalid_qr, axis=1)

# Plot the magnon dispersion with spherical sites
colors = rcParams['axes.prop_cycle'].by_key()['color']
plt.fill_between(pathq_q, Emin_q, Emax_q, color=colors[0], alpha=0.4)
plt.plot(pathq_q, Emin_q, color='0.5')
plt.plot(pathq_q, Emax_q, color='0.5')
plt.plot(pathq_q, E_q, color=colors[0], label='spherical')

# Plot the magnon dispersion with parallelepipedic sites
plt.plot(pathq_q, Euc_q, '-o', mec='k', color=colors[1], label='unit cell')

# Use high-symmetry points as tickmarks for the x-axis
qticks = []
qticklabels = []
for sp, spq_c in zip(sp_p, spq_pc):
    for q, q_c in enumerate(q_qc):
        if np.allclose(q_c, spq_c):
            qticks.append(pathq_q[q])
            qticklabels.append(sp)
plt.xticks(qticks, qticklabels)
# Plot also vertical lines for each special point
for pq in qticks:
    plt.axvline(pq, color='0.5', linewidth=1, zorder=0)

# Labels and limits
plt.ylabel(mwlabel)
plt.xlim(qlim)
plt.ylim(mwlim)

plt.legend(title='Site geometry')

plt.savefig(filename, format=filename.split('.')[-1],
            bbox_inches='tight')
