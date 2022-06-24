"""Plot the magnon dispersion of Co(hcp)"""

# General modules
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

# Script modules
from gpaw import GPAW
from gpaw.response.heisenberg import (calculate_fm_magnon_energies,
                                      calculate_single_site_magnon_energies)


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
Jsph1_qabr = np.load('Co_Jsph1_qabr.npy')
Jsph2_qabr = np.load('Co_Jsph2_qabr.npy')
Jmix_qp = np.load('Co_Jmix_qp.npy')

# Define range of spherical radii to plot
rmin = 0.9
rmax = 1.5

# Define a radius for the "best estimate"
rbest = 1.2

# Define a radius to plot the inequivalent spherical site model for
rineq = 0.6

# Labels and limits
mwlabel = r'$\hbar\omega$ [meV]'
mwlim = (0., 600.)

filename = 'Co_magnon_dispersion.png'


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
# We distribute the unit cell magnetization evenly between the sites
mm_ar = muc / 2. * np.ones(Jsph1_qabr.shape[2:], dtype=float)
Esph1_qnr = calculate_fm_magnon_energies(Jsph1_qabr, q_qc, mm_ar) * 1.e3  # meV
Esph2_qnr = calculate_fm_magnon_energies(Jsph2_qabr, q_qc, mm_ar) * 1.e3
Emix_qp = calculate_single_site_magnon_energies(Jmix_qp, q_qc, muc) * 1.e3

# We separate the two magnon modes by sorting them
Esph1_qnr = np.sort(Esph1_qnr, axis=1)
Esph2_qnr = np.sort(Esph2_qnr, axis=1)

# First, we plot the magnon dispersion calculated with equivalent spherical
# sublattice site kernels
colors = rcParams['axes.prop_cycle'].by_key()['color']
# We define the lower bound on the magnon energy as the minimum within the
# chosen rc range and the upper bound as the maximum value
Evalid_qnr = Esph2_qnr[..., np.logical_and(rc_r >= rmin, rc_r <= rmax)]
Emin_qn = np.min(Evalid_qnr, axis=2)
Emax_qn = np.max(Evalid_qnr, axis=2)
Ebest_qn = Esph2_qnr[..., np.where(np.abs(rc_r - rbest) < 1.e-8)[0][0]]
# Plot one mode at a time
for n in range(2):
    plt.fill_between(pathq_q, Emin_qn[:, n], Emax_qn[:, n],
                     color=colors[0], alpha=0.4)
    plt.plot(pathq_q, Emin_qn[n], color='0.5')
    plt.plot(pathq_q, Emax_qn[n], color='0.5')
    if n == 0:
        label = 'eq. spheres'
    else:
        label = None
    plt.plot(pathq_q, Ebest_qn[n], color=colors[0], label=label)

# Secondly, we plot the dispersion with inequivalent spherical sites
Eineq_qn = Esph2_qnr[..., np.where(np.abs(rc_r - rineq) < 1.e-8)[0][0]]
plt.plot(pathq_q, Eineq_qn[:, 0], color=colors[1], label='ineq. spheres')
plt.plot(pathq_q, Eineq_qn[:, 1], color=colors[1])

# Lastly plot the magnon dispersion with only a single site in the unit cell
plt.plot(pathq_q, Emix_qp[:, 0], '-o',
         mec='k', color=colors[2], label='unit cell')
plt.plot(pathq_q, Emix_qp[:, 1], '-^',
         mec='k', color=colors[3], label='cylinder')

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
