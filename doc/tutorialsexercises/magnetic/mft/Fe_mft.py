"""Compute the isotropic exchange constants of Fe on a band path between all
the high-symmetry points of the bcc lattice."""

# General modules
import numpy as np

# Script modules
from gpaw.response.site_kernels import (SphericalSiteKernels,
                                        ParallelepipedicSiteKernels)
from gpaw.response.chiks import ChiKS
from gpaw.response.mft import IsotropicExchangeCalculator


# ---------- Inputs ---------- #

# Parameters reused from the ground state
gs = 'Fe_all.gpw'
kpts = 24
nbands = 6

# We choose the plane wave energy cutoff of the mft response calculation
# so as to provide magnon energies converged within 5%, based on the
# convergence study in [arXiv:2204.04169]
ecut = 1000  # eV
# We compute the transverse magnetic susceptibility without broadening
eta = 0.

# We map out the high-symmetry path G-N-P-G-H, by generating all commensurate
# q-vectors on the path
qGN_qc = np.array([[0., 0., x / kpts]
                   for x in range(kpts // 2 + 1)])
qNP_qc = np.array([[x / kpts, x / kpts, 1 / 2. - x / kpts]
                   for x in range(kpts // 4 + 1)])
qPG_qc = np.array([[x / kpts, x / kpts, x / kpts]
                   for x in reversed(range(kpts // 4 + 1))])
qGH_qc = np.array([[x / kpts, -x / kpts, x / kpts]
                   for x in range(kpts // 2 + 1)])
q_qc = np.vstack([qGN_qc, qNP_qc[1:], qPG_qc[1:], qGH_qc[1:]])

# Define the Fe site radii to try for the spherical site kernels
rc_r = np.linspace(0.5, 1.75, 51)

# ---------- Script ---------- #

# Initialize the ChiKS calculator, which is responsible for computing the
# transverse magnetic susceptibility of the Kohn-Sham system
chiks = ChiKS(gs,
              ecut=ecut, nbands=nbands, eta=eta, txt='Fe_chiks.txt')
# When initialized from a file, the ChiKS calculator has a serial copy of
# the ground state calculator. From it, we extract the atoms
atoms = chiks.calc.atoms

# Initialize the exchange calculator
isoexch_calc = IsotropicExchangeCalculator(chiks)

# Initialize the site kernels
positions = atoms.positions  # sublattice positions
# We give multiple spherical radii to the site kernels instance by specifying
# different partitionings of space into magnetic sublattices. In the case of
# Fe, we only have a single sublattice
rc_pa = [[rc] for rc in rc_r]  # p: partition, a: sublattice index
sitekernels = SphericalSiteKernels(positions, rc_pa)
# We may also try to use the entire unit cell for the sublattice site kernel.
# To do this, we use the parallelepipedic site kernel and use the bcc unit
# cell as the parallelepipedic cell
cell_pav = [[atoms.get_cell()]]
ucsitekernels = ParallelepipedicSiteKernels(positions, cell_pav)

# Allocate arrays for the exchange constants
nq = len(q_qc)
npartitions = sitekernels.npartitions
J_qabp = np.empty((nq, 1, 1, npartitions), dtype=complex)
Juc_qabp = np.empty((nq, 1, 1, 1), dtype=complex)

# Compute the isotropic exchange coupling along the chosen bandpath
for q, q_c in enumerate(q_qc):
    # The IsotropicExchangeCalculator will keep a buffer with the transverse
    # magnetic susceptibility untill we ask for a new q-vector. Thus, we may
    # compute the exchange constants with multiple different site kernels
    # instances, virtually without any computational overhead.
    J_qabp[q] = isoexch_calc(q_c, sitekernels)
    Juc_qabp[q] = isoexch_calc(q_c, ucsitekernels)
# Since we only have a single site, reduce the arrays
J_qr = J_qabp[:, 0, 0, :]
Juc_q = Juc_qabp[:, 0, 0, 0]

# Save the bandpath, spherical radii and computed exchange constants
np.save('Fe_q_qc.npy', q_qc)
np.save('Fe_rc_r.npy', rc_r)
np.save('Fe_J_qr.npy', J_qr)
np.save('Fe_Juc_q.npy', Juc_q)
