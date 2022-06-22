"""Compute magnon dispersion on band path between all the high-symmetry
points of the hcp lattice. The resulting magnon energies are saved."""

# General modules
import numpy as np

# CAMD modules
from gpaw import GPAW
from gpaw.response.mft import IsotropicExchangeCalculator, \
    compute_magnon_energy_FM

# ----- Setup exchange calculator ----- #

# Settings specific to exchange calculator
ecut = 400  # Energy cutoff in eV for response calc (number of G-vectors)
# How to map onto discrete lattice of magnetic moments
sitePos_mv = 'atoms'  # Magnetic sites at all atoms
shapes_m = ['sphere', 'sphere']  # Use spheres to define magnetic sites
rcBest = 1.3  # Pick optimal rc value

# Load converged ground state
calc = GPAW(f'converged_gs_calc.gpw', parallel={'domain': 1})
Co = calc.get_atoms()
mm = calc.get_magnetic_moments()

# Get number of converged bands
nbands_response = calc.parameters['convergence']['bands']

# Get size of k-space simulation grid
k = calc.parameters['kpts']['size'][0]

# Setup exchange calculator
exchCalc = IsotropicExchangeCalculator(calc,
                                       sitePos_mv,
                                       shapes_m=shapes_m,
                                       ecut=ecut,
                                       nbands=nbands_response)

# ----- Magnon dispersion between high symmetry points ----- #

# Construct bandpath between all the high-symmetry points
# G -> K -> H -> A -> L -> M -> G
# where G is the Gamma point.
# Note that the response calculation only works for points of the form
# q = (n_x/k, n_y/k, n_z/k) where n_i is an integer and (k, k, k) is the
# simulation grid
GK = np.array([[x, x, 0] for x in np.arange(0, 1/3+0.01, 1/k)])
KH = np.array([[1/3, 1/3, x] for x in np.arange(1/k, 1/2+0.01, 1/k)])
HA = np.array([[1/3-x, 1/3-x, 1/2] for x in np.arange(1/k, 1/3+0.01, 1/k)])
AL = np.array([[x, 0, 1/2] for x in np.arange(1/k, 1/2+0.01, 1/k)])
LM = np.array([[1/2, 0, 1/2-x] for x in np.arange(1/k, 1/2+0.01, 1/k)])
MG = np.array([[1/2-x, 0, 0] for x in np.arange(1/k, 1/2+0.01, 1/k)])
q_qc = np.vstack([GK, KH, HA, AL, LM, MG])

# Compute exchange coupling along bandpath
Nq = len(q_qc)
N_sites = len(exchCalc.sitePos_mv)
J_rmnq = np.empty([1, N_sites, N_sites, Nq], dtype=complex)
for q, q_c in enumerate(q_qc):
    J_rmn = exchCalc(q_c, rc_rm=rcBest)
    J_rmnq[:, :, :, q] = J_rmn
J_mnq = J_rmnq[0, :, :, :]

# Compute magnon energies along bandpath
E_mq = compute_magnon_energy_FM(J_mnq, q_qc, mm)

# Save energies and q-points
np.save('q_qc.npy', q_qc)
np.save('high_sym_path_E_mq.npy', E_mq)