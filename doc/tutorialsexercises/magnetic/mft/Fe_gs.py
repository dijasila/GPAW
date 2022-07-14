"""DFT ground state calculation of Fe(bcc), storing occupied and unoccupied
Kohn-Sham orbitals in a file."""

# Script modules
from gpaw import GPAW, PW, FermiDirac
from ase.build import bulk

# ---------- Inputs ---------- #

# Crystal structure
a = 2.867  # Lattice constant
mm = 2.21  # Initial magnetic moment

# The derivation of the Heisenberg exchange constants based on the magnetic
# force theorem takes an outset in the L(S)DA exchange-correlation functional.
# Thus, to be formally consistent, we need to use the LDA functional for the
# ground state.
xc = 'LDA'
# The plane wave energy cutoff of the ground state calculation is chosen:
# 1) high enough to provide well converged Kohn-Sham orbitals,
# 2) larger or equal to the energy cutoff of the mft response calculation.
# In a benchmark study, the latter will typically be the stricter requirement.
pw = 800  # eV
# When specifying the k-point grid, one should keep in mind that the q-points
# of the mft response calculation are required to be commensurate with the
# grid. For a bcc crystal, a k-point grid that samples a multiple of 4 k-points
# along all axes will support calculations at the high-symmetry points N, H and
# P. We choose N_k^(1/3)=32, which should yield converged magnon energies
# according to the convergence study of [arXiv:2204.04169].
kpts = 32  # final grid is (kpts, kpts, kpts)
# In the original paper [arXiv:2204.04169], it was shown that the mft response
# calculations are well converged when including only bands corresponding to
# shells with partial or full occupation.
nbands_mft = 6  # 4s + 3d
# To stabilize the ground state calculation, we add some extra empty bands
nbands_gs = nbands_mft + 4  # 4p + 5s
# For response calculations, we typically need a more strict convergence of the
# ground state density, magnetization and Kohn-Sham orbitals than usual.
conv = {'density': 1.e-8,
        'forces': 1.e-8,
        'bands': nbands_mft}  # Converge only the bands needed subsequently
occw = 0.001  # Fermi temperature in eV

# ---------- Script ---------- #

# Set up crystal structure
atoms = bulk('Fe', 'bcc', a=a)
atoms.set_initial_magnetic_moments([mm])
atoms.center()

# Construct the ground state calculator
kpt_grid = {'size': (kpts, kpts, kpts),
            'gamma': True}  # When converged, the grid offset shouldn't matter
calc = GPAW(xc='LDA',
            mode=PW(pw),
            kpts=kpt_grid,
            nbands=nbands_gs,
            convergence=conv,
            occupations=FermiDirac(occw),
            txt='Fe_gs.txt')

# DFT ground state calculation
atoms.calc = calc
atoms.get_potential_energy()

# Save the ground state and Kohn-Sham orbitals
# WARNING: This will generally leave large files on your computer. It is wise
# to remove them, once all subsequent calculations have finished.
calc.write('Fe_all.gpw', mode='all')
