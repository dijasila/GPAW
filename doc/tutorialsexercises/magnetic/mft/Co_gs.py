"""DFT ground state calculation of Co(hcp)."""

# Script modules
from gpaw import GPAW, PW, FermiDirac
from ase.build import bulk

# ---------- Inputs ---------- #

# Crystal structure
a = 2.5071
c = 4.0695
mm = 1.67

# Calculator parameters
xc = 'LDA'
pw = 800  # eV
# For a hcp crystal, a k-point grid that samples a multiple of 6 k-points
# along all axes will support calculations at the high-symmetry points M, K and
# A. For this tutorial, we choose N_k^(1/3)=24.
kpts = 24  # final grid is (kpts, kpts, kpts)
nbands_mft = 2 * 6  # natoms * (4s + 3d)
nbands_gs = nbands_mft + 2 * 4  # + natoms * (4p + 5s)
conv = {'density': 1.e-8,
        'forces': 1.e-8,
        'bands': nbands_mft}  # Converge only the bands needed subsequently
occw = 0.001  # Fermi temperature in eV

# ---------- Script ---------- #

# Set up crystal structure
atoms = bulk('Co', 'hcp', a=a, c=c)
atoms.set_initial_magnetic_moments([mm, mm])
atoms.center()

calc = GPAW(xc='LDA',
            mode=PW(pw),
            kpts={'size': (kpts, kpts, kpts), 'gamma': True},
            nbands=nbands_gs,
            convergence=conv,
            occupations=FermiDirac(occw),
            txt='Co_gs.txt')

atoms.calc = calc
atoms.get_potential_energy()

# Save the ground state w.o. the Kohn-Sham orbitals
calc.write('Co.gpw')
