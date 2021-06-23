from ase.phonons import Phonons
from gpaw import GPAW
from gpaw.elph.electronphonon import ElectronPhononCoupling
from gpaw.raman.elph import get_elph_matrix

# Load pre-computed ground state calculation (primitive cell)
calc = GPAW('gs.gpw')
atoms = calc.atoms

# Load results from phonon and electron-phonon coupling calculations
phonon = Phonons(atoms, supercell=(1, 1, 1))
elph = ElectronPhononCoupling(atoms, supercell=(2, 2, 2))

# Construct electron-phonon matrix of Bloch functions
get_elph_matrix(atoms, calc, elph, phonon, dump=2, load_gx_as_needed=False)
