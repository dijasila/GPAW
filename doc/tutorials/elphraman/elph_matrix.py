from ase.phonons import Phonons
from gpaw import GPAW
from gpaw.raman.elph import EPC

# Load pre-computed ground state calculation (primitive cell)
calc = GPAW('gs.gpw', parallel={'band': 1})
atoms = calc.atoms

# Load results from phonon and electron-phonon coupling calculations
phonon = Phonons(atoms, supercell=(1, 1, 1))
elph = EPC(atoms, supercell=(2, 2, 2))

# Construct electron-phonon matrix of Bloch functions
elph.get_elph_matrix(calc, phonon)
