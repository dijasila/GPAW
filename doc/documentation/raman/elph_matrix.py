from gpaw import GPAW
from gpaw.raman.elph import get_elph_matrix

# Load pre-computed calculation
calc = GPAW("gs.gpw")
atoms = calc.atoms

# Construct electron-phonon matrix of Bloch functions
get_elph_matrix(atoms, calc, dump=2, load_gx_as_needed=True)
