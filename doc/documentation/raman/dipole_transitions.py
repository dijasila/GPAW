from gpaw import GPAW
from gpaw.raman.dipoletransition import get_dipole_transitions

# Load pre-computed calculation
calc = GPAW("gs.gpw")
atoms = calc.atoms

# Get transition dipole moments
get_dipole_transitions(atoms, calc)
