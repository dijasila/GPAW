from ase import Atoms
from ase.build import molecule
from ase.visualize import view
from gpaw import GPAW
from gpaw.wannier import calculate_overlaps

calc = GPAW(nbands=5)
atoms = molecule('CO')
atoms.center(vacuum=3.)
atoms.calc = calc
atoms.get_potential_energy()

# Initialize the Wannier class
w = calculate_overlaps(calc).localize()
centers = w.get_centers()
view(atoms + Atoms(symbols='X5', positions=centers))
