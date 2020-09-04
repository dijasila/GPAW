from ase import Atoms
from ase.visualize import view
from gpaw import GPAW
from gpaw.wannier.overlaps import calculate_overlaps
from gpaw.wannier.edmiston_ruedenberg import localize

calc = GPAW('si.gpw')
atoms = calc.get_atoms()

overlaps = calculate_overlaps(calc)
wan = localize(overlaps)

watoms = atoms + Atoms(symbols='X16', positions=wan.centers)
view(watoms * (2, 2, 2))
