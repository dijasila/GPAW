from ase import Atoms
from ase.visualize import view
from gpaw import GPAW
from gpaw.wannier import calculate_overlaps

calc = GPAW('si.gpw')
atoms = calc.get_atoms()

overlaps = calculate_overlaps(calc)
wan = overlaps.localize()

watoms = atoms + Atoms(symbols='X16', positions=wan.centers)
view(watoms * (2, 2, 2))
