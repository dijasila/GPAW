from pathlib import Path
from ase.io import read
from gpaw import GPAW
from gpaw.point_groups import SymmetryChecker

charges = {'Ico': -2,
           'I': -2,
           'Th': 2}

for path in Path().glob('*-*.xyz'):
    print(path)
    pg = path.name.split('-')[0]
    atoms = read(path)
    atoms.center(vacuum=3.5)
    atoms.calc = GPAW(mode='lcao',
                      charge=charges.get(pg, 0.0),
                      txt=path.with_suffix('.txt'))
    atoms.get_potential_energy()
    center = atoms.get_center_of_mass()
    checker = SymmetryChecker(pg, center, 2.5)
    result = checker.check_band(atoms.calc, 0)
    print(result)
