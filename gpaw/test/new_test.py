from gpaw.calculator import GPAW as OldGPAW
from gpaw.new.ase_interface import GPAW as NewGPAW
from ase import Atoms


def new():
    atoms = Atoms('H2')
    atoms.positions[1, 2] = 0.8
    atoms.center(vacuum=1)
    params = {}  # 'txt': '-'}
    atoms.calc = NewGPAW(**params,
                         txt='new.txt')
    atoms.get_forces()
    if 1:
        atoms.positions[1, 2] = 1.75
        atoms.calc = NewGPAW(**params,
                             txt='new2.txt')
        atoms.get_forces()
    atoms.calc = OldGPAW(**params,
                         txt='old.txt')
    atoms.get_forces()


if __name__ == '__main__':
    new()
