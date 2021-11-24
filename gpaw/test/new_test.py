from gpaw.calculator import GPAW as OldGPAW
from gpaw.new.ase_interface import GPAW as NewGPAW
from ase import Atoms


def new():
    atoms = Atoms('H2')
    atoms.positions[1, 2] = 0.8
    atoms.center(vacuum=1)
    params = {'txt': '-'}
    atoms.calc = NewGPAW(**params)
    atoms.get_forces()
    atoms.calc = OldGPAW(**params)
    atoms.get_forces()


if __name__ == '__main__':
    new()
