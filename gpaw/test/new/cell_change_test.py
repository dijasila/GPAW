from ase import Atoms
from gpaw.new.ase_interface import GPAW


def test_new_cell():
    gpu = 0
    atoms = Atoms('H', pbc=True, cell=[1, 1, 1])
    atoms.calc = GPAW(
        mode={'name': 'pw'},
        kpts=(4, 1, 1),
        parallel={'gpu': gpu})
    atoms.get_potential_energy()
    atoms.cell[2, 2] = 0.9
    atoms.get_potential_energy()
6