from ase.build import molecule
from gpaw import GPAW

def test_gllbghost():
    atoms = molecule('H2')
    atoms.center(vacuum=2)
    calc = GPAW(mode='lcao', basis='dzp', setups={0: 'paw', 1: 'ghost'},
                xc='GLLBSC')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
