from ase import Atoms
from ase.calculators.test import numeric_force
from gpaw import GPAW, PW, Davidson
from gpaw.hybrids import HybridXC


def test_forces():
    a = Atoms('H2',
              positions=[(0, 0, 0), (0, 0, 0.75)],
              pbc=True)
    a.center(vacuum=1.5)
    a.calc = GPAW(
        mode=PW(200, force_complex_dtype=True),
        setups='ae',
        symmetry='off',
        parallel={'kpt': 1, 'band': 1},
        eigensolver=Davidson(1),
        # kpts={'size': (1, 1, 2), 'gamma': True},
        # xc='HSE06',
        xc=HybridXC('EXX'),
        txt='H2.txt')
    a.get_potential_energy()
    f = a.get_forces()
    f0 = numeric_force(a, 0, 2)
    print(f0)
    print(f)
