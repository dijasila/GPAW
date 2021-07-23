from gpaw import GPAW, PW
from ase import Atoms
import pytest


def test_q_spiral():
    L = 2.0
    atom = Atoms('H', magmoms=[1], cell=[L, L, L], pbc=True)
    atom.calc = GPAW(mode='pw', txt=None)
    E1 = atom.get_potential_energy()
    a1, b1 = (atom.calc.get_eigenvalues(spin=s)[0] for s in [0, 1])
    
    magmoms = [[0, 0, 1]]
    atom.calc = GPAW(mode='pw',
                     symmetry='off',
                     experimental={'magmoms': magmoms, 'soc': False},
                     txt=None)
    E2 = atom.get_potential_energy()
    a2, b2 = atom.calc.get_eigenvalues()

    assert E2 - E1 == pytest.approx(0.0, abs=1e-8), E2 - E1
    assert a2 - a1 == pytest.approx(0.0, abs=1e-8), a2 - a1
    assert b2 - b1 == pytest.approx(0.0, abs=1e-6), b2 - b1

    q = [0, 0, 0]
    atom.calc = GPAW(mode=PW(qspiral=q),
                     symmetry='off',
                     experimental={'magmoms': magmoms, 'soc': False},
                     txt=None)
    E3 = atom.get_potential_energy()
    a3, b3 = atom.calc.get_eigenvalues()
    
    assert E3 - E1 == pytest.approx(0.0, abs=1e-8)
    assert a3 - a1 == pytest.approx(0.0, abs=1e-8), a3 - a1
    assert b3 - b1 == pytest.approx(0.0, abs=1e-6), b3 - b1


if __name__ == '__main__':
    test_q_spiral()
    print('Test completed.')
