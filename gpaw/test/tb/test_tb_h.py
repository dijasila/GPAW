import pytest
from ase import Atoms
from gpaw import GPAW


@pytest.mark.tb
def test_h():
    a = Atoms('H', cell=[9, 9, 9], pbc=1)
    a.calc = GPAW(mode='tb', txt='tb.txt')
    # a.calc = GPAW(mode='lcao', txt='lcao.txt')
    e = a.get_potential_energy()
    print(e)
