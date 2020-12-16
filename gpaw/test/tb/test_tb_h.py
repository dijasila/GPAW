import pytest
from ase import Atoms
from gpaw import GPAW


@pytest.mark.tb
def test_h():
    # a = Atoms('Al', cell=[9, 9, 9], pbc=1)
    a = Atoms('H', cell=[9, 9, 9], pbc=1)
    a.center()
    a.calc = GPAW(mode='tb', h=0.15, txt='tb.txt', setups='sz')
    # a.calc = GPAW(mode='lcao', txt='lcao.txt')
    e = a.get_potential_energy()
    print(e)
    print(a.calc.wfs.kpt_u[0].f_n)
    print(a.calc.wfs.kpt_u[0].P_ani[0])
