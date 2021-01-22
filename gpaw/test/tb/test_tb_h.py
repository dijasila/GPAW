import pytest
from ase import Atoms
from gpaw import GPAW


@pytest.mark.tb
def test_h():
    # a = Atoms('Al', cell=[9, 9, 9], pbc=1)
    a = Atoms('H', cell=[11, 11, 11], pbc=1)
    a.center()
    # a.calc = GPAW(mode='lcao', h=0.15, txt='tb.txt', setups='sz')
    # a.calc = GPAW(mode='pw', txt='pw1.txt', setups='sz')
    a.calc = GPAW(mode='lcao', txt='lcao.txt', setups='1s')
    # a.calc = GPAW(mode='pw', txt='pw0.txt', setups='s0')
    e = a.get_potential_energy()
    if 0:
        import matplotlib.pyplot as plt
        plt.plot(a.calc.hamiltonian.vt_sG[0, 23, 24])
        a.calc = GPAW(mode='pw', txt='pw.txt', setups='ssp')
        # a.calc = GPAW(mode='lcao', txt='lcao.txt')
        e = a.get_potential_energy()
        plt.plot(a.calc.hamiltonian.vt_sG[0, 23, 24])
        plt.show()
    print(e)
    print(a.calc.wfs.kpt_u[0].f_n)
    print(a.calc.wfs.kpt_u[0].P_ani[0])
    #print(a.calc.wfs.P_aqMi[0])
    print(a.calc.get_eigenvalues())
