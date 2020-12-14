import numpy as np
from ase import Atoms
from gpaw import GPAW, TB
from gpaw.tb.repulsion import Repulsion


def test_h2():
    D = np.linspace(0.5, 1.0, 20)
    D = np.linspace(0.74 * 0.90, 0.74 * 1.1, 30)
    rep = Repulsion(23.86275938, -14.81309506,   0.86202836)
    E = []
    for d in D:
        a = Atoms('H2', positions=[[0, 0, 0], [0, 0, d]], cell=[9, 9, 9], pbc=1)
        #a.center(vacuum=5)
        #a.calc = GPAW(mode=TB(ZeroRepulsion()), txt=f'{d}.txt')
        a.calc = GPAW(mode='tb', txt=f'{d}.txt')
        #a.calc = GPAW(mode='lcao', basis='dzp', txt=f'{d}.txt')
        if 0:
            a.calc = GPAW(mode='lcao')
        e = a.get_potential_energy()# + rep(d)
        print(d, e, rep(d))
        E.append(e)
        return
    import matplotlib.pyplot as plt
    plt.plot(D, E)
    plt.show()
    