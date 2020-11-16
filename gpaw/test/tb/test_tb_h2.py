import numpy as np
from ase import Atoms
from gpaw import GPAW


def test_h2():
    D = np.linspace(0.5, 3.5, 20)
    E = []
    for d in D:
        a = Atoms('H2', positions=[[0, 0, 0], [0, 0, d]])
        a.center(vacuum=2)
        a.calc = GPAW(mode='tb', txt=f'{d}.txt')
        if 0:
            a.calc = GPAW(mode='lcao')
        e = a.get_potential_energy()
        E.append(e)
    import matplotlib.pyplot as plt
    plt.plot(D, E)
    plt.show()
    