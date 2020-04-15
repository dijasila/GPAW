import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.calculators.test import numeric_force
from gpaw import GPAW, PW, Davidson
from gpaw.hybrids import HybridXC
from gpaw.hybrids.energy import non_self_consistent_energy as nsce


def test_forces():
    a = Atoms('H2',
              positions=[(0, 0, 0), (0, 0, 0.75),
                 ],#        (2, 0, 0), (2, 0, 0.75)],
              pbc=True,
              #magmoms=[0, 0, 0, 0.1]
              )
    a.center(vacuum=1.5)
    a.calc = GPAW(
        mode=PW(200, force_complex_dtype=True),
        # setups='ae',
        symmetry='off',
        parallel={'kpt': 1, 'band': 1},
        eigensolver=Davidson(1),
        kpts={'size': (1, 1, 2), 'gamma': True},
        # xc='HSE06',
        xc=HybridXC('EXX'),
        #convergence={'forces': 1e-3},
        txt='H2.txt')
    # from jj import plot as P
    D = np.linspace(0.7, 0.99, 15)
    #D = [0.7, 0.8, 0.9, 1.0]
    F = []
    FF = []
    E = []
    import jj
    for d in D:
        a.set_distance(0, 1, d)
        #e = a.get_potential_energy()
        #e2 = nsce(a.calc, 'EXX')
        # print(e2)
        f = a.get_forces()[0, 2]
        #print(f)
        # f00 = numeric_force(a, 0, 2)
        f0 = numeric_force(a, 0, 2, 0.01)
        print(f0, f0 - f)
        return
        jj.plot(f0,f)
        #F.append(f[0, 2] - f0)
        # P(d, (e, e2.sum(), f[0, 2], 0))
        E.append(e)
        # F.append(f[0, 2])
    print(F)
    print(FF)
    #print(np.linalg.lstsq(FF, F))
    return
    for i in range(1, 14):
        print(F[i], F[i] + (E[i - 1] - E[i + 1]) / (D[2] - D[0]))
    f = a.get_forces()
    f0 = -2.5373125013143927
    print(f0)
    print(f)
    