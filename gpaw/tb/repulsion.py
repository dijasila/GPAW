import numpy as np
from ase import Atoms
from scipy.optimize import minimize

from gpaw import GPAW


class Repulsion:
    def __init__(self, params):
        self.params = params

    def __call__(self, r):
        a, b, c = self.params
        return (a + r * b) * np.exp(-r / c)

    def __repr__(self):
        return f'Repulsion({self.params})'

    def plot(self):
        import matplotlib.pyplot as plt
        r = np.linspace(0, 4, 101)
        plt.plot(r, self(r))
        plt.show()


def fit(system: Atoms, atom: Atoms, eps: float = 0.01):
    d0 = system.get_distance(0, 1)
    energies = np.zeros(3)
    distances = np.linspace((1 - eps) * d0, (1 + eps) * d0, 3)
    sign = -1
    for calc in [GPAW(mode='tb', txt='tb.txt'),
                 GPAW(mode='lcao', txt='lcao.txt')]:
        atom.calc = calc
        e0 = atom.get_potential_energy()
        atoms = system.copy()
        atoms.calc = calc
        for i, d in enumerate(distances):
            atoms.set_distance(0, 1, d)
            e = atoms.get_potential_energy() - 2 * e0
            print(d, e)
            energies[i] += sign * e
        sign = 1

    print(distances, energies)

    def f(params):
        rep = Repulsion(params)
        return sum((rep(d) - e)**2 for d, e in zip(distances, energies))

    res = minimize(f, [1.0, 0.0, 1.0])
    print(res)
    return Repulsion(res.x)


if __name__ == '__main__':
    rep = fit(Atoms('H2', [(0, 0, 0), (0, 0, 0.74)], cell=[9, 9, 9], pbc=True),
              Atoms('H', cell=[9, 9, 9], pbc=True))
    rep.plot()
