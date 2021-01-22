import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.neighborlist import primitive_neighbor_list as neighbors
from scipy.optimize import minimize


class Repulsion:
    cutoff: float

    def __call__(self,
                 r: float,
                 nderivs: int = 0) -> float:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.name}{self.params}'

    def plot(self):
        import matplotlib.pyplot as plt
        r = np.linspace(0, 4, 101)
        plt.plot(r, self(r))
        plt.show()


class ZeroRepulsion(Repulsion):
    cutoff = 0.0

    def __call__(self, r, nderivs=0):
        return np.zeros_like(r)


class ExpRepulsion(Repulsion):
    def __init__(self, a, b, c, eps=1e-6):
        self.params = a, b, c
        self.cutoff = -c * np.log(eps)

    def __call__(self, r, nderivs: int = 0) -> float:
        a, b, c = self.params
        if nderivs == 0:
            return (a + b * r) * np.exp(-r / c)
        assert nderivs == 1
        return (b - (a + b * r) / c) * np.exp(-r / c)


class PolynomialRepulsion(Repulsion):
    def __init__(self, a, b, c, eps=1e-6):
        self.params = a, b, c
        self.cutoff = c

    def __call__(self, r, nderivs: int = 0) -> float:
        a, b, c = self.params
        r[r > c] = 0.0
        if nderivs == 0:
            d = r - c
            return (a + b * d) * d**2
        assert nderivs == 1
        return (2 * a + 3 * b * d) * d


def evaluate_pair_potential(repulsions, symbol_a, position_av, cell_cv, pbc_c):
    """Evaluate pair-potential.

    >>> rep = Repulsion(np.e, 0.0, 1.0)
    >>> rep(1.0)
    1.0
    >>> energy, forces = evaluate_pair_potential(
    ...    {('H', 'Li'): rep},
    ...    ['H', 'Li'],
    ...    np.array([[0, 0, 0], [0, 0, 1.0]]),
    ...    np.eye(3),
    ...    np.zeros(3, bool))
    >>> energy
    1.0
    >>> forces
    array([[ 0.,  0., -1.],
          [ 0.,  0.,  1.]])
    """
    cutoffs = {}
    repulsions2 = {}
    symbols = set(symbol_a)
    for s1 in symbols:
        for s2 in symbols:
            rep = repulsions.get((s1, s2))
            if rep is None:
                rep = repulsions.get((s2, s1))
                if rep is None:
                    rep = ZeroRepulsion()
            repulsions2[(s1, s2)] = rep
            cutoffs[(s1, s2)] = rep.cutoff

    ijdD = neighbors('ijdD',
                     pbc_c, cell_cv, position_av, cutoffs,
                     np.array([atomic_numbers[symb] for symb in symbol_a]))
    energy = 0.0
    forces = np.zeros_like(position_av)
    for i, j, d, D in zip(*ijdD):
        rep = repulsions2[(symbol_a[i], symbol_a[j])]
        energy += rep(d)
        force = rep(d, nderivs=1) * D / d
        forces[i] += force
        forces[j] -= force
    return 0.5 * energy, 0.5 * forces


def fit(system: Atoms, atom: Atoms, eps: float = 0.01):
    from gpaw import GPAW, TB
    d0 = system.get_distance(0, 1)
    energies = np.zeros(3)
    distances = np.linspace((1 - eps) * d0, (1 + eps) * d0, 3)
    sign = -1
    for calc in [GPAW(mode=TB({}), txt='tb.txt'),
                 GPAW(mode='lcao', basis='dzp', txt='lcao.txt')]:
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
        rep = ExpRepulsion(*params)
        return sum((rep(d) - e)**2 for d, e in zip(distances, energies))

    res = minimize(f, [1.0, 0.0, 1.0])
    print(res)
    return ExpRepulsion(*res.x)


if __name__ == '__main__':
    if 0:
        rep = fit(Atoms('H2', [(0, 0, 0), (0, 0, 0.78)],
                        cell=[9, 9, 9], pbc=True),
                  Atoms('H', cell=[9, 9, 9], pbc=True),
                  0.1)
    rep = fit(Atoms('Al2', [(0, 0, 0), (0, 0, 2.1)], cell=[9, 9, 9], pbc=True),
              Atoms('Al', cell=[9, 9, 9], pbc=True),
              0.02)
    rep.plot()
