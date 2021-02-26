import pytest
import numpy as np
from ase import Atoms
from gpaw import GPAW
import gpaw.mom as mom
from gpaw.test import equal


@pytest.mark.mom
def test_mom_lcao_forces():
    f_n = [[1., 1., 1., 1., 0., 1., 0., 0.],
           [1., 1., 1., 1., 1., 0., 0., 0.]]
    L = 10.0
    d = 1.13
    delta = 0.01

    atoms = Atoms('CO',
                  [[0.5 * L, 0.5 * L, 0.5 * L - 0.5 * d],
                   [0.5 * L, 0.5 * L, 0.5 * L + 0.5 * d]])
    atoms.set_cell([L, L, L])

    calc = GPAW(mode='lcao',
                basis='dzp',
                nbands=8,
                h=0.2,
                xc='PBE',
                spinpol=True,
                convergence={'energy': 100,
                             'density': 1e-3,
                             'bands': -1})

    atoms.calc = calc
    mom.mom_calculation(calc, atoms, f_n)
    F = atoms.get_forces()

    E = []
    p = atoms.positions.copy()
    for i in [-1, 1]:
        pnew = p.copy()
        pnew[0, 2] -= delta / 2. * i
        pnew[1, 2] += delta / 2. * i
        atoms.set_positions(pnew)

        E.append(atoms.get_potential_energy())

    f = np.sqrt(((F[1, :] - F[0, :])**2).sum()) * 0.5
    fnum = (E[0] - E[1]) / (2. * delta)     # central difference

    print(fnum)
    equal(fnum, 12.1329758577, 0.01)
    equal(f, fnum, 0.05)
