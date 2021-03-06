import pytest
import numpy as np
from ase import Atoms
from gpaw import GPAW, mom
from gpaw.test import equal


@pytest.mark.mom
def test_mom_lcao_forces():
    f_sn = [[1., 1., 1., 1., 0., 1., 0., 0.],
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
                symmetry='off',
                convergence={'energy': 100,
                             'density': 1e-3})

    atoms.calc = calc
    mom.mom_calculation(calc, atoms, f_sn)
    F = atoms.get_forces()

    # Test overlaps
    calc.wfs.occupations.initialize_reference_orbitals()
    for kpt in calc.wfs.kpt_u:
        f_n = calc.get_occupation_numbers(spin=kpt.s)
        unoccupied = [True for i in range(len(f_n))]
        P = calc.wfs.occupations.calculate_weights(kpt, 1.0, unoccupied)
        assert (np.allclose(P, f_n))

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
    equal(fnum, 12.0325744975, 0.01)
    equal(f, fnum, 0.1)
