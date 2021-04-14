import numpy as np
import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.mom import prepare_mom_calculation
from gpaw.test import equal
from gpaw.directmin.lcao.directmin_lcao import DirectMinLCAO


@pytest.mark.mom
def test_mom_lcao_forces_directopt(in_tmp_dir):
    f_sn = []
    for s in range(2):
        f_sn.append([1. if i < 5 else 0. for i in range(26)])
    f_sn[0][4] = 0.
    f_sn[0][5] = 1.

    L = 6.0
    d = 1.13
    delta = 0.01

    atoms = Atoms('CO',
                  [[0.5 * L, 0.5 * L, 0.5 * L - 0.5 * d],
                   [0.5 * L, 0.5 * L, 0.5 * L + 0.5 * d]])
    atoms.set_cell([L, L, L])

    calc = GPAW(mode='lcao',
                basis='dzp',
                h=0.20,
                xc='PBE',
                spinpol=True,
                symmetry='off',
                mixer={'name': 'dummy'},
                nbands='nao',
                eigensolver=DirectMinLCAO(searchdir_algo={'name': 'LSR1P',
                                                          'method': 'LSR1'},
                                          linesearch_algo={'name': 'UnitStep'}),
                convergence={'energy': 100,
                             'density': 1e-3,
                             'eigenstates': 1e-3})

    atoms.calc = calc
    prepare_mom_calculation(calc, atoms, f_sn)
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
    equal(fnum, 11.9364629545, 0.01)
    equal(f, fnum, 0.1)
