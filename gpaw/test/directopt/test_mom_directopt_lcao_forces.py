import numpy as np
import pytest
from ase import Atoms
from gpaw import GPAW, restart
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.lcao.directmin_lcao import DirectMinLCAO
from gpaw.test import equal


@pytest.mark.mom
def test_mom_directopt_lcao_forces(in_tmp_dir):
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
                occupations={'name': 'fixed-uniform'},
                eigensolver='direct-min-lcao',
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                convergence={'energy': 1e-3,
                             'density': 1e-3,
                             'eigenstates': 1e-3})
    atoms.calc = calc
    atoms.get_potential_energy()

    calc.set(eigensolver=DirectMinLCAO(searchdir_algo={'name': 'LSR1P',
                                                       'method': 'LSR1'},
                                       linesearch_algo={'name': 'UnitStep'},
                                       need_init_orbs=False))
    prepare_mom_calculation(calc, atoms, f_sn)
    F = atoms.get_forces()

    # Test overlaps
    calc.wfs.occupations.initialize_reference_orbitals()
    for kpt in calc.wfs.kpt_u:
        f_n = calc.get_occupation_numbers(spin=kpt.s)
        unoccupied = [True for _ in range(len(f_n))]
        P = calc.wfs.occupations.calculate_weights(kpt, 1.0, unoccupied)
        assert (np.allclose(P, f_n))

    calc.write('co.gpw', mode='all')

    # Exercise fixed occupations and no update of numbers in OccupationsMOM
    atoms, calc = restart('co.gpw', txt='-')
    e0 = atoms.get_potential_energy()
    for i in [True, False]:
        prepare_mom_calculation(calc, atoms, f_sn,
                                use_fixed_occupations=i,
                                update_numbers=i)
        e1 = atoms.get_potential_energy()
        for spin in range(calc.get_number_of_spins()):
            if i:
                f_n = calc.get_occupation_numbers(spin=spin)
                assert (np.allclose(f_sn[spin], f_n))
            assert (np.allclose(f_sn[spin],
                                calc.wfs.occupations.numbers[spin]))
        equal(e0, e1, 1e-3)

    E = []
    for i in [-1, 1]:
        atoms, calc = restart('co.gpw', txt='-')
        p = atoms.positions.copy()
        p[0, 2] -= delta / 2. * i
        p[1, 2] += delta / 2. * i
        atoms.set_positions(p)
        E.append(atoms.get_potential_energy())

    f = np.sqrt(((F[1, :] - F[0, :])**2).sum()) * 0.5
    fnum = (E[0] - E[1]) / (2. * delta)     # central difference

    print(fnum)
    assert fnum == pytest.approx(11.9364629545, 0.01)
    assert f == pytest.approx(fnum, 0.1)
