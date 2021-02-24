import pytest
from ase.build import molecule
from gpaw import GPAW, restart
import gpaw.mom as mom
from gpaw.test import equal


@pytest.mark.mom
def test_mom_lcao(gpw_files):
    dE_ref = [8.1458184518, 7.8544875500]

    H2O = molecule('H2O')
    H2O.center(vacuum=3)

    # Ground-state calculation spin polarized
    calc = GPAW(mode='lcao',
                basis='dzp',
                nbands=6,
                h=0.2,
                xc='PBE',
                spinpol=True,
                symmetry='off',
                convergence={'energy': 100,
                             'density': 1e-3,
                             'bands': -1})

    H2O.calc = calc
    E_gs = H2O.get_potential_energy()
    calc.write('h2o_lcao_gs.gpw', 'all')

    # Test spin-mixed and triplet calculations
    for s in [0, 1]:
        H2O, calc = restart('h2o_lcao_gs.gpw', txt=None)

        f_n = []
        for spin in range(calc.get_number_of_spins()):
            f_ns = calc.get_occupation_numbers(spin=spin)
            f_n.append(f_ns)

        # Excited-state MOM calculation
        f_n[0][3] -= 1.
        f_n[s][4] += 1.

        mom.mom_calculation(calc, H2O, f_n)

        E_es = H2O.get_potential_energy()

        dE = E_es - E_gs
        print(dE)
        equal(dE, dE_ref[s], 0.01)

    # Ground-state calculation spin paired
    calc = GPAW(mode='lcao',
                basis='dzp',
                nbands=6,
                h=0.2,
                xc='PBE',
                symmetry='off',
                convergence={'energy': 100,
                             'density': 1e-3,
                             'bands': -1})

    H2O.calc = calc
    E_gs = H2O.get_potential_energy()
    calc.write('h2o_lcao_gs.gpw', 'all')

    # Test singlet spin paired
    f_n = [calc.get_occupation_numbers(spin=0)/2.]

    # Excited-state MOM calculation
    f_n[0][3] -= 0.5
    f_n[0][4] += 0.5

    mom.mom_calculation(calc, H2O, f_n)

    E_es = H2O.get_potential_energy()

    dE = E_es - E_gs
    print(dE)
    equal(dE, 9.2318246288, 0.01)