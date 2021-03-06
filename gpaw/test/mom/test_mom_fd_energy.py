import pytest
import numpy as np
from ase.build import molecule
from gpaw import GPAW, restart, mom
from gpaw.test import equal


@pytest.mark.mom
def test_mom_fd_energy():
    dE_ref = [7.6319602946, 7.4176240132]

    atoms = molecule('H2O')
    atoms.center(vacuum=3)

    calc = GPAW(mode='fd',
                basis='dzp',
                nbands=6,
                h=0.2,
                xc='PBE',
                spinpol=True,
                symmetry='off',
                convergence={'energy': 100,
                             'density': 1e-3,
                             'eigenstates': 100,
                             'bands': 'all'})

    atoms.calc = calc
    # Ground-state calculation
    E_gs = atoms.get_potential_energy()

    calc.write('h2o_fd_gs.gpw', 'all')

    # Test spin polarized excited-state calculations
    for s in [0, 1]:
        atoms, calc = restart('h2o_fd_gs.gpw')

        f_sn = []
        for spin in range(calc.get_number_of_spins()):
            f_n = calc.get_occupation_numbers(spin=spin)
            f_sn.append(f_n)
        f_sn[0][3] -= 1.
        f_sn[s][4] += 1.

        mom.mom_calculation(calc, atoms, f_sn)

        E_es = atoms.get_potential_energy()

        # Test overlaps
        calc.wfs.occupations.initialize_reference_orbitals()
        for kpt in calc.wfs.kpt_u:
            f_sn = calc.get_occupation_numbers(spin=kpt.s)
            unoccupied = [True for i in range(len(f_sn))]
            P = calc.wfs.occupations.calculate_weights(kpt, 1.0, unoccupied)
            assert (np.allclose(P, f_sn))

        dE = E_es - E_gs
        print(dE)
        equal(dE, dE_ref[s], 0.01)

    calc = GPAW(mode='fd',
                basis='dzp',
                nbands=6,
                h=0.2,
                xc='PBE',
                symmetry='off',
                convergence={'energy': 100,
                             'density': 1e-3,
                             'eigenstates': 100,
                             'bands': 'all'})

    atoms.calc = calc
    # Ground-state calculation
    E_gs = atoms.get_potential_energy()

    # Test spin paired excited-state calculation
    f_n = [calc.get_occupation_numbers(spin=0) / 2.]
    f_n[0][3] -= 0.5
    f_n[0][4] += 0.5

    mom.mom_calculation(calc, atoms, f_n)
    E_es = atoms.get_potential_energy()

    dE = E_es - E_gs
    print(dE)
    equal(dE, 8.7357394806, 0.01)
