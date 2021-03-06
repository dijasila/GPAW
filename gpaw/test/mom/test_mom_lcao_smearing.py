import pytest
from ase.build import molecule
from gpaw import GPAW, mom
from gpaw.test import equal


@pytest.mark.mom
def test_mom_lcao_smearing():
    atoms = molecule('CO')
    atoms.center(vacuum=3)

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
    # Ground-state calculation
    E_gs = atoms.get_potential_energy()

    f_sn = []
    for spin in range(calc.get_number_of_spins()):
        f_n = calc.get_occupation_numbers(spin=spin)
        f_sn.append(f_n)

    ne0_gs = f_sn[0].sum()
    f_sn[0][3] -= 1.
    f_sn[0][5] += 1.

    # Excited-state MOM calculation with Gaussian
    # smearing of the occupation numbers
    mom.mom_calculation(calc, atoms, f_sn, width=0.1)
    E_es = atoms.get_potential_energy()

    f_n0 = calc.get_occupation_numbers(spin=0)
    ne0_es = f_n0.sum()

    dE = E_es - E_gs
    dne0 = ne0_es - ne0_gs
    print(dE)
    equal(dE, 9.8313927688, 0.01)
    equal(dne0, 0.0, 1e-16)
