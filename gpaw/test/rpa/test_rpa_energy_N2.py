import pytest
from gpaw.mpi import world
from gpaw.utilities import compiled_with_sl
from ase.build import molecule
from gpaw import GPAW, PW
from gpaw.test import equal
from gpaw.xc.rpa import RPACorrelation
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy

pytestmark = pytest.mark.skipif(
    world.size != 1 and not compiled_with_sl(),
    reason='world.size != 1 and not compiled_with_sl()')


def test_rpa_rpa_energy_N2(in_tmp_dir):
    ecut = 25

    N2 = molecule('N2')
    N2.center(vacuum=2.0)

    calc = GPAW(mode=PW(force_complex_dtype=True),
                xc='PBE',
                parallel={'domain': 1},
                eigensolver='rmm-diis')
    N2.calc = calc
    E_n2_pbe = N2.get_potential_energy()

    calc.diagonalize_full_hamiltonian(nbands=104, scalapack=True)
    calc.write('N2.gpw', mode='all')

    E_n2_hf = nsc_energy('N2.gpw', 'EXX')

    rpa = RPACorrelation('N2.gpw', nfrequencies=8)
    E_n2_rpa = rpa.calculate(ecut=[ecut])

    N = molecule('N')
    N.set_cell(N2.cell)

    calc = GPAW(mode=PW(force_complex_dtype=True),
                xc='PBE',
                parallel={'domain': 1},
                eigensolver='rmm-diis')
    N.calc = calc
    E_n_pbe = N.get_potential_energy()

    calc.diagonalize_full_hamiltonian(nbands=104, scalapack=True)
    calc.write('N.gpw', mode='all')

    E_n_hf = nsc_energy('N.gpw', 'EXX')

    rpa = RPACorrelation('N.gpw', nfrequencies=8)
    E_n_rpa = rpa.calculate(ecut=[ecut])

    print('Atomization energies:')
    print('PBE: ', E_n2_pbe - 2 * E_n_pbe)
    print('HF: ', E_n2_hf - 2 * E_n_hf)
    print('HF+RPA: ', E_n2_hf - 2 * E_n_hf + E_n2_rpa[0] - 2 * E_n_rpa[0])

    equal(E_n2_rpa - 2 * E_n_rpa, -1.68, 0.02)
