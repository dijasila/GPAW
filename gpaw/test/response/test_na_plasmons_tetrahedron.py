import pytest
from ase import Atoms
from gpaw.mpi import world
from gpaw.utilities import compiled_with_sl
from gpaw import GPAW, PW
from gpaw.response.df import DielectricFunction
from gpaw.test import equal, findpeak

# Comparing the EELS spectrum of sodium for different block
# parallelizations. Intended to be run with 8 cores.

pytestmark = pytest.mark.skipif(
    world.size < 4 or not compiled_with_sl(),
    reason='world.size < 4 or not compiled_with_sl()')


@pytest.mark.response
@pytest.mark.slow
def test_response_na_plasmons_tetrahedron(in_tmp_dir):
    a = 4.23 / 2.0
    a1 = Atoms('Na',
               scaled_positions=[[0, 0, 0]],
               cell=(a, a, a),
               pbc=True)

    a1.calc = GPAW(mode=PW(250),
                   kpts={'size': (10, 10, 10), 'gamma': True},
                   parallel={'band': 1},
                   txt='small.txt')

    a1.get_potential_energy()
    a1.calc.diagonalize_full_hamiltonian(nbands=20)
    a1.calc.write('gs_Na.gpw', 'all')

    kwargs = {'integrationmode': 'tetrahedron integration',
              'ecut': 40}

    # Calculate the dielectric functions
    df1 = DielectricFunction('gs_Na.gpw',
                             nblocks=1,
                             txt='1block.txt',
                             **kwargs)

    df1NLFCx, df1LFCx = df1.get_dielectric_function(direction='x')
    # df1NLFCy, df1LFCy = df1.get_dielectric_function(direction='y')
    # df1NLFCz, df1LFCz = df1.get_dielectric_function(direction='z')

    df2 = DielectricFunction('gs_Na.gpw',
                             nblocks=4,
                             txt='4block.txt',
                             **kwargs)

    df2NLFCx, df2LFCx = df2.get_dielectric_function(direction='x')
    # df2NLFCy, df2LFCy = df2.get_dielectric_function(direction='y')
    # df2NLFCz, df2LFCz = df2.get_dielectric_function(direction='z')

    # Compare plasmon frequencies and intensities
    w_w = df1.chi0.omega_w

    w1, I1 = findpeak(w_w, -(1. / df1LFCx).imag)
    w2, I2 = findpeak(w_w, -(1. / df2LFCx).imag)
    equal(w1, w2, 1e-2)
    equal(I1, I2, 1e-3)
