import pytest
from gpaw.mpi import world
from gpaw.utilities import compiled_with_sl
import numpy as np
from ase.build import bulk
from gpaw import GPAW, FermiDirac, PW
from gpaw.response.g0w0 import G0W0
import pickle

pytestmark = pytest.mark.skipif(
    world.size != 1 and not compiled_with_sl(),
    reason='world.size != 1 and not compiled_with_sl()')


@pytest.mark.response
def test_do_GW_too(in_tmp_dir):
    atoms = bulk('BN', 'zincblende', a=3.615)
    ref_gap = 4.7747085087305425
    calc = GPAW(mode=PW(400),
                kpts={'size': (2, 2, 2), 'gamma': True},
                xc='LDA',
                eigensolver='rmm-diis',
                parallel={'domain': 1},
                occupations=FermiDirac(0.001))

    atoms.calc = calc
    atoms.get_potential_energy()

    calc.diagonalize_full_hamiltonian(scalapack=True)
    calc.write('BN_bulk_k2_ecut400_allbands.gpw', mode='all')

    gw = G0W0('BN_bulk_k2_ecut400_allbands.gpw',
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              xc='rALDA',
              method='G0W0',
              ecut=40,
              fxc_mode='GWP',
              do_GW_too=True)

    gw.calculate()

    with open('gw_results_GW.pckl', 'rb') as handle:
        results_GW = pickle.load(handle)
    calculated_gap = np.min(results_GW['qp'][0, :, 1])\
        - np.max(results_GW['qp'][0, :, 0])
    assert calculated_gap == pytest.approx(ref_gap)
