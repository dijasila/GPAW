import pytest
from gpaw.mpi import world
from gpaw.utilities import compiled_with_sl
import numpy as np
from ase.build import bulk
from gpaw import GPAW, FermiDirac, PW
from gpaw.response.g0w0 import G0W0
from gpaw.test import equal

pytestmark = pytest.mark.skipif(
    world.size != 1 and not compiled_with_sl(),
    reason='world.size != 1 and not compiled_with_sl()')


@pytest.mark.response
@pytest.mark.parametrize('fxc_mode, ref_gap', [
    ('GWP', 4.6672015836973255),
    ('GWS', 4.988260472871817),
])

def test_fxc_mode(in_tmp_dir,fxc_mode, ref_gap):
    atoms = bulk('BN', 'zincblende', a=3.615)

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
              fxc_mode=fxc_mode)

    result = gw.calculate()

    calculated_gap=np.min(result['qp'][0, :, 1]) - np.max(result['qp'][0, :, 0])
    assert calculated_gap == pytest.approx(ref_gap)
