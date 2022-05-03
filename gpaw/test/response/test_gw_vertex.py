import pytest
from gpaw.mpi import world
from gpaw.utilities import compiled_with_sl
import numpy as np
from gpaw.response.g0w0 import G0W0

pytestmark = pytest.mark.skipif(
    world.size != 1 and not compiled_with_sl(),
    reason='world.size != 1 and not compiled_with_sl()')


@pytest.mark.response
@pytest.mark.parametrize('fxc_mode, ref_gap', [('GWP', 4.6672),
                                               ('GWS', 4.9882)])
def test_fxc_mode(in_tmp_dir, fxc_mode, ref_gap, gpw_files):
    gw = G0W0(gpw_files['bn_pw_wfs'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              xc='rALDA',
              method='G0W0',
              ecut=40,
              fxc_mode=fxc_mode)
    result = gw.calculate()
    calculated_gap = (np.min(result['qp'][0, :, 1]) -
                      np.max(result['qp'][0, :, 0]))
    assert calculated_gap == pytest.approx(ref_gap, abs=0.001)
