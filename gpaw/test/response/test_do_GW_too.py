import pytest
from gpaw.mpi import world
import numpy as np
from gpaw.response.g0w0 import G0W0
import pickle


@pytest.mark.response
def test_do_GW_too(in_tmp_dir, gpw_files, scalapack):
    ref_gap = 4.7747
    gw = G0W0(gpw_files['bn_pw_wfs'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              xc='rALDA',
              method='G0W0',
              ecut=40,
              fxc_mode='GWP',
              do_GW_too=True,
              restartfile=None)

    gw.calculate()

    world.barrier()

    with open('gw_results_GW.pckl', 'rb') as handle:
        results_GW = pickle.load(handle)
   
    np.testing.assert_array_equal(results0['qp'], results_GW['qp'],
                                  err_msg='G0W0 and do_GW_too not equivalent',
                                  verbose=True)
