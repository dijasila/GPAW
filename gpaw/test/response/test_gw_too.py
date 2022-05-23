import pytest
from gpaw.mpi import world
import numpy as np
from gpaw.response.g0w0 import G0W0
import pickle


@pytest.mark.response
def test_do_GW_too(in_tmp_dir, gpw_files, scalapack):

    ecut_extrapolation = True
    gw0 = G0W0(gpw_files['bn_pw_wfs'],
               bands=(3, 5),
               nbands=9,
               nblocks=1,
               ecut_extrapolation=ecut_extrapolation,
               ecut=40,
               restartfile=None)

    results0 = gw0.calculate()

    gw = G0W0(gpw_files['bn_pw_wfs'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              xc='rALDA',
              ecut_extrapolation=ecut_extrapolation,
              ecut=40,
              fxc_mode='GWP',
              do_GW_too=True,
              restartfile=None)

    gw.calculate()

    world.barrier()

    with open('gw_results_GW.pckl', 'rb') as handle:
        results_GW = pickle.load(handle)
    np.testing.assert_allclose(results0['qp'], results_GW['qp'], rtol=1e-03)
