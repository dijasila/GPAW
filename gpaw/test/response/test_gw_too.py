import pytest
from gpaw.mpi import world
import numpy as np
from gpaw.response.g0w0 import G0W0
from gpaw import GPAW, PW
from ase.build import bulk
import pickle


@pytest.mark.response
def test_do_GW_too(in_tmp_dir, gpw_files, scalapack):
    atoms = bulk('C')
    atoms.center()
    calc = GPAW(mode=PW(200),
                convergence={'bands': 6},
                nbands=12,
                kpts={'gamma': True, 'size': (2, 2, 2)},
                xc='LDA')

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs-gw.wfs', 'all')

    ecut_extrapolation = True
    gw0 = G0W0('gs-gw.wfs', 'gw0',
               bands=(3, 5),
               nblocks=1,
               ecut_extrapolation=ecut_extrapolation,
               ecut=40,
               restartfile=None)

    results0 = gw0.calculate()

    gw = G0W0('gs-gw.wfs', 'gwtoo',
              bands=(3, 5),
              nblocks=1,
              xc='rALDA',
              ecut_extrapolation=ecut_extrapolation,
              ecut=40,
              fxc_mode='GWP',
              do_GW_too=True,
              restartfile=None)

    gw.calculate()

    world.barrier()

    files = gw.savepckl()

    with open(files['GW'], 'rb') as handle:
        results_GW = pickle.load(handle)
    np.testing.assert_allclose(results0['qp'], results_GW['qp'], rtol=1e-03)
