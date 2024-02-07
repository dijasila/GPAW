""" Tests extrapolation to infinite energy cutoff + block parallelization.
It takes ~10 s on one core"""

import pytest
from gpaw.response.g0w0 import G0W0
from gpaw.mpi import world
import numpy as np
from ase.units import Bohr, Hartree
from gpaw import GPAW


@pytest.mark.response
def test_response_gw_hBN_extrapolate(in_tmp_dir, scalapack, gpw_files,
                                     needs_ase_master, gpaw_new):
    if gpaw_new and world.size > 1:
        pytest.skip('Hybrids not working in parallel with GPAW_NEW=1')
    gw = G0W0(gpw_files['hbn_pw'],
              'gw-hBN',
              ecut=30,
              frequencies={'type': 'nonlinear',
                           'domega0': 0.1},
              eta=0.2,
              truncation='2D',
              q0_correction=True,
              kpts=[0],
              bands=(3, 5),
              ecut_extrapolation=[20, 25, 30],
              nblocksmax=True)

    results = gw.calculate()
    e_qp = results['qp'][0, 0]

    ev = -1.4528
    ec = 3.69469
    assert e_qp[0] == pytest.approx(ev, abs=0.01)
    assert e_qp[1] == pytest.approx(ec, abs=0.01)

    calc = GPAW(gpw_files['hbn_pw'])
    volume = calc.atoms.cell.volume / (Bohr**3)
    for ie, ecut in enumerate([20, 25, 30]):
        gw = G0W0(gpw_files['hbn_pw'],
                  f'gw-hBN-separate-ecut{ecut}',
                  ecut=ecut,
                  frequencies={'type': 'nonlinear',
                               'omegamax': 43.2,  # We need same grid as above
                               'domega0': 0.1},
                  eta=0.2,
                  truncation='2D',
                  q0_correction=True,
                  kpts=[0],
                  bands=(3, 5),
                  ecut_extrapolation=False,
                  nblocksmax=True)
        res = gw.calculate()
        assert np.allclose(res['sigma_eskn'][0], results['sigma_eskn'][ie])
