import pytest
import numpy as np
from gpaw.response.g0w0 import G0W0
from ase.units import Hartree as Ha
#from gpaw.response.mpa_interpolation import mpa_cond1, pole_is_out

@pytest.mark.response
def test_mpa_WS_ecut_extrap(in_tmp_dir, gpw_files, scalapack):
    ref_result = np.asarray([[[ 12.715556,  21.389845],
                              [-33.660224,  16.462673],
                              [  9.755104,  22.868525]]])

    mpa_dict = {'npoles': 4, 'wrange': [0 * Ha, 2 * Ha],
                'varpi': Ha,
                'eta0': 0.01 * Ha,
                'eta_rest': 0.1 * Ha,
                'alpha': 1}

    gw = G0W0(gpw_files['bn_pw'],
              bands=(3, 5),
              nblocks=1,
              ecut=40,
              ecut_extrapolation=True,
              integrate_gamma='WS',
              ppa=False,
              mpa=mpa_dict)

    results = gw.calculate()
    print(results)
    np.testing.assert_allclose(results['qp'], ref_result, rtol=1e-03)


@pytest.mark.response
def test_mpa_WS(in_tmp_dir, gpw_files, scalapack):
    ref_result = np.asarray([[[11.335173, 21.581602],
                   [ 5.411369, 16.046205],
                   [ 8.79945 , 22.366197]]])

    mpa_dict = {'npoles': 4, 'wrange': [0 * Ha, 2 * Ha],
                'varpi': Ha,
                'eta0': 0.01 * Ha,
                'eta_rest': 0.1 * Ha,
                'alpha': 1}

    gw = G0W0(gpw_files['bn_pw'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              ecut=400,
              integrate_gamma='WS',
              ppa=False,
              mpa=mpa_dict)

    results = gw.calculate()
    print(results)
    np.testing.assert_allclose(results['qp'], ref_result, rtol=1e-03)

@pytest.mark.response
def test_mpa(in_tmp_dir, gpw_files, scalapack):
    ref_result = np.asarray([[[11.283458, 21.601906],
                              [ 5.326717, 16.066114],
                              [ 8.73869 , 22.457025]]])

    mpa_dict = {'npoles': 4, 'wrange': [0 * Ha, 2 * Ha],
                'varpi': Ha,
                'eta0': 0.01 * Ha,
                'eta_rest': 0.1 * Ha,
                'alpha': 1}

    gw = G0W0(gpw_files['bn_pw'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              ecut=40,
              ppa=False,
              mpa=mpa_dict)

    results = gw.calculate()
    np.testing.assert_allclose(results['qp'], ref_result, rtol=1e-03)

"""
def test_mpa_conditions():
    c = mpa_cond1(0, complex(4.0, 0.1))[0]
    assert np.allclose(c, complex(2.0001562194924314, -1e-8), atol=1e-10)
    E = [complex(1, -0.11), complex(1.05, 0.1), complex(2, 0.2),
         complex(5, 1)]
    bools = [False, True, False, True]
    for i in range(len(E)):
        assert pole_is_out(i, 3., 0.1, E) == bools[i]
        """
