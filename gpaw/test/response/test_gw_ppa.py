import pytest
import numpy as np
from gpaw.response.g0w0 import G0W0

from ase.units import Hartree as Ha

def Whook(gw, Wdict):
    f = open(f'Wmodel_ppa{gw.ppa}_mpa{True if gw.mpa else False}.txt', 'w')
    for occ in [0,1]:
        for w in np.linspace(-2, 2, 400):
            S_GG, dSdw_GG = Wmodel.get_HW(w, 2*occ-1, occ)
            if S_GG is None:
                continue
            print(occ, w, S_GG[0,0].real, S_GG[0,0].imag, S_GG[0,1].real, S_GG[0,1].imag,
                        dSdw_GG[0,0].real, dSdw_GG[0,0].imag, dSdw_GG[0,1].real, dSdw_GG[0,1].imag,
                        file=f)
        print(file=f)
        print(file=f)
        print(file=f)
    f.close()

@pytest.mark.response
def test_ff(in_tmp_dir, gpw_files, scalapack):
    ref_result = np.asarray([[[11.290542, 21.613646],
                              [ 5.356609, 16.065227],
                              [ 8.75117 , 23.156368]]])
    gw = G0W0(gpw_files['bn_pw'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              ecut=40)

    results = gw.calculate()
    np.testing.assert_allclose(results['qp'], ref_result, rtol=1e-03)


@pytest.mark.response
@pytest.mark.parametrize("mpa", [True,False])
def test_ppa(in_tmp_dir, gpw_files, scalapack, mpa):
    ref = {True: {'sigma': np.array([[[ 4.89101558, -4.17131973],
                                       [ 5.05514407, -3.99726968],
                                       [ 4.93276253, -4.46327288]]]),
                  'dsigma': np.array([[[-0.19163136, -0.17102652],
                                       [-0.20993491, -0.16145541],
                                       [-0.19894703, -0.18120556]]])},
           False: {'sigma': np.array([[[ 4.88189928, -4.16821152],
                                       [ 5.03747407, -3.99005103],
                                       [ 4.91928432, -4.45762978]]]),
                  'dsigma': np.array([[[-0.19045636, -0.17244613],
                                       [-0.20989414, -0.16195759],
                                       [-0.19874346, -0.18166906]]])}}
    mpa_dict = {'npoles':1, 'wrange':[1e-10j,1j*Ha], 'wshift':[0.1*Ha, 0.1*Ha], 'alpha':1 }

    gw = G0W0(gpw_files['bn_pw'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              ecut=80,
              ppa=not mpa,
              mpa=mpa_dict if mpa else False) 

    results = gw.calculate()
    np.testing.assert_allclose(ref[mpa]['sigma'], results['sigma'], rtol=1e-05)
    np.testing.assert_allclose(ref[mpa]['dsigma'], results['dsigma'], rtol=1e-05)


