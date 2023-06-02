import pytest
import numpy as np
from gpaw.response.g0w0 import G0W0

from ase.units import Hartree as Ha

@pytest.mark.response
@pytest.mark.parametrize("mpa", [True,False])
def test_ppa(in_tmp_dir, gpw_files, scalapack, mpa):
    ref_result = np.asarray([[[11.30094393, 21.62842077],
                              [5.33751513, 16.06905725],
                              [8.75269938, 22.46579489]]])
    mpa_dict = {'npoles':1, 'wrange':[1e-10j,1j*Ha], 'wshift':[0.1*Ha, 0.1*Ha], 'alpha':1 }

    gw = G0W0(gpw_files['bn_pw_wfs'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              ecut=40,
              ppa=not mpa,
              mpa=mpa_dict if mpa else False) 

    results = gw.calculate()
    np.testing.assert_allclose(results['qp'], ref_result, rtol=1e-03)
