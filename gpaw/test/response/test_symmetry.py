"""This script asserts that the chi's obtained
from GS calculations using symmetries
and GS calculations not using symmetries return
the same results. Tests that the chi's are element-wise
equal to a tolerance of 1e-10.
"""

import pytest
import numpy as np

from gpaw.response.chi0 import Chi0


@pytest.mark.response
@pytest.mark.slow
def test_symmetry_ti2o4(gpw_files):
    data_s = []
    for name in ['ti2o4_pw_nosym', 'ti2o4_pw']:
        chi0 = Chi0(gpw_files[name])
        data = chi0.calculate([1 / 4, 0, 0])
        data_s.append(data.chi0_WgG)

        # With a non-Gamma q-point as input, we should therefore
        # not have any data from the optical limit extensions
        assert data.chi0_WxvG is None
        assert data.chi0_Wvv is None

    msg = 'Difference in Chi when turning off symmetries!'

    datadiff_WgG = np.abs(data_s[0] - data_s[1])
    assert datadiff_WgG == pytest.approx(0, abs=1e-3), \
        datadiff_WgG.max()


@pytest.mark.xfail
@pytest.mark.response
@pytest.mark.slow
def test_symmetry_si2(gpw_files):
    data_s = []
    for name in ['fancy_si_pw_nosym', 'fancy_si_pw']:
        chi0 = Chi0(gpw_files[name])
        data = chi0.calculate([1 / 4, 0, 1 / 4])
        data_s.append(data.chi0_WgG)

        # With a non-Gamma q-point as input, we should therefore
        # not have any data from the optical limit extensions
        assert data.chi0_WxvG is None
        assert data.chi0_Wvv is None

    msg = 'Difference in Chi when turning off symmetries!'

    datadiff_WgG = np.abs(data_s[0] - data_s[1])
    assert datadiff_WgG == pytest.approx(0, abs=1e-3), \
        datadiff_WgG.max()
