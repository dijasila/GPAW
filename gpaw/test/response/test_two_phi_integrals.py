import pytest
from gpaw import GPAW
from gpaw.response.paw import PAWPairDensityCalculator
import numpy as np


@pytest.mark.response
def test_two_phi_integrals(gpw_files):
    calc = GPAW(gpw_files['bn_pw'])

    setup = calc.wfs.setups[0]
    k_Gv = np.array([[0.0, 0.0, 0.0]])
    pair_density_calc = PAWPairDensityCalculator(pawdata=setup)
    dO_aii = pair_density_calc(k_Gv)
    assert dO_aii[0] == pytest.approx(setup.dO_ii, 1e-8, 1e-7)
