import numpy as np
import pytest

from gpaw.response.g0w0 import G0W0


reference_kn = [[-3.5495421, -3.4162368, -3.46210712,
                 -3.46286397, -3.33511845, 0.54208154],
                [-3.65617723, -3.65617723, -3.65613705,
                 -3.14292294, -3.14331901, 5.12060442]]
reference_kn = [[-3.5495421, -3.4162368, -3.46210712,
                 -3.46286397, -3.33511845, 0.54208154],
                [-3.65584223, -3.65617723, -3.65613705,
                 -3.14321622, -3.14331901, 5.12060442]]


@pytest.mark.response
def test_hubbard_GW(in_tmp_dir, gpw_files):
    # This tests checks the actual numerical accuracy which is asserted below
    gw = G0W0(gpw_files['ag_plusU_pw_wfs'], 'gw',
              integrate_gamma=0,
              frequencies={'type': 'nonlinear',
                           'domega0': 0.1, 'omegamax': None},
              eta=0.2)
    results = gw.calculate()

    qp_kn = results['qp'][0]

    assert np.allclose(qp_kn, reference_kn)
