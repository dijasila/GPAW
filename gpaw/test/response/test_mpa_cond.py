import pytest
from gpaw.response.MPAinterpolation import mpa_RE_solver
import numpy as np


@pytest.mark.response
def test_rcond():
    omega = np.array([0. + 3.67493225e-12j, 0. + 1.00000000e+00j])
    einv = np.array([7.26813572e-10 + 8.02893598e-10j,
                     3.11863353e-10 + 1.16709471e-09j])

    omegat_n, R_n, MPred, PPcond_rate = mpa_RE_solver(1, omega, einv)
    assert omegat_n == pytest.approx(-4.193989600054e-10 - 6.93450352695e-10j)
    assert R_n == pytest.approx(1.2565709652955386 - 1e-08j)
    assert PPcond_rate == 1
