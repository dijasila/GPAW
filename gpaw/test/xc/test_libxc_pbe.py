import pytest
import numpy as np
from gpaw.xc.kernel import XCKernel
from gpaw.xc.libxc import LibXC
from gpaw.xc.gga import PurePythonGGAKernel


@pytest.mark.ci
@pytest.mark.libxc
def test_pbe():
    n = np.zeros((1, 1)) + 0.23
    s = np.zeros((1, 1)) + 0.0001
    v = np.zeros((1, 1))
    w = np.zeros((1, 1))

    e1 = np.zeros(1)
    PurePythonGGAKernel('pyPBE').calculate(e1, n, v, s, w)

    e2 = np.zeros(1)
    XCKernel('PBE').calculate(e2, n, v, s, w)

    e3 = np.zeros(1)
    LibXC('PBE').calculate(e3, n, v, s, w)

    assert e2 == pytest.approx(e1, abs=1e-10)
    assert e3 == pytest.approx(e1, abs=1e-10)
