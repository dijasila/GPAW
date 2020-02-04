import pytest
import numpy as np

from gpaw.grid_descriptor import GridDescriptor
from gpaw.xc import XC


def test_wda():
    a = 2.5
    n = 4
    gd = GridDescriptor((n, n, n), (a, a, a))
    d = gd.zeros()[np.newaxis]
    v = gd.zeros()[np.newaxis]
    e = gd.zeros()
    d[:] = 0.1
    xc = XC('LDA')
    elda = xc.calculate(gd, d, v, e)
    xc = XC('WDA')
    ewda = xc.calculate(gd, d, v, e)
    print(e[0, 0], e[0, 0, 0] * a**3)
    print(elda, ewda)
    assert ewda == pytest.approx(elda)


if __name__ == '__main__':
    test_wda()
