import pytest
import numpy as np

from gpaw.grid_descriptor import GridDescriptor
from gpaw.xc import XC


def test_wda():
    a = 2.5
    n = 8
    gd = GridDescriptor((n, n, n), (a, a, a))
    d = gd.empty()[np.newaxis]
    v = gd.empty()[np.newaxis]
    e = gd.empty()
    d[:] = 0.123
    xc = XC('LDA')
    v[:] = 0.0
    elda = xc.calculate(gd, d, v, e)
    print(v[0, 0, 0])
    xc = XC('WDA')
    # xc = XC('PBE')
    v[:] = 0.0
    ewda = xc.calculate(gd, d, v, e)
    print(v[0, 0, 0])
    print(e[0, 0], e[0, 0, 0] * a**3)
    print(elda, ewda)
    assert ewda == pytest.approx(elda)


if __name__ == '__main__':
    test_wda()
