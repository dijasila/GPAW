from math import pi
import pytest
from gpaw.sphericcal_harmonics import YL, Yl
from gpaw.gaunt import gam


def yLL(L1, L2):
    s = 0.0
    for c1, n1 in YL[L1]:
        for c2, n2 in YL[L2]:
            s += c1 * c2 * gam(n1[0] + n2[0], n1[1] + n2[1], n1[2] + n2[2])
    return s


def test_yy():
    Lmax = len(YL)
    for L1 in range(Lmax):
        for L2 in range(Lmax):
            r = 0.0
            if L1 == L2:
                r = 1.0
            assert yLL(L1, L2) == pytest.approx(r, abs=1e-14)


def test_y_c_code():
    assert Yl(0, 0) == pytest.approx((4 * pi)**-0.5)
    with pytest.raises():
        Yl(8, 0)
