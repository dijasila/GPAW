from gpaw.sphericcal_harmonics import YL
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
            assert abs(yLL(L1, L2) - r) < 1e-14
