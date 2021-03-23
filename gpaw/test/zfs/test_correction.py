import numpy as np
import pytest
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.grid_descriptor import GridDescriptor
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
from gpaw.spline import Spline
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.zero_field_splitting.zfs import zfs2


def wave_functions(phi, rgd, gd):
    splines = [Spline(l=0, rmax=rgd.r_g[-1], f_g=phi),
               Spline(l=1, rmax=rgd.r_g[-1], f_g=phi)]
    lfc = LFC(gd, [splines])
    # d = 2.1
    # L = gd.cell_cv[0, 0]
    # z = 0.5 * d / L
    lfc.set_positions([(0, 0, 0.5)])
    psi = gd.zeros(2)
    dct = {0: np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])}
    lfc.add(psi, dct)
    return psi


def mddc(pd, rho1, rho2):
    G_G = pd.G2_qG[0]**0.5
    G_G[0] = 1.0
    G_Gv = pd.get_reciprocal_vectors() / G_G[:, np.newaxis]

    r1 = pd.fft(rho1)
    r2 = pd.fft(rho2)
    D = zfs2(pd, G_Gv, r1 * r2.conj())
    return D


def test_paw_correction():
    rc = 1.0
    rgd = EquidistantRadialGridDescriptor(0.05, 200)
    r = rgd.r_g
    phi = np.exp(-r**2)  # - 1.5 * np.exp(-r**2 * 7)
    phi /= rgd.integrate(phi**2)**0.5
    phit, _ = rgd.pseudize_normalized(phi, rgd.floor(rc))
    assert rgd.integrate(phit**2) == pytest.approx(1.0)

    if 0:
        rgd.plot(phi)
        rgd.plot(phit, show=1)

    a = 5.5 * rc * 1
    n = 32 * 1
    gd = GridDescriptor([n, n, n], [a, a, a])
    pd = PWDescriptor(ecut=None, gd=gd)

    import matplotlib.pyplot as plt
    for p in [phi, phit]:
        psi1, psi2 = wave_functions(p, rgd, gd)
        plt.plot(psi1[0, 0])
        plt.plot(psi2[0, 0])

        rho1 = psi1 * psi2

        D = mddc(pd, rho1, rho1)
        print(D.real.diagonal())
        print(D)
        break

    plt.show()
