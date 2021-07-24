import numpy as np
import pytest
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.spherical_harmonics import second_derivatives
from gpaw.spline import Spline
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.zero_field_splitting.zfs import (integral_greater, integral_lesser,
                                           zfs2)
from gpaw.gaunt import gaunt


def wave_functions(phi, rgd, gd):
    splines = [Spline(l=0, rmax=rgd.r_g[-1], f_g=phi),
               Spline(l=1, rmax=rgd.r_g[-1], f_g=phi)]
    lfc = LFC(gd, [splines])
    # d = 2.1
    # L = gd.cell_cv[0, 0]
    # z = 0.5 * d / L
    lfc.set_positions([(0., 0., 0.)])
    psi = gd.zeros(2)
    dct = {0: np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])}
    lfc.add(psi, dct)
    return psi


def mddc(pd, rho1, rho2):
    G_G = pd.G2_qG[0]**0.5
    G_G[0] = 1.0
    G_Gv = pd.get_reciprocal_vectors() / G_G[:, np.newaxis]
    # G_Gv[0] = 1 / 3**.5

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

    a = 5.5 * rc * 3
    n = 32 * 3
    gd = GridDescriptor([n, n, n], [a, a, a])
    pd = PWDescriptor(ecut=None, gd=gd, dtype=complex)

    # import matplotlib.pyplot as plt
    for p in [phi, phit]:
        psi1, psi2 = wave_functions(p, rgd, gd)
        # plt.plot(psi1[0, 0])
        # plt.plot(psi2[0, 0])

        rho11 = psi1 * psi1
        rho12 = psi1 * psi2
        rho22 = psi2 * psi2

        D = mddc(pd, rho11, rho22)
        D12 = mddc(pd, rho12, rho12)
        print()
        print(D.real.diagonal())
        print(D12.real.diagonal())
        print((D - D12).real.diagonal())
        break

    Z = second_derivatives(0, kind='greater')
    print(Z[0, :, 0, 0])
    print(Z[0, :, 1, 1])
    print(Z[0, :, 2, 2])
    G = gaunt(1)
    i1 = integral_greater(phi * phi, phi * phi * r**2, 0, r, rgd.dr_g)
    d = G[0, 0, 0] * i1 * Z[0].T @ G[3, 3, 4:9] * 4 * np.pi
    print(d)
    # plt.show()
