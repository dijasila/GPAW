import numpy as np
import pytest
from scipy.special import expn
import ase.units as units

from gpaw.grid_descriptor import GridDescriptor
from gpaw.hyperfine import (hyperfine_parameters, paw_correction, smooth_part,
                            integrate, alpha, g_factor_e)
from gpaw import GPAW
from gpaw.atom.radialgd import RadialGridDescriptor
from gpaw.lfc import LFC


@pytest.mark.serial
def test_thomson_integral():
    dr = 0.0001
    rgd = RadialGridDescriptor.new('equidistant', dr, 16000)
    beta = 0.0
    rT = 0.001
    n0_g = np.exp(-rgd.r_g)

    n0 = integrate(n0_g, rgd, rT, beta)
    ref = expn(2, rT / 2) * np.exp(rT / 2)
    assert n0 == pytest.approx(ref, 1e-5)


@pytest.mark.serial
def test_thomson_integral2():
    dr = 0.0001
    rgd = RadialGridDescriptor.new('equidistant', dr, 66000)
    rT = 0.001
    beta = 0.001
    n0_g = rgd.zeros()
    n0_g[1:] = rgd.r_g[1:]**-beta
    n0_g[0] = np.nan

    n0 = integrate(n0_g, rgd, rT, beta)
    ref = (2 / rT)**beta * np.pi * rT / np.sin(np.pi * rT)
    assert n0 == pytest.approx(ref, 1e-4)


class Setup:
    def __init__(self):
        self.rgd = RadialGridDescriptor.new('equidistant', 0.01, 400)
        self.data = Data(self.rgd)
        self.l_j = [0, 1]
        self.Z = 1


class Data:
    def __init__(self, rgd: RadialGridDescriptor):
        self.phit_jg = [np.exp(-(rgd.r_g / 0.5)**2),
                        np.exp(-(rgd.r_g / 0.5)**2) * rgd.r_g]
        self.phi_jg = [rgd.zeros(), rgd.zeros()]


@pytest.fixture
def things():
    setup = Setup()

    n = 40
    L = n * 0.1
    gd = GridDescriptor((n, n, n), (L, L, L), (1, 1, 1))

    splines = []
    for j, phit_g in enumerate(setup.data.phit_jg):
        splines.append(setup.rgd.spline(phit_g, 1.5, l=j, points=101))

    lfc = LFC(gd, [splines])

    spos_ac = np.zeros((1, 3)) + 0.5
    lfc.set_positions(spos_ac)

    return gd, lfc, setup, spos_ac


@pytest.mark.serial
def test_gaussian(things):
    gd, lfc, setup, spos_ac = things
    spin_denisty_R = gd.zeros()
    lfc.add(spin_denisty_R, {0: np.array([1.0, 0, 0, 0])})
    spin_denisty_R *= spin_denisty_R

    W_avv = smooth_part(spin_denisty_R, gd, spos_ac)
    print(W_avv)
    assert abs(W_avv[0] - np.eye(3) * W_avv[0, 0, 0]).max() < 1e-7

    spin_denisty_ii = np.zeros((4, 4))
    spin_denisty_ii[0, 0] = 1.0
    W1_vv = paw_correction(spin_denisty_ii, setup)
    print(W1_vv)
    assert abs(W_avv[0] + W1_vv).max() < 1e-7


@pytest.mark.serial
def test_gaussian2(things):
    gd, lfc, setup, spos_ac = things
    spin_denisty_R = gd.zeros()
    lfc.add(spin_denisty_R, {0: np.array([0, 0, 1.0, 0])})
    spin_denisty2_R = gd.zeros()
    lfc.add(spin_denisty2_R, {0: np.array([0, 1.0, 0, 0])})
    spin_denisty_R = spin_denisty_R * spin_denisty2_R

    W_avv = smooth_part(spin_denisty_R, gd, spos_ac)
    print(W_avv)
    assert abs(W_avv[0] - np.array([[0, 0, 0],
                                    [0, 0, 1],
                                    [0, 1, 0]]) * W_avv[0, 1, 2]).max() < 1e-7

    spin_denisty_ii = np.zeros((4, 4))
    spin_denisty_ii[2, 1] = 1.0
    W1_vv = paw_correction(spin_denisty_ii, setup)
    print(W1_vv)
    assert abs(W_avv[0] + W1_vv).max() < 1e-6


g_factor_proton = 5.586


@pytest.mark.serial
def test_h(gpw_files):
    calc = GPAW(gpw_files['h_pw'])
    A_vv = hyperfine_parameters(calc)[0] * g_factor_proton
    print(A_vv)

    energy = (2 / 3 * alpha**2 * units.Ha * units._me / units._mp *
              g_factor_e * g_factor_proton)  # in eV
    frequency = energy * units._e / units._hplanck  # Hz
    wavelength = units._c / frequency  # meters
    assert wavelength == pytest.approx(0.211, abs=0.001)

    energy *= 0.94  # density at nucleus is slightly below 1/pi for PBE
    print(energy)
    assert abs(A_vv - np.eye(3) * energy).max() < 1e-7


def thomson():
    """Analytic integrals for testing."""
    from sympy import var, integrate, oo, E, expint
    x, a, b = var('x, a, b')
    print(integrate(E**(-b * x) / (1 + x)**2, (x, 0, oo)))
    print(expint(2, 1.0))
