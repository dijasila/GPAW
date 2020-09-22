import numpy as np
import pytest

from gpaw.grid_descriptor import GridDescriptor
from gpaw.hyperfine import hyperfine_parameters, paw_correction, smooth_part
from gpaw import GPAW
from gpaw.atom.radialgd import RadialGridDescriptor
from gpaw.lfc import LFC


def atest_h(gpw_files):
    calc = GPAW(gpw_files['h_pw'])
    print(hyperfine_parameters(calc))


def test_o2(gpw_files):
    calc = GPAW(gpw_files['o2_pw'])
    print(hyperfine_parameters(calc))


class Setup:
    def __init__(self):
        self.rgd = RadialGridDescriptor.new('equidistant', 0.01, 400)
        self.data = Data(self.rgd)
        self.l_j = [0, 1]


class Data:
    def __init__(self, rgd: RadialGridDescriptor):
        self.phit_jg = [np.exp(-(rgd.r_g / 0.5)**2),
                        np.exp(-(rgd.r_g / 0.5)**2) * rgd.r_g]
        self.phi_jg = [rgd.zeros(), rgd.zeros()]


@pytest.fixture
def things():
    setup = Setup()

    n = 20
    L = n * 0.2
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

    W1_a, W2_avv = smooth_part(spin_denisty_R, gd, spos_ac)
    print(W1_a, W2_avv)

    spin_denisty_ii = np.zeros((4, 4))
    spin_denisty_ii[0, 0] = 1.0
    W1, W2_vv = paw_correction(spin_denisty_ii, setup)
    print(W1, W2_vv)


@pytest.mark.serial
def test_gaussian2(things):
    gd, lfc, setup, spos_ac = things
    spin_denisty_R = gd.zeros()
    lfc.add(spin_denisty_R, {0: np.array([0, 1.0, 0, 0])})
    spin_denisty_R *= spin_denisty_R

    W1_a, W2_avv = smooth_part(spin_denisty_R, gd, spos_ac)
    print(W1_a, W2_avv)

    spin_denisty_ii = np.zeros((4, 4))
    spin_denisty_ii[1, 1] = 1.0
    W1, W2_vv = paw_correction(spin_denisty_ii, setup)
    print(W1, W2_vv)


