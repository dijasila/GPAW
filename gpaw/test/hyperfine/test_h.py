import numpy as np
import pytest

from gpaw.grid_descriptor import GridDescriptor
from gpaw.hyperfine import hyperfine_parameters, paw_correction, smooth_part
from gpaw import GPAW
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor


def atest_h(gpw_files):
    calc = GPAW(gpw_files['h_pw'])
    print(hyperfine_parameters(calc))


def test_o2(gpw_files):
    calc = GPAW(gpw_files['o2_pw'])
    print(hyperfine_parameters(calc))


@pytest.mark.serial
def test_gaussian():
    n = 20
    L = n * 0.2
    gd = GridDescriptor((n, n, n), (L, L, L), (1, 1, 1))
    R_vR = gd.get_grid_point_coordinates()
    sd_R = np.exp(-(((R_vR - L / 2) / 0.5)**2).sum(0))
    W1_a, W2_avv = smooth_part(sd_R, gd, np.zeros((1, 3)) + L / 2)

    setup = Setup()
    W1, W2_vv = paw_correction(np.ones((1, 1)), setup)


class Setup:
    def __init__(self):
        self.rgd = EquidistantRadialGridDescriptor(0.01, 400)
        self.data = Data(self.rgd)
        self.l_j = [0]


class Data:
    def __init__(self, rgd):
        self.phit_jg = [np.exp(-(rgd.r_g / 0.5)**2)]
        self.phi_jg = [rgd.zeros()]
