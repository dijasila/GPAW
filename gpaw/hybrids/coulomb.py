import numpy as np

from math import pi


class ShortRangeCoulomb:
    def __init__(self, omega):
        self.omega = omega

    def get_potential(self, pd):
        G2_G = pd.G2_qG[0]
        x_G = 1 - np.exp(-G2_G / (4 * self.omega**2))
        with np.errstate(invalid='ignore'):
            v_G = 4 * pi * x_G / G2_G
        if pd.kd.gamma:
            v_G[0] = pi / self.omega**2
        return v_G
