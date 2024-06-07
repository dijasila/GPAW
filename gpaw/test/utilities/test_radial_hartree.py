import numpy as np
import pytest
from gpaw.utilities import py_radial_hartree
from scipy.special import erf, exp1


def test_py_radial_hartree():
    """
    take a Guassian, calculate radial part analytically and
    compare it to numerical results
    """

    def gaussian(r_g: np.ndarray, l: int, alpha: float):
        prefactor = (
            (4 * alpha) ** (l + 3 / 2)
            * np.math.factorial(l)
            / (np.math.factorial(2 * l + 1) * np.sqrt(4 * np.pi))
        )
        return r_g**l * np.exp(-alpha * r_g**2) * prefactor

    def potential_00(r_g, alpha):
        r2_g = r_g.copy()
        r2_g[0] = 1
        pot = erf(r_g * np.sqrt(alpha)) / r2_g
        pot[0] = 2 * np.sqrt(alpha / np.pi)
        return pot * 4 * np.pi

    def potential_20(r_g, alpha):
        r = r_g.copy()
        r[0] = 1

        pot = 3 * erf(r * np.sqrt(alpha)) / (2 * alpha) + np.exp(
            -(r**2) * alpha
        ) * r * (
            -3
            - 2 * r**2 * alpha
            + 2
            * np.exp(r**2 * alpha)
            * r**4
            * alpha**2
            * exp1(r**2 * alpha)
        ) / np.sqrt(
            np.pi * alpha
        )
        pot[0] = 0
        return (pot / r**3) * 4 * np.pi / 5

    dr = 0.01
    r_g = np.arange(1000.0) * dr
    # take a gaussian for l = 0
    n_g = gaussian(r_g, 0, 1)

    # analytical solutions of radial hartree
    # for l=0, and l=2 for a gaussian with l=0
    pot_ref = np.asarray([potential_00(r_g, 1), potential_20(r_g, 1)])

    # numerical solutions for l=0,2 for a gaussian with l=0
    pot_num = np.asarray([py_radial_hartree(l, n_g, r_g) for l in [0, 2]])

    assert pot_num == pytest.approx(pot_ref, 0.1)
