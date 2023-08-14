from typing import TYPE_CHECKING

import numpy as np


def add_to_density(f: float, psit_X: np.ndarray, nt_X: np.ndarray) -> None:
    nt_X += f * abs(psit_X)**2


def pw_precond(G2_G: np.ndarray,
               r_G: np.ndarray,
               ekin: float,
               o_G: np.ndarray) -> None:
    x = 1 / ekin / 3 * G2_G
    a = 27.0 + x * (18.0 + x * (12.0 + x * 8.0))
    xx = x * x
    o_G[:] = -4.0 / 3 / ekin * a / (a + 16.0 * xx * xx) * r_G


if not TYPE_CHECKING:
    try:
        from _gpaw import add_to_density, pw_precond  # noqa
    except ImportError:
        pass
