from typing import TYPE_CHECKING

from gpaw.typing import Array1D, ArrayND


def add_to_density(f: float,
                   psit_X: ArrayND,
                   nt_X: ArrayND) -> None:
    nt_X += f * abs(psit_X)**2


def pw_precond(G2_G: Array1D,
               r_G: Array1D,
               ekin: float,
               o_G: Array1D) -> None:
    x = 1 / ekin / 3 * G2_G
    a = 27.0 + x * (18.0 + x * (12.0 + x * 8.0))
    xx = x * x
    o_G[:] = -4.0 / 3 / ekin * a / (a + 16.0 * xx * xx) * r_G


def pw_insert(coef_G: Array1D,
              Q_G: Array1D,
              x: float,
              array_Q: Array1D) -> None:
    array_Q[:] = 0.0
    array_Q.ravel()[Q_G] = x * coef_G


if not TYPE_CHECKING:
    try:
        from _gpaw import add_to_density, pw_precond, pw_insert  # noqa
    except ImportError:
        pass
