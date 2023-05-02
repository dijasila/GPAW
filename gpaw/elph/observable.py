import numpy as np
from typing import Union

from gpaw.calculator import GPAW
from gpaw.typing import ArrayND


class Observable:
    """Base class for observables based on electron-phonon coupling.
    """
    def __init__(self, calc: GPAW, w_ql: Union[ArrayND, str],
                 elph: str = 'gsqklnn.npy') -> None:
        """Initialise.

            Parameters
            ----------
            calc: GPAW
                Converged ground state calculation
            w_ql: ndarray, str
                Array of phonon frequencies in eV, or name of file with them
        """
        self.calc = calc

        # Phonon frequencies
        if isinstance(w_ql, str):
            self.w_ql = np.load(w_ql)
        elif isinstance(w_ql, np.ndarray):
            self.w_ql = w_ql
        else:
            raise TypeError
        assert np.max(self.w_ql) < 1.  # else not eV units

        self.g_sqklnn = np.load(elph, mmap_mode='c')

    @classmethod
    def get_bose_factor(cls, w_l: ArrayND, T: float = 300.):
        """Get Bose-Einstein occupation.

        Parameters
        ----------
        w_l: ndarray
            array of phonon frequencies in eV
        T: float
            temperature in K (default 300K)
        """
        KtoeV = 8.617278e-5
        return 1. / (np.exp(w_l / (KtoeV * T)) - 1.)
