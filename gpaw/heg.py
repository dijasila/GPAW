"""Analytic expressions of the homogeneous electron gas (HEG).

The goal of this module is to centralize all the standard expressions
needed around GPAW for doing HEG things.  Please move appropriate
code here whenever relevant."""

import numpy as np
from ase.utils import lazyproperty


class HEG:
    def __init__(self, rs):
        self._rs = rs

    @property
    def rs(self):
        return self._rs

    @lazyproperty
    def qF(self):
        return (9.0 * np.pi / 4.0)**(1.0 / 3.0) / self.rs

    def lindhard_function(self, q, u):
        """Lindhard function at imaginary frequency u."""
        qF = self.qF

        Q = q / 2.0 / qF
        U = u / q / qF
        lchi = ((Q * Q - U * U - 1.0) / 4.0 / Q * np.log(
            (U * U + (Q + 1.0) * (Q + 1.0)) / (U * U + (Q - 1.0) * (Q - 1.0))))
        lchi += -1.0 + U * np.arctan((1.0 + Q) / U) + U * np.arctan(
            (1.0 - Q) / U)
        lchi *= qF / (2.0 * np.pi * np.pi)
        return lchi
