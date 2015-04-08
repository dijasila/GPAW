import numpy as np

from gpaw import mixer
from gpaw.utilities.blas import axpy
from gpaw.fd_operators import FDOperator


class BaseMixer(mixer.BaseMixer):
    """Pulay density mixer."""

    # XXX This may be integrated into gpaw/mixer.py
    def calculate_charge_sloshing(self, R_G):
        if self.dtype == float:
            return self.gd.integrate(np.fabs(R_G))
        else:
            return self.gd.integrate(np.absolute(R_G))

    # XXX That could probably use some adjustments
    def dotprod(self, R1_G, R2_G, dD1_ap, dD2_ap):
        if self.dtype == float:
          return np.vdot(R1_G, R2_G).real
        else:
          return np.vdot(R1_G.real, R2_G.real) + np.vdot(R1_G.imag, R2_G.imag)
