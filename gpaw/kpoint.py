# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a ``KPoint`` class."""


class KPoint:
    """Class for a single k-point.

    The KPoint class takes care of all wave functions for a
    certain k-point and a certain spin.

    Attributes:

    eps_n: float ndarray
        Eigenvalues.
    f_n: float ndarray
        Occupation numbers.  The occupation numbers already include the
        k-point weight in supercell calculations.
    """

    def __init__(self, weight: float, s: int, k: int, q: int, phase_cd):
        """Construct k-point object.

        Parameters:

        weight: float
            Weight of this k-point.
        s: int
            Spin index: up or down (0 or 1).
        k: int
            k-point IBZ-index.
        q: int
            local k-point index.
        phase_cd: complex ndarray
            Bloch phase-factors for translations - axis c=0,1,2
            and direction d=0,1.
        """

        self.weight = weight
        self.s = s  # spin index
        self.k = k  # k-point index
        self.q = q  # local k-point index
        self.phase_cd = phase_cd

        self.eps_n = None
        self.f_n = None
        self.projections = None  # Projections

        # Only one of these two will be used:
        self.psit = None  # UniformGridMatrix/PWExpansionMatrix
        self.C_nM = None  # LCAO coefficients for wave functions

        # LCAO stuff:
        self.rho_MM = None
        self.S_MM = None
        self.T_MM = None

    def __repr__(self):
        return (f'KPoint(weight={self.weight}, '
                f's={self.s}, k={self.k}, q={self.q})')

    @property
    def P_ani(self):
        if self.projections is not None:
            return {a: P_ni for a, P_ni in self.projections.items()}

    @property
    def psit_nG(self):
        if self.psit is not None:
            return self.psit.array
