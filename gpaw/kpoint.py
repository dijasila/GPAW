# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a ``KPoint`` class and the derived ``GlobalKPoint``."""

import numpy as np


class KPoint:
    """Class for a single k-point.

    The KPoint class takes care of all wave functions for a
    certain k-point and a certain spin.

    XXX This needs to be updated.

    Attributes:

    phase_cd: complex ndarray
        Bloch phase-factors for translations - axis c=0,1,2
        and direction d=0,1.
    eps_n: float ndarray
        Eigenvalues.
    f_n: float ndarray
        Occupation numbers.  The occupation numbers already include the
        k-point weight in supercell calculations.
    psit_nG: ndarray
        Wave functions.
    nbands: int
        Number of bands.

    Parallel stuff:

    comm: Communicator object
        MPI-communicator for domain.
    root: int
        Rank of the CPU that does the matrix diagonalization of
        H_nn and the Cholesky decomposition of S_nn.
    """

    def __init__(self, weight, s, k, q, phase_cd):
        """Construct k-point object.

        Parameters:

        gd: GridDescriptor object
            Descriptor for wave-function grid.
        weight: float
            Weight of this k-point.
        s: int
            Spin index: up or down (0 or 1).
        k: int
            k-point index.
        q: int
            local k-point index.
        dtype: type object
            Data type of wave functions (float or complex).
        timer: Timer object
            Optional.

        Note that s and k are global spin/k-point indices,
        whereas u is a local spin/k-point pair index for this
        processor.  So if we have `S` spins and `K` k-points, and
        the spins/k-points are parallelized over `P` processors
        (kpt_comm), then we have this equation relating s,
        k and u::

           rSK
           --- + u = sK + k,
            P

        ???where `r` is the processor rank within kpt_comm.  The
        total number of spin/k-point pairs, `SK`, is always a
        multiple of the number of processors, `P`?????
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

    @property
    def P_ani(self):
        if self.projections is not None:
            return {a: P_ni for a, P_ni in self.projections.items()}

    @property
    def psit_nG(self):
        if self.psit is not None:
            return self.psit.array
