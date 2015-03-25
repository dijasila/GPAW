import numpy as np

from gpaw import mixer
from gpaw.utilities.blas import axpy
from gpaw.fd_operators import FDOperator


class BaseMixer(mixer.BaseMixer):
    """Pulay density mixer."""

    def __init__(self, beta=0.1, nmaxold=3, weight=50.0, dotprod=None,
                 dtype=float):
        """Construct density-mixer object.

        Parameters
        ----------
        beta: float
            Mixing parameter between zero and one (one is most
            aggressive).
        nmaxold: int
            Maximum number of old densities.
        weight: float
            Weight parameter for special metric (for long wave-length
            changes).

        """

        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight
        self.dtype = dtype

        self.dNt = None

        self.mix_rho = False

        if dotprod is not None:  # slightly ugly way to override
            self.dotprod = dotprod

    def initialize_metric(self, gd):
        self.gd = gd

        if self.weight == 1:
            self.metric = None
        else:
            a = 0.125 * (self.weight + 7)
            b = 0.0625 * (self.weight - 1)
            c = 0.03125 * (self.weight - 1)
            d = 0.015625 * (self.weight - 1)
            self.metric = FDOperator([a,
                                      b, b, b, b, b, b,
                                      c, c, c, c, c, c, c, c, c, c, c, c,
                                      d, d, d, d, d, d, d, d],
                                     [(0, 0, 0),
                                      (-1, 0, 0), (1, 0, 0),                 #b
                                      (0, -1, 0), (0, 1, 0),                 #b
                                      (0, 0, -1), (0, 0, 1),                 #b
                                      (1, 1, 0), (1, 0, 1), (0, 1, 1),       #c
                                      (1, -1, 0), (1, 0, -1), (0, 1, -1),    #c
                                      (-1, 1, 0), (-1, 0, 1), (0, -1, 1),    #c
                                      (-1, -1, 0), (-1, 0, -1), (0, -1, -1), #c
                                      (1, 1, 1), (1, 1, -1), (1, -1, 1),     #d
                                      (-1, 1, 1), (1, -1, -1), (-1, -1, 1),  #d
                                      (-1, 1, -1), (-1, -1, -1)              #d
                                      ],
                                     gd, self.dtype).apply
            self.mR_G = gd.empty(dtype=self.dtype)

    def mix(self, nt_G, D_ap, phase_cd=None):
        iold = len(self.nt_iG)
        if iold > 0:
            if iold > self.nmaxold:
                # Throw away too old stuff:
                del self.nt_iG[0]
                del self.R_iG[0]
                del self.D_iap[0]
                del self.dD_iap[0]
                # for D_p, D_ip, dD_ip in self.D_a:
                #     del D_ip[0]
                #     del dD_ip[0]
                iold = self.nmaxold

            # Calculate new residual (difference between input and output)
            R_G = nt_G - self.nt_iG[-1]
            # Use np.absolute instead of np.fabs
            self.dNt = self.gd.integrate(np.absolute(R_G))
            self.R_iG.append(R_G)
            self.dD_iap.append([])
            for D_p, D_ip in zip(D_ap, self.D_iap[-1]):
                self.dD_iap[-1].append(D_p - D_ip)

            # Update matrix:
            A_ii = np.zeros((iold, iold))
            i1 = 0
            i2 = iold - 1

            if self.metric is None:
                mR_G = R_G
            else:
                mR_G = self.mR_G
                self.metric(R_G, mR_G, phase_cd=phase_cd)

            for R_1G in self.R_iG:
                # Inner product between new and old residues
                # XXX For now, use only real part of residues
                # For complex quantities a .conjugate should be added ??
                a = self.gd.comm.sum(np.vdot(R_1G.real, mR_G.real))
                if self.dtype == complex:
                    a += self.gd.comm.sum(np.vdot(R_1G.imag, mR_G.imag))

                A_ii[i1, i2] = a
                A_ii[i2, i1] = a
                i1 += 1
            A_ii[:i2, :i2] = self.A_ii[-i2:, -i2:]
            self.A_ii = A_ii

            try:
                B_ii = np.linalg.inv(A_ii)
            except np.linalg.LinAlgError:
                alpha_i = np.zeros(iold)
                alpha_i[-1] = 1.0
            else:
                alpha_i = B_ii.sum(1)
                try:
                    # Normalize:
                    alpha_i /= alpha_i.sum()
                except ZeroDivisionError:
                    alpha_i[:] = 0.0
                    alpha_i[-1] = 1.0

            # Calculate new input density:
            nt_G[:] = 0.0

            for D in D_ap:
                D[:] = 0.0
            beta = self.beta

            for i, alpha in enumerate(alpha_i):
                axpy(alpha, self.nt_iG[i], nt_G)
                axpy(alpha * beta, self.R_iG[i], nt_G)
                for D_p, D_ip, dD_ip in zip(D_ap, self.D_iap[i],
                                            self.dD_iap[i]):
                    axpy(alpha, D_ip, D_p)
                    axpy(alpha * beta, dD_ip, D_p)

        # Store new input density (and new atomic density matrices):
        self.nt_iG.append(nt_G.copy())
        self.D_iap.append([])
        for D_p in D_ap:
            self.D_iap[-1].append(D_p.copy())
