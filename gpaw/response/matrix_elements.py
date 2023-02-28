
import numpy as np

from gpaw.response import timer
from gpaw.response.paw import get_pair_density_paw_corrections
from gpaw.response.kspair import KohnShamKPointPair


class PairDensity:
    """Data class for transition distributed pair density arrays."""

    def __init__(self, tblocks, n_mytG):
        self.tblocks = tblocks
        self.n_mytG = n_mytG

    def get_global_array(self):
        """Get the global (all gathered) pair density array n_tG."""
        n_tG = self.tblocks.collect(self.n_mytG)

        return n_tG


class NewPairDensityCalculator:
    """Class for calculating pair densities

    n_kt(G+q) = n_nks,n'k+qs'(G+q) = <nks| e^-i(G+q)r |n'k+qs'>_V0

    for a single k-point pair (k,k+q) in the plane wave mode"""
    def __init__(self, gs, context):
        self.gs = gs
        self.context = context

        # Save PAW correction for all calls with same q_c
        self.pawcorr = None
        self.currentq_c = None

    @timer('Initialize PAW corrections')
    def initialize_paw_corrections(self, qpd):
        """Initialize PAW corrections, if not done already, for the given q"""
        q_c = qpd.q_c
        if self.pawcorr is None or not np.allclose(q_c - self.currentq_c, 0.):
            self.pawcorr = self._initialize_paw_corrections(qpd)
            self.currentq_c = q_c

    def _initialize_paw_corrections(self, qpd):
        pawdatasets = self.gs.pawdatasets
        spos_ac = self.gs.spos_ac
        return get_pair_density_paw_corrections(pawdatasets, qpd, spos_ac)

    @timer('Calculate pair density')
    def __call__(self, kptpair: KohnShamKPointPair, qpd) -> PairDensity:
        """Calculate the pair densities for all transitions t of the (k,k+q)
        k-point pair:

        n_kt(G+q) = <nks| e^-i(G+q)r |n'k+qs'>_V0

                    /
                  = | dr e^-i(G+q)r psi_nks^*(r) psi_n'k+qs'(r)
                    /V0
        """
        Q_aGii = self.get_paw_projectors(qpd)
        Q_G = self.get_fft_indices(kptpair, qpd)

        tblocks = kptpair.tblocks
        n_mytG = qpd.empty(tblocks.blocksize)

        # Calculate smooth part of the pair densities:
        with self.context.timer('Calculate smooth part'):
            ut1cc_mytR = kptpair.kpt1.ut_tR.conj()
            n_mytR = ut1cc_mytR * kptpair.kpt2.ut_tR
            # Unvectorized
            for myt in range(tblocks.nlocal):
                n_mytG[myt] = qpd.fft(n_mytR[myt], 0, Q_G) * qpd.gd.dv

        # Calculate PAW corrections with numpy
        with self.context.timer('PAW corrections'):
            P1 = kptpair.kpt1.projections
            P2 = kptpair.kpt2.projections
            for (Q_Gii, (a1, P1_myti),
                 (a2, P2_myti)) in zip(Q_aGii, P1.items(), P2.items()):
                P1cc_myti = P1_myti[:tblocks.nlocal].conj()
                C1_Gimyt = np.einsum('Gij, ti -> Gjt', Q_Gii, P1cc_myti)
                P2_imyt = P2_myti.T[:, :tblocks.nlocal]
                n_mytG[:tblocks.nlocal] += np.sum(
                    C1_Gimyt * P2_imyt[np.newaxis, :, :], axis=1).T

        return PairDensity(tblocks, n_mytG)

    def get_paw_projectors(self, qpd):
        """Make sure PAW correction has been initialized properly
        and return projectors"""
        self.initialize_paw_corrections(qpd)
        return self.pawcorr.Q_aGii

    @timer('Get G-vector indices')
    def get_fft_indices(self, kptpair, qpd):
        """Get indices for G-vectors inside cutoff sphere."""
        from gpaw.response.pair import fft_indices

        kpt1 = kptpair.kpt1
        kpt2 = kptpair.kpt2
        kd = self.gs.kd
        q_c = qpd.q_c

        return fft_indices(kd=kd, K1=kpt1.K, K2=kpt2.K, q_c=q_c, qpd=qpd,
                           shift0_c=kpt1.shift_c - kpt2.shift_c)
