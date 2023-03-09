
import numpy as np

from gpaw.response import timer
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
        self._pawcorr = None
        self._currentq_c = None

    def initialize_paw_corrections(self, qpd):
        """Initialize the PAW corrections ahead of the actual calculation."""
        self.get_paw_corrections(qpd)

    def get_paw_corrections(self, qpd):
        """Get PAW corrections correcsponding to a specific q-vector."""
        if self._pawcorr is None \
           or not np.allclose(qpd.q_c - self._currentq_c, 0.):
            self._pawcorr = self.gs.pair_density_paw_corrections(qpd)
            self._currentq_c = qpd.q_c

        return self._pawcorr

    @timer('Calculate pair density')
    def __call__(self, kptpair: KohnShamKPointPair, qpd) -> PairDensity:
        """Calculate the pair densities for all transitions t of the (k,k+q)
        k-point pair:

        n_kt(G+q) = <nks| e^-i(G+q)r |n'k+qs'>_V0

                    /
                  = | dr e^-i(G+q)r psi_nks^*(r) psi_n'k+qs'(r)
                    /V0
        """
        kpt1 = kptpair.kpt1
        kpt2 = kptpair.kpt2

        # Fourier transform the pseudo waves to the coarse real-space grid
        # and symmetrize them along with the projectors
        P1h, ut1_hR, shift1_c = self.gs.transform_and_symmetrize(
            *kpt1.get_orbitals())
        P2h, ut2_hR, shift2_c = self.gs.transform_and_symmetrize(
            *kpt2.get_orbitals())

        # Get the plane-wave indices to Fourier transform products of
        # Kohn-Sham orbitals in k and k + q
        dshift_c = shift1_c - shift2_c
        Q_G = self.get_fft_indices(kpt1.K, kpt2.K, qpd, dshift_c)

        tblocks = kptpair.tblocks
        n_mytG = qpd.empty(tblocks.blocksize)

        # Calculate smooth part of the pair densities:
        with self.context.timer('Calculate smooth part'):
            ut1cc_mytR = ut1_hR[kpt1.h_myt].conj()
            n_mytR = ut1cc_mytR * ut2_hR[kpt2.h_myt]
            # Unvectorized
            for myt in range(tblocks.nlocal):
                n_mytG[myt] = qpd.fft(n_mytR[myt], 0, Q_G) * qpd.gd.dv

        # Calculate PAW corrections with numpy
        with self.context.timer('PAW corrections'):
            Q_aGii = self.get_paw_corrections(qpd).Q_aGii
            P1 = kpt1.projectors_in_transition_index(P1h)
            P2 = kpt2.projectors_in_transition_index(P2h)
            for (Q_Gii, (a1, P1_myti),
                 (a2, P2_myti)) in zip(Q_aGii, P1.items(), P2.items()):
                P1cc_myti = P1_myti[:tblocks.nlocal].conj()
                C1_Gimyt = np.einsum('Gij, ti -> Gjt', Q_Gii, P1cc_myti)
                P2_imyt = P2_myti.T[:, :tblocks.nlocal]
                n_mytG[:tblocks.nlocal] += np.sum(
                    C1_Gimyt * P2_imyt[np.newaxis, :, :], axis=1).T

        return PairDensity(tblocks, n_mytG)

    def get_fft_indices(self, K1, K2, qpd, dshift_c):
        from gpaw.response.pair import fft_indices
        return fft_indices(self.gs.kd, K1, K2, qpd, dshift_c)
