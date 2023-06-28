from abc import ABC, abstractmethod

import numpy as np

from gpaw.utilities.blas import gemmdot

from gpaw.sphere.integrate import spherical_truncation_function_collection

from gpaw.response import timer
from gpaw.response.kspair import KohnShamKPointPair
from gpaw.response.pair import phase_shifted_fft_indices
from gpaw.response.site_paw import calculate_site_pair_density_correction


class MatrixElement(ABC):
    """Data class for transitions distributed Kohn-Sham matrix elements."""

    def __init__(self, tblocks, qpd):
        self.tblocks = tblocks
        self.qpd = qpd

        self.array = self.zeros()
        assert self.array.shape[0] == tblocks.blocksize

    @abstractmethod
    def zeros(self):
        """Generate matrix element array with zeros."""

    @property
    def local_array_view(self):
        return self.array[:self.tblocks.nlocal]

    def get_global_array(self):
        """Get the global (all gathered) matrix element."""
        return self.tblocks.all_gather(self.array)


class MatrixElementCalculator(ABC):
    r"""Abstract base class for matrix element calculators.

    In the PAW method, Kohn-Sham matrix elements,
                            ˰
    A_(nks,n'k's') = <ψ_nks|A|ψ_n'k's'>

    can be evaluated in the space of pseudo waves using the pseudo operator
            __  __
    ˷   ˰   \   \   ˷           ˰           ˷    ˰ ˷       ˷
    A = A + /   /  |p_ai>[<φ_ai|A|φ_ai'> - <φ_ai|A|φ_ai'>]<p_ai'|
            ‾‾  ‾‾
            a   i,i'

    to which effect,
                      ˷     ˷ ˷
    A_(nks,n'k's') = <ψ_nks|A|ψ_n'k's'>

    This is an abstract base class for calculating such matrix elements for a
    number of band and spin transitions t=(n,s)->(n',s') for a given k-point
    pair k and k + q:

    A_kt = A_(nks,n'k+qs')
    """

    def add_pseudo_contribution(self, kptpair, *args):
        """Add the pseudo matrix element to an output array.

        The pseudo matrix element is evaluated on the coarse real-space grid
        and integrated:

        ˷       ˷     ˰ ˷
        A_kt = <ψ_nks|A|ψ_n'k+qs'>

               /    ˷          ˰ ˷
             = | dr ψ_nks^*(r) A ψ_n'k+qs'(r)
               /
        """
        ikpt1 = kptpair.ikpt1
        ikpt2 = kptpair.ikpt2

        # Map the k-points from the irreducible part of the BZ to the BZ
        # k-point K (up to a reciprocal lattice vector)
        k1_c = self.gs.ibz2bz[kptpair.K1].map_kpoint()
        k2_c = self.gs.ibz2bz[kptpair.K2].map_kpoint()

        # Fourier transform the periodic part of the pseudo waves to the coarse
        # real-space grid and map them to the BZ k-point K (up to the same
        # reciprocal lattice vector as above)
        ut1_hR = self.get_periodic_pseudo_waves(kptpair.K1, ikpt1)
        ut2_hR = self.get_periodic_pseudo_waves(kptpair.K2, ikpt2)

        # Fold out the pseudo waves to the transition index
        ut1_mytR = ut1_hR[ikpt1.h_myt]
        ut2_mytR = ut2_hR[ikpt2.h_myt]

        self._add_pseudo_contribution(k1_c, k2_c, ut1_mytR, ut2_mytR, *args)

    def add_paw_correction(self, kptpair, *args):
        r"""Add the matrix element PAW correction to an output array.

        The PAW correction is calculated using the projector overlaps of the
        pseudo waves:
                __  __
                \   \   ˷     ˷              ˷     ˷
        ΔA_kt = /   /  <ψ_nks|p_ai> ΔA_aii' <p_ai'|ψ_n'k+qs'>
                ‾‾  ‾‾
                a   i,i'

        where the PAW correction tensor is calculated on a radial grid inside
        each augmentation sphere of position R_a, using the atom-centered
        partial waves φ_ai(r):
                        ˰           ˷    ˰ ˷
        ΔA_aii' = <φ_ai|A|φ_ai'> - <φ_ai|A|φ_ai'>

                  /                   ˰
                = | dr [φ_ai^*(r-R_a) A φ_ai'(r-R_a)
                  /       ˷             ˰ ˷
                        - φ_ai^*(r-R_a) A φ_ai'(r-R_a)]
        """
        ikpt1 = kptpair.ikpt1
        ikpt2 = kptpair.ikpt2

        # Map the projections from the irreducible part of the BZ to the BZ
        # k-point K
        P1h = self.gs.ibz2bz[kptpair.K1].map_projections(ikpt1.Ph)
        P2h = self.gs.ibz2bz[kptpair.K2].map_projections(ikpt2.Ph)

        # Fold out the projectors to the transition index
        P1_amyti = ikpt1.projectors_in_transition_index(P1h)
        P2_amyti = ikpt2.projectors_in_transition_index(P2h)
        assert P1_amyti.atom_partition.comm.size ==\
            P2_amyti.atom_partition.comm.size == 1,\
            'We need access to the projections of all atoms'

        self._add_paw_correction(P1_amyti, P2_amyti, *args)

    @abstractmethod
    def _add_pseudo_contribution(self, k1_c, k2_c, ut1_mytR, ut2_mytR, *args):
        """Add pseudo contribution based on the pseudo waves in real space."""

    @abstractmethod
    def _add_paw_correction(self, P1_amyti, P2_amyti, *args):
        """Add paw correction based on the projector overlaps."""

    def get_periodic_pseudo_waves(self, K, ikpt):
        """FFT the Kohn-Sham orbitals to real space and map them from the
        irreducible k-point to the k-point in question."""
        ut_hR = self.gs.gd.empty(ikpt.nh, self.gs.dtype)
        for h, psit_G in enumerate(ikpt.psit_hG):
            ut_hR[h] = self.gs.ibz2bz[K].map_pseudo_wave(
                self.gs.global_pd.ifft(psit_G, ikpt.ik))

        return ut_hR


class PairDensity(MatrixElement):

    def zeros(self):
        return self.qpd.zeros(self.tblocks.blocksize)


class NewPairDensityCalculator(MatrixElementCalculator):
    r"""Class for calculating pair densities

    n_kt(G+q) = n_nks,n'k+qs'(G+q) = <nks| e^-i(G+q)r |n'k+qs'>

                /
              = | dr e^-i(G+q)r ψ_nks^*(r) ψ_n'k+qs'(r)
                /

    for a single k-point pair (k, k + q) in the plane-wave mode."""
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
            with self.context.timer('Initialize PAW corrections'):
                self._pawcorr = self.gs.pair_density_paw_corrections(qpd)
                self._currentq_c = qpd.q_c

        return self._pawcorr

    @timer('Calculate pair density')
    def __call__(self, kptpair: KohnShamKPointPair, qpd) -> PairDensity:
        r"""Calculate the pair density for all transitions t.

        In the PAW method, the all-electron pair density is calculated in
        two contributions, the pseudo pair density and a PAW correction,

        n_kt(G+q) = ñ_kt(G+q) + Δn_kt(G+q),

        see [PRB 103, 245110 (2021)] for details.
        """
        # Initialize a blank pair density object
        pair_density = PairDensity(kptpair.tblocks, qpd)
        n_mytG = pair_density.local_array_view

        self.add_pseudo_contribution(kptpair, qpd, n_mytG)
        self.add_paw_correction(kptpair, qpd, n_mytG)

        return pair_density

    @timer('Calculate the pseudo pair density')
    def _add_pseudo_contribution(self, k1_c, k2_c, ut1_mytR, ut2_mytR,
                                 qpd, n_mytG):
        r"""Add the pseudo pair density to an output array.

        The pseudo pair density is first evaluated on the coarse real-space
        grid and then FFT'ed to reciprocal space,

                    /               ˷          ˷
        ñ_kt(G+q) = | dr e^-i(G+q)r ψ_nks^*(r) ψ_n'k+qs'(r)
                    /V0
                                 ˷          ˷
                  = FFT_G[e^-iqr ψ_nks^*(r) ψ_n'k+qs'(r)]

        where the Kohn-Sham orbitals are normalized to the unit cell.
        """
        # Calculate the pseudo pair density in real space, up to a phase of
        # e^(-i[k+q-k']r).
        # This phase does not necessarily vanish, since k2_c only is required
        # to equal k1_c + qpd.q_c modulo a reciprocal lattice vector.
        nt_mytR = ut1_mytR.conj() * ut2_mytR

        # Get the FFT indices corresponding to the Fourier transform
        #                       ˷          ˷
        # FFT_G[e^(-i[k+q-k']r) u_nks^*(r) u_n'k's'(r)]
        Q_G = phase_shifted_fft_indices(k1_c, k2_c, qpd)

        # Add the desired plane-wave components of the FFT'ed pseudo pair
        # density to the output array
        for n_G, n_R in zip(n_mytG, nt_mytR):
            n_G[:] += qpd.fft(n_R, 0, Q_G) * qpd.gd.dv

    @timer('Calculate the pair density PAW corrections')
    def _add_paw_correction(self, P1_amyti, P2_amyti, qpd, n_mytG):
        r"""Add the pair-density PAW correction to the output array.

        The correction is calculated from
                     __  __
                     \   \   ˷     ˷     ˷    ˷
        Δn_kt(G+q) = /   /  <ψ_nks|p_ai><p_ai'|ψ_n'k+qs'> Q_aii'(G+q)
                     ‾‾  ‾‾
                     a   i,i'

        where the pair-density PAW correction tensor is given by:

                      /
        Q_aii'(G+q) = | dr e^-i(G+q)r [φ_ai^*(r-R_a) φ_ai'(r-R_a)
                      /                  ˷             ˷
                                       - φ_ai^*(r-R_a) φ_ai'(r-R_a)]
        """
        Q_aGii = self.get_paw_corrections(qpd).Q_aGii
        for a, Q_Gii in enumerate(Q_aGii):
            # Make outer product of the projector overlaps
            P1ccP2_mytii = P1_amyti[a].conj()[..., np.newaxis] \
                * P2_amyti[a][:, np.newaxis]
            # Sum over partial wave indices and add correction to the output
            n_mytG[:] += np.einsum('tij, Gij -> tG', P1ccP2_mytii, Q_Gii)


class SitePairDensityCalculator(MatrixElementCalculator):
    r"""Class for calculating site pair densities.

    The site pair density is defined via smooth truncation functions Θ(r∊Ω_ap)
    for every site a and site partitioning p, interpolating smoothly between
    unity for positions inside the spherical site volume and zero outside it:

    n^ap_kt = n^ap_(nks,n'k+qs') = <ψ_nks|Θ(r∊Ω_ap)|ψ_n'k+qs'>

             /
           = | dr Θ(r∊Ω_ap) ψ_nks^*(r) ψ_n'k+qs'(r)
             /

    For details, see [publication in preparation].
    """

    def __init__(self, gs, context, atomic_site_data):
        """Construct the SitePairDensityCalculator."""
        self.gs = gs
        self.context = context
        self.atomic_site_data = atomic_site_data

        # Set up spherical truncation function collection on the coarse
        # real-space grid
        adata = atomic_site_data
        self.stfc = spherical_truncation_function_collection(
            gs.gd, adata.spos_ac, adata.rc_ap, adata.drcut, adata.lambd_ap,
            dtype=complex)

        # PAW correction tensor
        self._N_apii = None

    def get_paw_correction_tensor(self):
        if self._N_apii is None:
            self._N_apii = self.calculate_paw_correction_tensor()
        return self._N_apii

    def calculate_paw_correction_tensor(self):
        """Calculate the site pair density correction tensor N_ii'^ap."""
        N_apii = []
        adata = self.atomic_site_data
        for A, rc_p, lambd_p in zip(adata.A_a, adata.rc_ap, adata.lambd_ap):
            pawdata = self.gs.pawdatasets[A]
            N_apii.append(calculate_site_pair_density_correction(
                pawdata, rc_p, adata.drcut, lambd_p))
        return N_apii

    @timer('Calculate site pair density')
    def __call__(self, kptpair):
        """Calculate the site pair density for all transitions t.

        The calculation is split in a pseudo site pair density contribution and
        a PAW correction:

        n^ap_kt = ñ^ap_kt + Δn^ap_kt
        """
        # Initialize site pair density
        n_mytap = np.zeros((kptpair.tblocks.blocksize,)
                           + self.atomic_site_data.shape, dtype=complex)
        self.add_pseudo_contribution(kptpair, n_mytap)
        self.add_paw_correction(kptpair, n_mytap)
        return n_mytap

    @timer('Calculate pseudo site pair density')
    def _add_pseudo_contribution(self, k1_c, k2_c, ut1_mytR, ut2_mytR,
                                 n_mytap):
        """Add the pseudo site pair density to the output array.

        The pseudo pair density is evaluated on the coarse real-space grid and
        integrated together with the smooth truncation function,

                  /              ˷          ˷
        ñ^ap_kt = | dr Θ(r∊Ω_ap) ψ_nks^*(r) ψ_n'k+qs'(r)
                  /

        where the Kohn-Sham orbitals are normalized to the unit cell.
        """
        # Construct pseudo waves with Bloch phases
        r_cR = self.gs.ibz2bz.r_cR  # scaled grid coordinates
        psit1_mytR = np.exp(2j * np.pi * gemmdot(k1_c, r_cR))[np.newaxis]\
            * ut1_mytR
        psit2_mytR = np.exp(2j * np.pi * gemmdot(k2_c, r_cR))[np.newaxis]\
            * ut2_mytR
        # Calculate real-space pair densities ñ_kt(r)
        nt_mytR = psit1_mytR.conj() * psit2_mytR

        # Integrate Θ(r∊Ω_ap) ñ_kt(r)
        ntlocal = nt_mytR.shape[0]
        adata = self.atomic_site_data
        nt_amytp = {a: np.empty((ntlocal, adata.npartitions), dtype=complex)
                    for a in range(adata.nsites)}
        self.stfc.integrate(nt_mytR, nt_amytp, q=0)

        # Add integral to output array
        for a in range(adata.nsites):
            n_mytap[:, a] += nt_amytp[a]

    @timer('Calculate site pair density PAW correction')
    def _add_paw_correction(self, P1_Amyti, P2_Amyti, n_mytap):
        r"""Add the site pair density PAW correction to the output array.

        For every site a, we only need a PAW correction for that site itself,
                   __
                   \   ˷     ˷              ˷     ˷
        Δn^ap_kt = /  <ψ_nks|p_ai> N_apii' <p_ai'|ψ_n'k+qs'>
                   ‾‾
                   i,i'

        where N_apii' is the site pair density correction tensor.
        """
        N_apii = self.get_paw_correction_tensor()
        for a, (A, N_pii) in enumerate(zip(
                self.atomic_site_data.A_a, N_apii)):
            # Make outer product of the projector overlaps
            P1ccP2_mytii = P1_Amyti[A].conj()[..., np.newaxis] \
                * P2_Amyti[A][:, np.newaxis]
            # Sum over partial wave indices and add correction to the output
            n_mytap[:, a] += np.einsum('tij, pij -> tp', P1ccP2_mytii, N_pii)
