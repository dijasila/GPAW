# General modules
import numpy as np

# GPAW modules
from gpaw.response.chiks import ChiKS
from gpaw.response.localft import (LocalFTCalculator, add_LSDA_Bxc,
                                   add_magnetization)
from gpaw.response.site_kernels import SiteKernels
from gpaw.response.susceptibility import symmetrize_reciprocity

# ASE modules
from ase.units import Hartree


class IsotropicExchangeCalculator:
    r"""Calculator class for the Heisenberg exchange constants

    _           2
    J^ab(q) = - ‾‾ B^(xc†) K^(a†)(q) χ_KS^('+-)(q) K^b(q) B^(xc)
                V0

    calculated for an isotropic system in a plane wave representation using
    the magnetic force theorem within second order perturbation theory, see
    [arXiv:2204.04169].

    Entering the formula for the isotropic exchange constant at wave vector q
    between sublattice a and b is the unit cell volume V0, the functional
    derivative of the (LDA) exchange-correlation energy with respect to the
    magnitude of the magnetization B^(xc), the sublattice site kernels K^a(q)
    and K^b(q) as well as the reactive part of the static transverse magnetic
    susceptibility of the Kohn-Sham system χ_KS^('+-)(q).

    The site kernels encode the partitioning of real space into sites of the
    Heisenberg model. This is not a uniquely defined procedure, why the user
    has to define them externally through the SiteKernels interface."""

    def __init__(self, chiks, localft_calc):
        """Construct the IsotropicExchangeCalculator object

        Parameters
        ----------
        chiks : ChiKS
            ChiKS calculator object
        """
        assert isinstance(chiks, ChiKS)
        # Check that chiks has the assumed properties
        assumed_props = dict(
            gammacentered=True,
            kpointintegration='point integration',
            nblocks=1
        )
        for key, item in assumed_props.items():
            assert getattr(chiks, key) == item,\
                f'Expected chiks.{key} == {item}. Got: {getattr(chiks, key)}'

        self.chiks = chiks
        self.context = chiks.context

        # Check assumed properties of the LocalFTCalculator
        assert isinstance(localft_calc, LocalFTCalculator)
        assert localft_calc.context is self.context
        assert localft_calc.gs is chiks.gs
        self.localft_calc = localft_calc

        # Bxc field buffer
        self._Bxc_G = None

        # chiksr buffer
        self.currentq_c = None
        self._pd = None
        self._chiksr_GG = None
        self._chiksr_corr_GG = None

    def __call__(self, q_c, site_kernels, goldstone_corr=False, txt=None):
        """Calculate the isotropic exchange constants for a given wavevector.

        Parameters
        ----------
        q_c : nd.array
            Wave vector q in relative coordinates
        site_kernels : SiteKernels
            Site kernels instance defining the magnetic sites of the crystal
        goldstone_corr : bool
            Include a minimal Goldstone correction to χ_KS^('+-)(q).
        txt : str
            Separate file to store the chiks calculation output in (optional).
            If not supplied, the output will be written to the standard text
            output location specified when initializing chiks.

        Returns
        -------
        J_abp : nd.array (dtype=complex)
            Isotropic Heisenberg exchange constants between magnetic sites a
            and b for all the site partitions p given by the site_kernels.
        """
        assert isinstance(site_kernels, SiteKernels)

        # Get ingredients
        Bxc_G = self.get_Bxc()
        pd, chiksr_GG = self.get_chiksr(q_c, txt=txt)
        if goldstone_corr:
            chiksr_GG = chiksr_GG + self.get_goldstone_correction()
        V0 = pd.gd.volume

        # Allocate an array for the exchange constants
        nsites = site_kernels.nsites
        J_pab = np.empty(site_kernels.shape + (nsites,), dtype=complex)

        # Compute exchange coupling
        for J_ab, K_aGG in zip(J_pab, site_kernels.calculate(pd)):
            for a in range(nsites):
                for b in range(nsites):
                    J = np.conj(Bxc_G) @ np.conj(K_aGG[a]).T @ chiksr_GG \
                        @ K_aGG[b] @ Bxc_G
                    J_ab[a, b] = - 2. * J / V0

        # Transpose to have the partitions index last
        J_abp = np.transpose(J_pab, (1, 2, 0))

        return J_abp * Hartree  # Convert from Hartree to eV

    def get_Bxc(self):
        """Get B^(xc)_G from buffer."""
        if self._Bxc_G is None:  # Calculate, if buffer is empty
            self._Bxc_G = self._calculate_Bxc()

        return self._Bxc_G

    def _calculate_Bxc(self):
        """Use the PlaneWaveBxc calculator to calculate the plane wave
        coefficients B^xc_G"""
        # Create a plane wave descriptor encoding the plane wave basis. Input
        # q_c is arbitrary, since we are assuming that chiks.gammacentered == 1
        pd0 = self.chiks.get_PWDescriptor([0., 0., 0.])

        return self.localft_calc(pd0, add_LSDA_Bxc)

    def get_chiksr(self, q_c, txt=None):
        """Get χ_KS^('+-)(q) from buffer."""
        q_c = np.asarray(q_c)
        if self.currentq_c is None or not np.allclose(q_c, self.currentq_c):
            # Calculate chiks for any new q-point or if buffer is empty
            self.currentq_c = q_c
            self._pd, self._chiksr_GG = self._calculate_chiksr(q_c, txt=txt)

        return self._pd, self._chiksr_GG

    def _calculate_chiksr(self, q_c, txt=None):
        r"""Use the ChiKS calculator to calculate the reactive part of the
        static Kohn-Sham susceptibility χ_KS^('+-)(q).

        First, the dynamic Kohn-Sham susceptibility

                                 __  __
                              1  \   \        f_nk↑ - f_mk+q↓
        χ_KS,GG'^+-(q,ω+iη) = ‾  /   /  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                              V  ‾‾  ‾‾ ħω - (ε_mk+q↓ - ε_nk↑) + iħη
                                 k  n,m
                                        x n_nk↑,mk+q↓(G+q) n_mk+q↓,nk↑(-G'-q)

        is calculated in the static limit ω=0. Then, the reactive part (see
        [PRB 103, 245110 (2021)]) is extracted,

                              1
        χ_KS,GG'^(+-')(q,ω) = ‾ [χ_KS,GG'^+-(q,ω+iη) + χ_KS,-G'-G^-+(-q,-ω+iη)]
                              2

                              1
                            = ‾ [χ_KS,GG'^+-(q,ω+iη) + χ_KS,G'G^(+-*)(q,ω+iη)]
                              2

        where it was used that n^+(r) and n^-(r) are each others Hermitian
        conjugates to reach the last equality.
        """
        # Initiate new output file, if supplied
        if txt is not None:
            self.context.new_txt_and_timer(txt)

        frequencies = [0.]
        pd, chiks_wGG = self.chiks.calculate(q_c, frequencies,
                                             spincomponent='+-')
        symmetrize_reciprocity(pd, chiks_wGG)

        # Take the reactive part
        chiksr_GG = 1 / 2. * (chiks_wGG[0] + np.conj(chiks_wGG[0]).T)

        return pd, chiksr_GG

    def get_goldstone_correction(self):
        """Get δχ_KS^('+-)_GG' from buffer."""
        if self._chiksr_corr_GG is None:  # Calculate, if buffer is empty
            self._chiksr_corr_GG = self._calculate_goldstone_correction()

        return self._chiksr_corr_GG

    def _calculate_goldstone_correction(self):
        r"""In a complete representation of the Kohn-Sham susceptibility, the
        rotational invariance of the spin axis in absence of spin-orbit
        coupling implies that

        |m> = 2 χ_KS^('+-)(q=0) |B^(xc)>,

        written in the plane-wave basis. However, using a finite basis, this
        identity will be slightly broken leading to a Goldstone inconsistency.

        To correct for this inconsistency, we may choose to add a minimal
        correction [paper in preparation],

        2 <B^(xc)|B^(xc)> δχ_KS^('+-) = |δm><B^(xc)| + |B^(xc)><δm|

                                                     <δm|B^(xc)>
                                        - |B^(xc)> ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ <B^(xc)|
                                                   <B^(xc)|B^(xc)>

        where

        |δm> = |m> - |m^χ>

        with:

        |m^χ> = 2 χ_KS^('+-)(q=0) |B^(xc)>.
        """
        pd0, chiksr0_GG = self.get_chiksr(np.array([0., 0., 0.]))
        m_G = self.localft_calc(pd0, add_magnetization)
        Bxc_G = self.get_Bxc()

        mchi_G = 2. * chiksr0_GG @ Bxc_G
        dm_G = m_G - mchi_G

        chiksr_corr_GG = np.outer(dm_G, np.conj(Bxc_G))\
            + np.outer(Bxc_G, np.conj(dm_G))\
            - np.outer(Bxc_G, np.conj(Bxc_G))\
            * np.dot(np.conj(dm_G), Bxc_G) / np.dot(np.conj(Bxc_G), Bxc_G)
        chiksr_corr_GG /= 2. * np.dot(np.conj(Bxc_G), Bxc_G)

        return chiksr_corr_GG
