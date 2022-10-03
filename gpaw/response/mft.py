# General modules
from abc import ABC, abstractmethod, abstractproperty

import numpy as np

# GPAW modules
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.chiks import ChiKS
from gpaw.response.localft import (LocalFTCalculator,
                                   add_LSDA_Bxc, add_magnetization)
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

    def __init__(self, chiks):
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

        # chiksr buffer
        self.currentq_c = None
        self._pd = None
        self._chiksr_GG = None

    def __call__(self, q_c, site_kernels, bxc_calc, txt=None):
        """Calculate the isotropic exchange constants for a given wavevector.

        Parameters
        ----------
        q_c : nd.array
            Wave vector q in relative coordinates
        site_kernels : SiteKernels
            Site kernels instance defining the magnetic sites of the crystal
        bxc_calc : BxcCalculator
            Calculator, which calculates the plane-wave components of B^(xc)
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
        assert isinstance(bxc_calc, BxcCalculator)
        assert bxc_calc.context is self.context

        # Initiate new output file, if supplied
        if txt is not None:
            self.context.new_txt_and_timer(txt)

        # Get ingredients
        pd, chiksr_GG = self.get_chiksr(q_c)
        Bxc_G = self.get_Bxc(bxc_calc)
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

    def get_Bxc(self, bxc_calc):
        """Retrieve the B^(xc) plane-wave components from the BxcCalculator."""

        if bxc_calc.in_buffer():  # Check buffer for preexisting calculation
            return bxc_calc.from_buffer()

        # Perform actual calculation
        bxc_args = self.prepare_bxc_args(bxc_calc)
        Bxc_G = bxc_calc.calculate(*bxc_args)

        return Bxc_G

    def prepare_bxc_args(self, bxc_calc):
        """Prepare the necessary arguments for the BxcCalculator."""
        bxc_args = tuple([self.get_bxc_arg(arg) for arg in bxc_calc.args])

        return bxc_args

    def get_bxc_arg(self, arg):
        """Factory function to retrieve different BxcCalculator arguments."""
        if arg == 'pd0':
            # Plane-wave descriptor for q=0
            return self.chiks.get_PWDescriptor([0., 0., 0.])
        elif arg == 'chiksr0_GG':
            return self.get_chiksr(np.array([0., 0., 0.]))
        else:
            raise NotImplementedError(f'The BxcCalculator argument {arg} has '
                                      'not yet been implemented')

    def get_chiksr(self, q_c):
        """Get χ_KS^('+-)(q) from buffer."""
        q_c = np.asarray(q_c)
        if self.currentq_c is None or not np.allclose(q_c, self.currentq_c):
            # Calculate chiks for any new q-point or if buffer is empty
            self.currentq_c = q_c
            self._pd, self._chiksr_GG = self._calculate_chiksr(q_c)

        return self._pd, self._chiksr_GG

    def _calculate_chiksr(self, q_c):
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
        frequencies = [0.]
        pd, chiks_wGG = self.chiks.calculate(q_c, frequencies,
                                             spincomponent='+-')
        symmetrize_reciprocity(pd, chiks_wGG)

        # Take the reactive part
        chiksr_GG = 1 / 2. * (chiks_wGG[0] + np.conj(chiks_wGG[0]).T)

        return pd, chiksr_GG


class BxcCalculator(ABC):

    """Abstract calculator base class for calculations of the plane-wave
    components of B^(xc). Keeps calculated components in a buffer and knows
    what arguments to give its self.calculate(*args) method."""

    def __init__(self, gs, context):
        """Construct the BxcCalculator with an empty buffer."""
        assert isinstance(gs, ResponseGroundStateAdapter)
        self.gs = gs
        assert isinstance(context, ResponseContext)
        self.context = context

        # Bxc field buffer
        self._Bxc_G = None

    def in_buffer(self):
        return self._Bxc_G is not None

    def from_buffer(self):
        assert self.in_buffer()
        return self._Bxc_G

    def calculate(self, *args):
        """Calculate the Bxc plane-wave components and fill buffer."""
        self._Bxc_G = self._calculate(*args)
        return self._Bxc_G

    @abstractmethod
    def _calculate(self, *args):
        pass

    @abstractproperty
    def args(self):
        # Return a list of the arguments (as strings) to self._calculate()
        pass


class LSDABxcCalculator(BxcCalculator):
    """Calculator for the LSDA magnetic xc potential. The calculation is based
    on a local Fourier transform of B^(xc) evaluated explicitly in real space.
    """

    def __init__(self, gs, context,
                 rshelmax=-1, rshewmin=None):
        """Construct the BxcCalculator with an internal LocalFTCalculator

        Parameters
        ----------
        rshelmax : int or None
            See LocalFTCalculator
        rshewmin : float or None
            See LocalFTCalculator
        """
        BxcCalculator.__init__(self, gs, context)

        # Construct a LocalFTCalculator to carry out the FT
        self.localft_calc = LocalFTCalculator.from_rshe_parameters(
            gs, context, rshelmax=rshelmax, rshewmin=rshewmin)

    @property
    def args(self):
        return ['pd0']

    def _calculate(self, pd0):
        return self.localft_calc(pd0, add_LSDA_Bxc)


class GoldstoneBxcCalculator(LSDABxcCalculator):
    """Calculator for magnetic xc potentials computed based on the invariance
    of the Kohn-Sham system under a rigid rotation of the spin axis. To comply
    with this invariance and to satisfy the Goldstone theorem,

    m = χ_KS^('+-)(q) B^(xc)

    in the plane-wave basis. This calculator inverts this expression to obtain
    the plane wave components of B^(xc)."""

    @property
    def args(self):
        return ['pd0', 'chiksr0_GG']

    def _calculate(self, pd0, chiksr0_GG):
        m_G = self.localft_calc(pd0, add_magnetization)
        Bxc_G = np.linalg.inv(chiksr0_GG) @ m_G

        return Bxc_G
