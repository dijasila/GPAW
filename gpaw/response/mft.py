# General modules
import numpy as np

# GPAW modules
import gpaw.mpi as mpi
from gpaw.response.chiks import ChiKS
from gpaw.response.kxc import PlaneWaveAdiabaticFXC
from gpaw.response.site_kernels import SiteKernels
from gpaw.xc import XC

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

        # Initialize the B^(xc) calculator
        # Once the response context object is ready, the user should be allowed
        # to supply the Bxc_calc themselves. This will expose the rshe
        # arguments to the user, which is not the case at present. XXX
        self.Bxc_calc = PlaneWaveBxc(self.chiks.calc,
                                     world=self.chiks.world,
                                     txt=self.chiks.fd,
                                     timer=self.chiks.timer)

        # Bxc field buffer
        self._Bxc_G = None

        # chiksr buffer
        self.currentq_c = None
        self._pd = None
        self._chiksr_GG = None

    def __call__(self, q_c, site_kernels, txt=None):
        """Calculate the isotropic exchange constants for a given wavevector.

        Parameters
        ----------
        q_c : nd.array
            Wave vector q in relative coordinates
        site_kernels : SiteKernels
            Site kernels instance defining the magnetic sites of the crystal
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

        return self.Bxc_calc(pd0)

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
        frequencies = [0.]
        pd, chiks_wGG = self.chiks.calculate(q_c, frequencies,
                                             spincomponent='+-',
                                             txt=txt)

        # Take the reactive part
        chiksr_GG = 1 / 2. * (chiks_wGG[0] + np.conj(chiks_wGG[0]).T)

        return pd, chiksr_GG


class PlaneWaveBxc(PlaneWaveAdiabaticFXC):
    """Calculator class for the plane wave coefficients of B^(xc)

               /
    B^(xc)_G = |dr B^(xc)(r) e^(-iG.r)
               /
                V0

    where V0 is the cell volume and

                δE_xc[n,m]   1
    B^(xc)(r) = ‾‾‾‾‾‾‾‾‾‾ = ‾ [V_xc^↑(r) - V_xc^↓(r)]
                  δm(r)      2

    in the local spin-density approximation for a collinear system."""

    def __init__(self, gs,
                 world=mpi.world, txt='-', timer=None,
                 rshelmax=-1, rshewmin=1.e-8):  # Overwrites rshewmin default
        """Construct the calculator based on functionality to compute fxc
        kernels. This is a temporary hack to leverage the PAW functionality
        of that code, but implies a significant computational overhead.

        Parameters
        ----------
        gs, world, txt, timer : see FXC
        rshelmax, rshewmin : see PlaneWaveAdiabaticFXC
        """
        PlaneWaveAdiabaticFXC.__init__(self, gs, '',
                                       world=world, txt=txt, timer=timer,
                                       rshelmax=rshelmax, rshewmin=rshewmin)

    def __call__(self, pd):
        """Calculate the plane wave components of Bxc"""
        # Use the fxc kernel functionality to compute B^(xc)_(G-G') / V0,
        # see [PRB 103, 245110 (2021)]
        Bxc_GG = self.calculate(pd)

        # Extract B^(xc)_G as the first column of the "kernel", renormalizing
        # by the cell volume
        V0 = pd.gd.volume
        Bxc_G = V0 * Bxc_GG[:, 0]

        return Bxc_G

    def _add_fxc(self, gd, n_sG, fxc_G):
        """This function defines the "fxc kernel" to Fourier transform."""
        self._add_Bxc(gd, n_sG, fxc_G)

    def _add_Bxc(self, gd, n_sG, Bxc_G):
        """Calculate Bxc in real space and add it to the array Bxc_G (here
        G denotes the real space grid points)."""
        # Allocate an array for the spin-dependent xc potential on the real
        # space grid
        v_sG = np.zeros(np.shape(n_sG))

        # Calculate the spin-dependent potential
        xc = XC('LDA')
        xc.calculate(gd, n_sG, v_sg=v_sG)

        # Add B^(xc) in real space to the output array
        Bxc_G += (v_sG[0] - v_sG[1]) / 2
