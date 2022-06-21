# General modules
import numpy as np
import sys

# GPAW modules
import gpaw.mpi as mpi
from gpaw.response.chiks import ChiKS
from gpaw.response.kxc import PlaneWaveAdiabaticFXC
from gpaw.response.site_kernels import SiteKernels
from gpaw.xc import XC

# ASE modules
from ase.units import Hartree


class IsotropicExchangeCalculator:
    """Calculator class for the Heisenberg exchange constants

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
            disable_point_group=True,
            disable_time_reversal=True,
            disable_non_symmorphic=True,
            kpointintegration='point integration',
            nblocks=1
        )
        for key, item in assumed_props.items():
            assert getattr(chiks, key) == item,\
                f'Expected chiks.{key} == {item}. Got: {getattr(chiks, key)}'

        self.chiks = chiks

        # Calculator for xc-kernel
        self.Bxc_calc = AdiabaticBXC(self.chiks.calc, world=self.chiks.world)

        # Bxc field buffer
        self._Bxc_G = None

        # chiks buffer
        self.currentq_c = None
        self._chiks_GG = None

    def __call__(self, q_c, site_kernels, txt=sys.stdout):
        """Calculate the isotropic exchange constants for a given wavevector.

        Parameters
        ----------
        q_c : nd.array
            Components of wavevector in relative coordinates
        site_kernels : SiteKernels
            Site kernels instance defining the magnetic sites of the crystal
        txt : str
            Where to send text output from chiks calculation

        Returns
        -------
        J_abr : nd.array (dtype=complex)
            Exchange between magnetic sites (a, b) for different
            parameters of the integration regions (r).
        """
        assert isinstance(site_kernels, SiteKernels)

        # Get ingredients
        Bxc_G = self.get_Bxc()
        chiks_GG = self.get_chiks(q_c, txt=txt)

        # Get plane-wave descriptor
        pd = self.chiks.get_PWDescriptor(q_c)  # Move to get_chiks! XXX
        V0 = pd.gd.volume

        # Allocate an array for the exchange constants
        nsites = site_kernels.nsites
        J_pab = np.empty(site_kernels.shape + (nsites,), dtype=complex)

        # Compute exchange coupling
        for J_ab, K_aGG in zip(J_pab, site_kernels.calculate(pd)):
            for a in range(nsites):
                for b in range(nsites):
                    Ka_GG = K_aGG[a, :, :]
                    Kb_GG = K_aGG[b, :, :]
                    J = np.conj(Bxc_G) @ np.conj(Ka_GG).T @ chiks_GG @ Kb_GG \
                        @ Bxc_G
                    J_ab[a, b] = 2. * J / V0

        # Transpose to have the partitions index last
        J_abp = np.transpose(J_pab, (1, 2, 0))

        return J_abp * Hartree  # Convert from Hartree to eV

    def get_Bxc(self):
        if self._Bxc_G is None:
            self._Bxc_G = self._calculate_Bxc()

        return self._Bxc_G

    def _calculate_Bxc(self):
        # Compute xc magnetic field
        # Note : Bxc is calculated from the xc-kernel, which is a 2-point
        # function, while B_xc is 1-point Because of how the different
        # Fourier transforms are defined, this gives an extra volume factor
        # See eq. 50 of Phys. Rev. B 103, 245110 (2021)
        print('Calculating Bxc')
        # Plane-wave descriptor (input is arbitrary)
        pd0 = self.chiks.get_PWDescriptor([0, 0, 0])
        V0 = pd0.gd.volume
        Bxc_GG = self.Bxc_calc(pd0)
        Bxc_G = V0 * Bxc_GG[:, 0]
        print('Done calculating Bxc')

        return Bxc_G

    def get_chiks(self, q_c, txt=None):
        q_c = np.asarray(q_c)
        if self.currentq_c is None or not np.allclose(q_c, self.currentq_c):
            # Calculate chiks for a new q-point and write it to the buffer
            self.currentq_c = q_c
            self._chiks_GG = self._calculate_chiks(q_c, txt=txt)

        return self._chiks_GG

    def _calculate_chiks(self, q_c, txt=None):
        """Calculate the reactive part of the static Kohn-Sham susceptibility.
        """
        frequencies = [0.]

        # Calculate the dynamic KS susceptibility in the static limit
        _, chiks_wGG = self.chiks.calculate(q_c, frequencies,
                                            spincomponent='-+',
                                            txt=txt)

        # Remove frequency axis
        # Where do we take the reactive part??? !!! XXX
        # Shouldn't this be more than a minus sign??? !!! XXX
        chiks_GG = - chiks_wGG[0, :, :]

        return chiks_GG


class AdiabaticBXC(PlaneWaveAdiabaticFXC):
    """Exchange-correlation magnetic field under the adiabatic assumption
    in the plane wave mode

    Note : Temporary hack. Refactor later. Computes full Bxc_GG-matrix,
    where only diagonal, Bxc_G, is needed.
    """

    def __init__(self, gs,
                 world=mpi.world, txt='-', timer=None,
                 rshelmax=-1, rshewmin=1.e-8, filename=None):
        """
        gs, world, txt, timer : see PlaneWaveAdiabaticFXC, FXC
        rshelmax, rshewmin, filename : see PlaneWaveAdiabaticFXC
        """

        PlaneWaveAdiabaticFXC.__init__(self, gs, '',
                                       world=world, txt=txt, timer=timer,
                                       rshelmax=rshelmax, rshewmin=rshewmin,
                                       filename=filename)

    def _add_fxc(self, gd, n_sG, fxc_G):
        """Calculate fxc in real-space grid"""

        fxc_G += self._calculate_fxc(gd, n_sG)

    def _calculate_fxc(self, gd, n_sG):
        """Calculate polarized fxc of spincomponents '+-', '-+'."""
        v_sG = np.zeros(np.shape(n_sG))     # Potential
        xc = XC('LDA')
        xc.calculate(gd, n_sG, v_sg=v_sG)

        return (v_sG[0] - v_sG[1]) / 2    # Definition of Bxc
