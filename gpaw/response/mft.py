# General modules
import numpy as np
import sys

# GPAW modules
import gpaw.mpi as mpi
from gpaw.response.susceptibility import FourComponentSusceptibilityTensor
from gpaw.response.kxc import PlaneWaveAdiabaticFXC
from gpaw.response.site_kernels import SiteKernels
from gpaw.xc import XC

# ASE modules
from ase.units import Hartree


class IsotropicExchangeCalculator:
    """Class calculating the isotropic Heisenberg exchange, J.

    J describes the exchange interactions between magnetic moments on a
    discrete lattice.

    J can be related to the static, transverse magnetic susceptibility of
    the Kohn-Sham system. This can be computed ab-initio by way of
    linear response theory.

    The magnetic moments are defined as integrals of the
    magnetisation density centered on the lattice sites. Both the shape
    and size of these integration regions can be varied.
    All the information about positions and integration regions for
    the magnetic sites are encoded in the wavevector dependent
    site-kernels, K_m(q).

    The central formula for computing J is
    J(q) = sum_{G1,G2,G3,G4} Bxc_G1 Kn_G1G2(q) chiks_G2G3^{-+}(q)
                             X Km_G3G4^* Bxc_G4^*

    Note that the response function, chiks, is computed with opposite sign
    relative to the implementation papers. If this is fixed, then the formula
    for J should also get a minus sign.

    """

    def __init__(self, gs, ecut=100, nbands=None, world=mpi.world):
        """Construct the IsotropicExchangeCalculator object

        Parameters
        ----------
        gs : str or gpaw calculator
            Calculator with converged ground state as input to the linear
            response calculation.
        ecut : number
            Cutoff energy in eV
            In response calculation, include all G-vectors with G^2/2 < ecut
        nbands : int
            Maximum band index to include in response calculation.
        world : obj
            MPI communicator.
        """

        # Calculator for response function
        self.chiksf = StaticChiKSFactory(gs,
                                         ecut=ecut,
                                         nblocks=1,
                                         eta=0,
                                         nbands=nbands,
                                         world=world)

        # Calculator for xc-kernel
        self.Bxc_calc = AdiabaticBXC(self.chiksf.calc, world=world)

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
        pd = self.chiksf.get_PWDescriptor(q_c)
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
        pd0 = self.chiksf.get_PWDescriptor([0, 0, 0])
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
            _, chiks_GG = self.chiksf('-+', q_c, txt=txt)
            self._chiks_GG = chiks_GG

        return self._chiks_GG


class StaticChiKSFactory(FourComponentSusceptibilityTensor):
    """Class calculating components of the static Kohn-Sham
    susceptibility tensor.

    Note : Temporary hack. Refactor later.
    Calls FourComponentSusceptibilityTensor in an akward way.
    """

    def __init__(self, gs, eta=0.0, ecut=50, nbands=None,
                 world=mpi.world, nblocks=1, txt=sys.stdout):

        """
        Currently, everything is in plane wave mode.
        If additional modes are implemented, maybe look to fxc to see how
        multiple modes can be supported.

        Parameters
        ----------
        gs : see gpaw.response.chiks, gpaw.response.kslrf
        eta, ecut, nbands, world, nblocks, txt : see gpaw.response.chiks,
            gpaw.response.kslrf
        """

        # Remove user access
        fixed_kwargs = {'gammacentered': True,
                        'disable_point_group': True,
                        'disable_time_reversal': True,
                        'bundle_integrals': True}

        FourComponentSusceptibilityTensor.__init__(self, gs, eta=eta,
                                                   ecut=ecut, nbands=nbands,
                                                   nblocks=nblocks,
                                                   world=world,
                                                   txt=txt, **fixed_kwargs)

    def __call__(self, spincomponent, q_c, txt=None):
        """Calculate a given component of chiKS.
        Substitutes calculate_component_array and returns zero frequency."""

        # Only compute static susceptibility
        frequencies = [0]

        # Perform calculation
        ecut = self.ecut * Hartree  # eV -> Hartree
        (_, G_Gc,
         chiks_wGG, _) = self.calculate_component_array(spincomponent, q_c,
                                                        frequencies,
                                                        array_ecut=ecut,
                                                        txt=txt)

        # Parallelisation : ensure only the root processor stores data,
        # then broadcasts to the rest
        NG = G_Gc.shape[0]
        chiks_GG = np.empty((NG, NG), dtype=complex)
        if self.chiks.world.rank == 0:  # Check if at root
            # Remove frequency axis
            chiks_GG[:, :] = - chiks_wGG[0, :, :]
            # Where do we take the reactive part??? !!! XXX
            # Shouldn't this be more than a minus sign??? !!! XXX

        # Broadcast data to all ranks
        self.chiks.world.broadcast(chiks_GG, 0)

        return G_Gc, chiks_GG

    def _calculate_component(self, spincomponent, pd, wd):
        """Hack to return chiKS twice instead of chiks, chi."""
        chiks_wGG = self.calculate_ks_component(spincomponent, pd,
                                                wd, txt=self.cfd)

        print('\nFinished calculating component', file=self.cfd)
        print('---------------', flush=True, file=self.cfd)

        return pd, wd, chiks_wGG, chiks_wGG


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
