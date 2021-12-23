
# General modules
import numpy as np
import sys

# GPAW modules
import gpaw.mpi as mpi
from gpaw.response.susceptibility import FourComponentSusceptibilityTensor
from gpaw.response.kxc import PlaneWaveAdiabaticFXC
from gpaw.response.site_kernels import calc_K_mixed_shapes
from gpaw.xc import XC

# ASE modules
from ase.units import Hartree


class IsotropicExchangeCalculator():
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

    def __init__(self, gs, sitePos_mv, shapes_m='sphere', ecut=100,
                 nbands=None, world=mpi.world):
        """Construct the IsotropicExchangeCalculator object

        Parameters
        ----------
        gs : str or gpaw calculator
            Calculator with converged ground state as input to the linear
            response calculation.
        sitePos_mv : nd.array, str or tuple
            Positions of magnetic sites.
            Options are : array of positions, 'atoms'
                or ('some atoms', [names of elements])
            E.g. ('some atoms', ['Fe', 'Co', 'Ni']) will use the positions
                of all Fe, Co and Ni atoms.
        shapes_m : str or list of str
            Shapes of the integration regions used to define magnetic moments.
            Options are 'sphere', 'cylinder' and 'unit cell'
        ecut : number
            Cutoff energy in eV
            In response calculation, include all G-vectors with G^2/2 < ecut
        nbands : int
            Maximum band index to include in response calculation.
        world : obj
            MPI communicator.

        """

        # Determine positions of magnetic sites
        atoms = gs.atoms
        if type(sitePos_mv) is str and sitePos_mv == 'atoms':
            sitePos_mv = atoms.get_positions()  # Absolute coordinates
        elif type(sitePos_mv) == tuple and sitePos_mv[0] == 'some atoms':
            # Which atomic sites to include
            siteFilter = np.array([x in sitePos_mv[1]
                                   for x in atoms.get_chemical_symbols()])
            sitePos_mv = atoms.get_positions()   # Absolute coordinates
            sitePos_mv = sitePos_mv[siteFilter]  # Filter for relevant atoms
        self.sitePos_mv = sitePos_mv
        self.N_sites = len(sitePos_mv)   # Number of magnetic sites

        # Determine shapes of integration regions
        if type(shapes_m) is str:
            shapes_m = [shapes_m]*self.N_sites
        self.shapes_m = shapes_m

        # Calculator for response function
        self.chiksf = StaticChiKSFactory(gs,
                                         ecut=ecut,
                                         nblocks=1,
                                         eta=0,
                                         nbands=nbands,
                                         world=world)

        # Calculator for xc-kernel
        self.Bxc_calc = AdiabaticBXC(self.chiksf.calc, world=world)

        # Make empty object for Bxc field
        self.Bxc_G = None

    def __call__(self, q_c, rc_rm=1, zc_rm='diameter', txt=sys.stdout):
        """Calculate the isotropic exchange between all magnetic sites
        for a given wavevector.

        Parameters
        ----------
        q_c : nd.array
            Components of wavevector in relative coordinates
        rc_rm : nd.array or number
            Characteristic size (radius) of integration region.
            If number, use same value for all sites
        zc_rm : nd.array, str, number of list of str
            Height of integration cylinder.
            Options are 'diameter', 'unit cell' or specifying directly
            as with rc_rm
        txt : str
            Where to save log-files

        Returns
        -------
        J_rmn : nd.array (dtype=complex)
            Exchange between magnetic sites (m,n) for different
            parameters of the integration regions (r).
        """

        # Get Bxc_G
        if self.Bxc_G is None:
            self._computeBxc()
        Bxc_G = self.Bxc_G

        # Compute transverse susceptibility
        _, chiks_GG = self.chiksf('-+', q_c, txt=txt)

        # Get plane-wave descriptor
        pd = self.chiksf.get_PWDescriptor(q_c)

        # Reformat rc_rm and get number of different radii
        N_sites = self.N_sites
        if type(rc_rm) in {int, float}:
            rc_rm = np.tile(rc_rm, [1, N_sites])
        Nr = len(rc_rm)     # Number of radii

        # Reformat zc_rm
        if type(zc_rm) in {int, float, str}:
            zc_rm = np.tile(zc_rm, [Nr, N_sites])

        # Loop through rc values
        J_rmn = np.zeros([Nr, N_sites, N_sites], dtype=np.complex128)
        for r in range(Nr):
            rc_m, zc_m = rc_rm[r], zc_rm[r]

            # Compute site-kernel
            K_GGm = calc_K_mixed_shapes(pd, self.sitePos_mv,
                                        shapes_m=self.shapes_m,
                                        rc_m=rc_m, zc_m=zc_m)

            # Compute exchange coupling
            J_mn = np.zeros([N_sites, N_sites], dtype=np.complex128)
            for m in range(N_sites):
                for n in range(N_sites):
                    Km_GG = K_GGm[:, :, m]
                    Kn_GG = K_GGm[:, :, n]
                    J = Bxc_G @ Kn_GG @ chiks_GG @ np.conj(Km_GG) \
                        @ np.conj(Bxc_G)
                    J_mn[m, n] = J
            J_rmn[r, :, :] = J_mn

        return J_rmn

    def _computeBxc(self):
        # Compute xc magnetic field
        # Note : Bxc is calculated from the xc-kernel, which is a 2-point
        # function, while B_xc is 1-point Because of how the different
        # Fourier transforms are defined, this gives an extra volume factor
        # See eq. 50 of Phys. Rev. B 103, 245110 (2021)
        print('Computing Bxc')
        # Plane-wave descriptor (input is arbitrary)
        self.pd0 = self.chiksf.get_PWDescriptor([0, 0, 0])
        Omega_cell = self.pd0.gd.volume
        Bxc_GG = self.Bxc_calc(self.pd0)
        self.Bxc_G = Omega_cell * Bxc_GG[:, 0]
        print('Done computing Bxc')


class StaticChiKSFactory(FourComponentSusceptibilityTensor):
    """Class calculating components of the static Kohn-Sham
    susceptibility tensor
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
        fixed_kwargs = {'gammacentered': True, 'disable_point_group': True,
                        'disable_time_reversal': True,
                        'bundle_integrals': True, 'bundle_kptpairs': False}

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
            chiks_GG[:, :] = chiks_wGG[0, :, :]

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
    in the plane wave mode"""

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

        return (v_sG[0] - v_sG[1])/2    # Definition of Bxc


def compute_magnon_energy_simple(J_q, q_qc, mm):
    """Compute magnon energy with single atom in magnetic unit cell"""
    # Check if J_mnq was passed instead of J_q
    if len(J_q.shape) == 3:
        J_q = J_q[0, 0, :]

    # Find index of Gamma point (q=0), i.e. row with all zeros
    zeroIndex = np.argwhere(np.all(q_qc == 0, axis=1))
    zeroIndex = int(zeroIndex[0])

    # Compute energies
    J0 = J_q[zeroIndex]
    E_q = 2 / mm * (J0 - J_q)

    # Imaginary part should be zero
    assert np.all(np.isclose(np.imag(E_q), 0))
    E_q = np.real(E_q)

    return E_q


def compute_magnon_energy_FM(J_mnq, q_qc, mm, return_H=False):
    """Compute magnon energy for ferromagnet with multiple sublattices
    Gamma point (q=0) must be included in dataset.
    """

    import numpy as np
    from numpy.linalg import eigvalsh

    N_sites, N_sites, Nq = J_mnq.shape

    # Reformat magnetisation
    if type(mm) in {float, int}:
        mm = np.ones(N_sites)*mm

    # Find rows where all components of q_c are zero (Gamma point)
    zeroIndex = np.argwhere([np.all(np.isclose(q_qc[q, :], 0))
                             for q in range(Nq)])
    try:
        zeroIndex = int(zeroIndex[0])
    except IndexError:
        zeroIndex = 0
    J0_mn = J_mnq[:, :, zeroIndex]   # Get J_mn(0)

    # Set up Hamiltonian matrix
    mm_inv_mn = np.diag(1 / mm)  # 1/M_mu * delta_mu,nu
    # 2/M_mu * sum_nu' J_mu,nu'(0) delta_mu,nu
    firstTerm_mn = 2 * mm_inv_mn * np.sum(J0_mn, axis=-1, keepdims=True)
    firstTerm_mnq = np.tile(firstTerm_mn[..., np.newaxis], [1, 1, Nq])
    mmProd_mn = np.outer(mm, mm)  # M_mu * M_nu
    mmProd_mnq = np.tile(mmProd_mn[:, :, np.newaxis],
                         [1, 1, Nq])  # Match dimension of J_mnq
    J_nmq = np.transpose(J_mnq, axes=[1, 0, 2])  # J^\nu\mu
    # -2J^nu,mu(q) / sqrt(M_mu * M_nu)
    secondTerm_mnq = -2 * J_nmq / np.sqrt(mmProd_mnq)
    H_mnq = firstTerm_mnq + secondTerm_mnq

    # Diagonalise Hamiltonian for all q values
    E_mq = np.zeros([N_sites, Nq])
    for q in range(Nq):
        H_mn = H_mnq[:, :, q]
        assert np.all(np.isclose(np.conj(H_mn.T), H_mn, atol=np.inf,
                                 rtol=1e-02))  # Check if Hermitian
        # 'eigvalsh' takes the lower triangluar part, then assumes
        #   Hermiticity to fill the rest of the matrix
        # This is faster than 'eigvals' and guarantees that the computed
        #   eigenvalues are real.
        E_mq[:, q] = eigvalsh(H_mn, UPLO='L')

    if return_H:
        return E_mq, H_mnq
    else:
        return E_mq
