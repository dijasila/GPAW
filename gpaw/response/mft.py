
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
