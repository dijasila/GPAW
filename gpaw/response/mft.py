
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
