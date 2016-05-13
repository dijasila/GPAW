"""Helper classes for doing jellium calculations."""

from math import pi

import numpy as np
from ase.units import Bohr

from gpaw.poisson import PoissonSolver

# Following two classes are old
class JelliumPoissonSolver(PoissonSolver):
    """Jellium Poisson solver."""
    
    mask_g = None  # where to put the jellium
    rs = None  # Wigner Seitz radius
    
    def get_mask(self, r_gv):
        """Choose which grid points are inside the jellium.

        r_gv: 4-dimensional ndarray
            positions of the grid points in Bohr units.

        Return ndarray of ones and zeros indicating where the jellium
        is.  This implementation will put the positive background in the
        whole cell.  Overwrite this method in subclasses."""
        
        return self.gd.zeros() + 1.0
        
    def initialize(self):
        PoissonSolver.initialize(self)
        r_gv = self.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        self.mask_g = self.get_mask(r_gv).astype(float)
        self.volume = self.gd.comm.sum(self.mask_g.sum()) * self.gd.dv
        
    def solve(self, phi, rho, eps=None, charge=0, maxcharge=1e-6,
              zero_initial_phi=False):

        if eps is None:
            eps = self.eps
        
        self.rs = (3 / pi / 4 * self.volume / charge)**(1 / 3.0)
        
        rho -= self.mask_g * (charge / self.volume)
        niter = self.solve_neutral(phi, rho, eps=eps)
        return niter


class JelliumSurfacePoissonSolver(JelliumPoissonSolver):
    def __init__(self, z1, z2, **kwargs):
        """Put the positive background charge where z1 < z < z2.

        z1: float
            Position of lower surface in Angstrom units.
        z2: float
            Position of upper surface in Angstrom units."""
        
        PoissonSolver.__init__(self, **kwargs)
        self.z1 = (z1 - 0.0001) / Bohr
        self.z2 = (z2 - 0.0001) / Bohr

    def get_mask(self, r_gv):
        return np.logical_and(r_gv[:, :, :, 2] > self.z1,
                              r_gv[:, :, :, 2] < self.z2)



class Jellium():
    """ The Jellium object """
    def __init__(self, charge):
        """ Initialize the Jellium object
        Input: charge, a positive number, the total Jellium background charge"""
        self.charge = charge
        self.rs = None  # the Wigner-Seitz radius
        self.volume = None
        self.mask_g = None
        self.gd = None

    def set_grid_descriptor(self, gd):
        """ Set the grid descriptor for the Jellium background charge"""
        self.gd = gd
        self.mask_g = self.get_mask().astype(float)
        self.volume = self.gd.comm.sum(self.mask_g.sum()) * self.gd.dv
        self.rs = (3 / pi / 4 * self.volume / self.charge)**(1 / 3.0)

    def get_mask(self):
        """Choose which grid points are inside the jellium.

        gd: grid descriptor

        Return ndarray of ones and zeros indicating where the jellium
        is.  This implementation will put the positive background in the
        whole cell.  Overwrite this method in subclasses."""
        
        return self.gd.zeros() + 1.0

    def add_to(self, rhot_g):
        """ Add Jellium background charge to pseudo charge density rhot_g"""
        rhot_g -= self.mask_g * (self.charge / self.volume)
        return rhot_g
        
class JelliumSlab(Jellium):
    """ The Jellium slab object """
    def __init__(self, charge, z1, z2):
        """Put the positive background charge where z1 < z < z2.
        
        z1: float
            Position of lower surface in Angstrom units.
        z2: float
            Position of upper surface in Angstrom units."""
        Jellium.__init__(self, charge)
        self.z1 = (z1 - 0.0001) / Bohr
        self.z2 = (z2 - 0.0001) / Bohr

    def get_mask(self):
        r_gv = self.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        return np.logical_and(r_gv[:, :, :, 2] > self.z1,
                              r_gv[:, :, :, 2] < self.z2)
