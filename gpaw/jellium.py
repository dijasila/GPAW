"""Helper classes for doing jellium calculations."""

from math import pi

import numpy as np
from ase.units import Bohr

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

    def add_charge_to(self, rhot_g):
        """ Add Jellium background charge to pseudo charge density rhot_g"""
        rhot_g -= self.mask_g * (self.charge / self.volume)

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
        #r_gv: 4-dimensional ndarray
        #    positions of the grid points in Bohr units.
        return np.logical_and(r_gv[:, :, :, 2] > self.z1,
                              r_gv[:, :, :, 2] < self.z2)
