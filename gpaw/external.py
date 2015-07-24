"""This module defines different external potentials."""
import numpy as np

from ase.units import Bohr, Hartree

import _gpaw


class ExternalPotential:
    vext_g = None

    def get_potential(self, gd):
        if self.vext_g is None:
            self.calculate_potential(gd)
        return self.vext_g


class ConstantPotential(ExternalPotential):
    """Constant potential for tests."""
    def __init__(self, constant=1.):
        self.constant = constant / Hartree

    def calculate_potential(self, gd):
        self.vext_g = gd.zeros() + self.constant

        
class ConstantElectricField(ExternalPotential):
    def __init__(self, strength, direction=[0, 0, 1]):
        """External constant electric field.
        
        strength: float
            Field strength in eV/Ang.
        direction: vector
            Polarisation direction.
        """
        d = np.asarray(direction)
        self.field = strength * d / (d**2).sum()**0.5 * Bohr / Hartree

    def calculate_potential(self, gd):
        assert not gd.pbc_c.any()
        center_v = 0.5 * gd.cell_cv.sum(0)
        r_gv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        self.vext_g = np.dot(r_gv - center_v, self.field)


class PointChargePotential(ExternalPotential):
    def __init__(self, R_pv, q_p, rc=0.2):
        self.R_pv = np.asarray(R_pv) / Bohr
        self.q_p = np.ascontiguousarray(q_p, float)
        self.rc = rc / Bohr
        
    def calculate_potential(self, gd):
        assert gd.orthogonal
        self.vext_g = gd.zeros()
        _gpaw.pc_potential(gd.beg_c, gd.h_cv.diagonal().copy(),
                           self.R_pv, self.q_p, self.rc, self.vext_g)

    def forces(self, gd, rhot_g):
        F_pv = np.zeros_like(self.R_pv)
        _gpaw.pc_potential(gd.beg_c, gd.h_cv.diagonal().copy(),
                           self.R_pv, self.q_p, self.rc, self.vext_g,
                           rhot_g, F_pv)
        return F_pv
