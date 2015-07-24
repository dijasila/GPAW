"""This module defines different external potentials."""
import numpy as np

from ase.units import Bohr, Hartree

import _gpaw

__all__ = ['ConstantPotential', 'ConstantElectricField']


def dict2potential(dct):
    if dct['name'] not in __all__:
        raise ValueError
    return globals()[dct['name']](**dct['kwargs'])
    

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
        
    def __str__(self):
        return 'Constant potential: {0:.3f} eV'.format(self.constant * Hartree)
        
    def calculate_potential(self, gd):
        self.vext_g = gd.zeros() + self.constant

    def todict(self):
        return {'name': 'ConstantPotential',
                'kwargs': {'constant': self.constant * Hartree}}
    
        
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

    def __str__(self):
        return ('Constant electric field: '
                '({0:.3f}, {0:.3f}, {0:.3f}) eV/Ang'
                .format(*(self.field * Hartree / Bohr)))

    def calculate_potential(self, gd):
        assert not gd.pbc_c.any()
        center_v = 0.5 * gd.cell_cv.sum(0)
        r_gv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        self.vext_g = np.dot(r_gv - center_v, self.field)
        
    def todict(self):
        return {'name': 'ConstantElectricField',
                'kwargs': {'strength': Hartree / Bohr,
                           'direction': self.field}}


class PointChargePotential(ExternalPotential):
    def __init__(self, q_p, R_pv=None, rc=0.2):
        self.q_p = np.ascontiguousarray(q_p, float)
        self.rc = rc / Bohr
        if R_pv is not None:
            self.set_positions(R_pv)
        else:
            self.R_pv = None
            
    def __str__(self):
        return ('Point-charge potential '
                '(points: {0}, cutoff: {1:.3f} Ang)'
                .format(len(self.q_p), self.rc * Bohr))
            
    def set_positions(self, R_pv):
        self.R_pv = np.asarray(R_pv) / Bohr
        self.vext_g = None
        
    def calculate_potential(self, gd):
        assert gd.orthogonal
        self.vext_g = gd.zeros()
        _gpaw.pc_potential(gd.beg_c, gd.h_cv.diagonal().copy(),
                           self.q_p, self.R_pv, self.rc, self.vext_g)

    def get_forces(self, calc):
        dens = calc.density
        F_pv = np.zeros_like(self.R_pv)
        gd = dens.finegd
        _gpaw.pc_potential(gd.beg_c, gd.h_cv.diagonal().copy(),
                           self.q_p, self.R_pv, self.rc, self.vext_g,
                           dens.rhot_g, F_pv)
        return F_pv * Hartree / Bohr
