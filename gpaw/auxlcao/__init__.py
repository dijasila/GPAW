"""
Implements various versions of resolution of identity method
for hybrid functionals.

"""

from gpaw.xc import XC
import numpy as np

class LCAOHybrid:
    orbital_dependent = True
    orbital_dependent_lcao = True
    type = 'auxhybrid'

    def __init__(self, xcname:str, **args):
        print(args)
 
        if xcname == 'EXX':
            self.exx_fraction = 1.0
            self.name = xcname
            self.localxc = XC('null')
        elif xcname == 'PBE0':
            self.exx_fraction = 0.25
            self.name = xcname
            self.localxc = XC('HYB_GGA_XC_PBEH')
        else:
            raise ValueError('Functional %s not supported by aux-lcao backend' % xcname)

        self.evv = np.nan
        self.ecc = np.nan
        self.evc = np.nan

    def set_grid_descriptor(self, gd):
        pass

    """ Calculate the semi-local part of the hybrid energy

    """
    def calculate(self, gd, nt_sr, vt_sr):
        self.ecc = 0.0
        self.evv = 0.0
        self.evc = 0.0
        energy = self.ecc + self.evv + self.evc
        energy += self.localxc.calculate(gd, nt_sr, vt_sr)
        return energy

    def initialize(self, density, hamiltonian, wfs):
        pass

    def set_positions(self, spos_ac):
        pass

    def get_description(self):
        return 'Experimental aux-lcao RI-'+self.name

    def get_kinetic_energy_correction(self):
        return 0.0

    def get_setup_name(self):
        return 'PBE'

    def calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        return self.localxc.calculate_paw_correction(setup, D_sp, dH_sp, a=a)

    def summary(self, log):
        log(self.get_description())

