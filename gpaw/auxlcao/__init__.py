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

        self.ldaxc = XC('LDA')
        self.use_lda = True

    def set_grid_descriptor(self, gd):
        pass

    """ Calculate the semi-local part of the hybrid energy

    """
    def calculate(self, gd, nt_sr, vt_sr):
        if self.use_lda:
            print('Using LDA')
            return self.ldaxc.calculate(gd, nt_sr, vt_sr)
        self.ecc = 0.0
        self.evv = 0.0
        self.evc = 0.0
        print('at calculate')
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
        if self.use_lda:
            raise ValueError('Error: Due to an internal error, LDA was used thorough the calculation')
        log(self.get_description())        

    def add_nlxc_matrix(self, H_MM, dH_asp, wfs, kpt):
        self.evv = 0.0
        self.evc = 0.0
        self.ekin = 0.0
        if self.use_lda:
            self.use_lda = False
            return
        print('Not using LDA anymore in add_nlxc')


