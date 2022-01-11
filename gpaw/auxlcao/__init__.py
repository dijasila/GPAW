from gpaw.xc import XC
import numpy as np

from gpaw.auxlcao.full4c import Full4C
from gpaw.auxlcao.rilvl import RILVL
from gpaw.auxlcao.rilvl import RIR

class LCAOHybrid:
    orbital_dependent = True
    orbital_dependent_lcao = True
    type = 'auxhybrid'

    def __init__(self, xcname:str, algorithm=None, omega=None):
        self.screening_omega = 0.0
        self.name = xcname
        if xcname == 'EXX':
            self.exx_fraction = 1.0
            self.localxc = XC('null')
        elif xcname == 'PBE0':
            self.exx_fraction = 0.25
            self.localxc = XC('HYB_GGA_XC_PBEH')
        elif xcname == 'HSE06':
            self.exx_fraction = 0.25
            self.screening_omega = 0.11
            self.localxc = XC('HYB_GGA_XC_HSE06')
        elif xcname == 'HSEFAST':
            self.exx_fraction = 0.25
            self.screening_omega = 0.11
            self.localxc = XC('HYB_GGA_XC_PBEH')
        else:
            raise ValueError('Functional %s not supported by aux-lcao backend' % xcname)

        if omega is not None:
            print('Overriding omega from %.2f according to parameters to %.2f Bohr^-1' % (self.screening_omega, omega))
            self.screening_omega = omega

        self.evv = np.nan
        self.ecc = np.nan
        self.evc = np.nan
        self.ekin = np.nan

        self.ldaxc = XC('LDA')
        self.use_lda = True

        if algorithm == '4C':
            self.ri_algorithm = Full4C(exx_fraction = self.exx_fraction, screening_omega = self.screening_omega)
        elif algorithm == 'RI-LVL':
            self.ri_algorithm = RILVL(exx_fraction = self.exx_fraction, screening_omega = self.screening_omega)
        elif algorithm == 'RI-R':
            self.ri_algorithm = RIR(exx_fraction = self.exx_fraction, screening_omega = self.screening_omega)
        else:
            if algorithm is None:
                s = 'Please spesify the algorithm variable i.e. xc=''%s:backend=aux-lcao:algorithm=ALG''\n'
            else:
                s = 'Unknown algorithm.'
            s += 'Available algorithms are:\n'
            s += '    4C       Full four center integrals evaluated by projecting basis function products to plane wave expansions. For debugging purposes only.\n'
            s += '    RI-LVL   Resolution of identity 2 atomic sphere local Coulomb potential fit\n'
            s += '    RI-R     Real space implementation of RI-LVL'
            raise ValueError(s)

    def set_grid_descriptor(self, gd):
        pass



    """ 

         Calculate the semi-local part of the hybrid energy

    """
    def calculate(self, gd, nt_sr, vt_sr):
        print(nt_sr.shape)
        if self.use_lda:
            self.use_lda = False
            return self.ldaxc.calculate(gd, nt_sr, vt_sr)
        evv, self.ekin = self.ri_algorithm.calculate_non_local()
        energy = evv + self.ecc + self.localxc.calculate(gd, nt_sr, vt_sr)
        print('returning ', energy, 'self.ekin is', self.ekin)
        return energy

    def initialize(self, density, hamiltonian, wfs):
        self.ecc = 0.0
        for setup in wfs.setups:
            if setup.ExxC is not None:        
                self.ecc += setup.ExxC * self.exx_fraction
            else:
                print('Warning, setup does not have core exchange')
        self.ri_algorithm.initialize(density, hamiltonian, wfs)

    def set_positions(self, spos_ac):
        self.ri_algorithm.set_positions(spos_ac)

    def get_description(self):
        return 'Experimental aux-lcao' + self.name + ' with algorithm' + self.ri_algorithm.name

    def get_kinetic_energy_correction(self):
        print('Ekin corr', self.ekin)
        return self.ekin

    def get_setup_name(self):
        return 'PBE'

    def calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        evv, ekin = self.ri_algorithm.calculate_paw_correction(setup, D_sp, dH_sp, a)
        evv += self.localxc.calculate_paw_correction(setup, D_sp, dH_sp, a=a)
        self.ekin += ekin
        return evv

    def summary(self, log):
        # Take no changes
        if self.use_lda:
            raise ValueError('Error: Due to an internal error, LDA was used thorough the calculation.')

        log(self.get_description())

    def add_nlxc_matrix(self, H_MM, dH_asp, wfs, kpt, yy):
        if self.use_lda:
            self.use_lda = False
            return

        self.ri_algorithm.nlxc(H_MM, dH_asp, wfs, kpt, yy)

    def write(self, *args):
        print('Warning: XC not writing to file')
        pass

    def read(self, *args):
        print('Warning: XC not reading from file')
        pass
