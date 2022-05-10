"""
Implements various versions of resolution of identity method
for hybrid functionals.

    Indices
         a      atoms

         j      index of radial splines, which are each assigned a given l-channel

         m      Indices m-quantum number of given l. Has 2*l+1 items.

         aj     list of atom wise spline list

         al     list of atom wise spline list with angular momentum matching
                the l index.

         M      running basis function index. (a,j,m) tuple

         A      running auxiliary function index. (a,j,m) tuple.

         G      running generalized gaussian index (a,l,m) tuple.

         x      running atom centered radial product index of phi_j and auxt_j,
                extra l spanning possible product channels. (a, j, j', l) tuple.

         X      running atom centered radial product index with angular
                momentum. (a, j, j', l, m) tuple.

    Arrays
         W_AA   auxiliary-auxiliary 2-center integrals.


"""


from gpaw.xc import XC
import numpy as np
from gpaw.auxlcao.algorithm import RILVL
from gpaw.auxlcao.reference_algorithm import RIVFullBasisDebug, RIVRestrictedBasisDebug

class LCAOHybrid:
    orbital_dependent = True
    orbital_dependent_lcao = True
    type = 'auxhybrid'

    def __init__(self, xcname:str, algorithm=None, omega = None, threshold=1e-2,
                                   debug={'cube':False,'ref':False}):
 
        self.screening_omega = 0.0
        self.threshold = threshold
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

        self.debug = debug

        self.evv = np.nan
        self.ecc = np.nan
        self.evc = np.nan
        self.ekin = np.nan

        self.ldaxc = XC('LDA')
        self.use_lda = True

        if algorithm == 'RIVRestrictedDebug':
            assert self.screening_omega == 0.0
            self.ri_algorithm = RIVRestrictedBasisDebug(self.exx_fraction)
            print('Threshold ignored?')
        if algorithm == 'RIVFullBasisDebug':
            assert self.screening_omega == 0.0
            self.ri_algorithm = RIVFullBasisDebug(self.exx_fraction, screening_omega = self.screening_omega)
            print('Threshold ignored?')
        elif algorithm == 'RI-LVL':
            self.ri_algorithm = RILVL(exx_fraction = self.exx_fraction, screening_omega = self.screening_omega, threshold=self.threshold)
        else:
            if algorithm is None:
                s = 'Please spesify the algorithm variable i.e. xc=''%s:backend=aux-lcao:algorithm=ALG''\n'
            else:
                s = 'Unknown algorithm.'
            s += 'Available algorithms are:\n'
            s += '    RIVRestrictedDebug\n'
            s += '    RIVFullBasisDebug\n'
            s += '    RI-LVL\n'
            raise ValueError(s)

    def set_grid_descriptor(self, gd):
        pass

    """ Calculate the semi-local part of the hybrid energy

    """
    def calculate(self, gd, nt_sr, vt_sr):
        if self.use_lda:
            return self.ldaxc.calculate(gd, nt_sr, vt_sr)
        energy = self.ecc + self.evv + self.evc
        energy += self.localxc.calculate(gd, nt_sr, vt_sr)
        return energy

    def initialize(self, density, hamiltonian, wfs):
        self.ecc = 0
        for setup in wfs.setups:
            if setup.ExxC is not None:        
                self.ecc += setup.ExxC * self.exx_fraction
            else:
                print('Warning, setup does not have core exchange')
        self.ri_algorithm.initialize(density, hamiltonian, wfs)

    def set_positions(self, spos_ac):
        self.ri_algorithm.set_positions(spos_ac, self.debug)

    def get_description(self):
        return 'Experimental aux-lcao RI-'+self.name

    def get_kinetic_energy_correction(self):
        return self.ekin

    def get_setup_name(self):
        return 'PBE'

    def calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        return self.localxc.calculate_paw_correction(setup, D_sp, dH_sp, a=a)

    def summary(self, log):
        # Take no changes
        if self.use_lda:
            raise ValueError('Error: Due to an internal error, LDA was used thorough the calculation.')

        log(self.get_description())

    def add_nlxc_matrix(self, H_MM, dH_asp, wfs, kpt):
        self.evv = 0.0
        self.evc = 0.0
        self.ekin = 0.0
        if self.use_lda:
            self.use_lda = False
            return

        if self.debug['cube']:
            self.ri_algorithm.cube_debug(self.debug['cube'])

        self.evv, self.evc, self.ekin = self.ri_algorithm.nlxc(H_MM, dH_asp, wfs, kpt)


    def write(self, *args):
        pass

    def read(self, *args):
        pass
