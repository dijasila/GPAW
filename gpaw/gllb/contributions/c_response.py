from gpaw.gllb.contributions.contribution import Contribution
from gpaw.gllb.contributions.contribution import Contribution
from gpaw.xc_functional import XCRadialGrid, XCFunctional, XC3DGrid
from gpaw.xc_correction import A_Liy
from gpaw.gllb import safe_sqr
from math import sqrt, pi

import numpy as npy

class C_Response(Contribution):
    def __init__(self, nlfunc, weight, coefficients):
        Contribution.__init__(self, nlfunc, weight)
        self.coefficients = coefficients
        
    # Initialize Response functional
    def initialize_1d(self):
        self.ae = self.nlfunc.ae

    # Calcualte the GLLB potential and energy 1d
    def add_xc_potential_and_energy_1d(self, v_g):
        w_i = self.coefficients.get_coefficients_1d()
        u2_j = safe_sqr(self.ae.u_j)
        v_g += self.weight * npy.dot(w_i * self.ae.f_j, u2_j) / (npy.dot(self.ae.f_j, u2_j) +1e-10)
        return 0.0 # Response part does not contribute to energy

    def initialize(self):
        self.vt_sg = self.nlfunc.finegd.empty(self.nlfunc.nspins)

    def calculate_spinpaired(self, e_g, n_g, v_g):
        raise NotImplementedError

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g, 
                                a2_g=None, aa2_g=None, ab2_g=None, deda2_g=None,
                                dedaa2_g=None, dedab2_g=None):
        raise NotImplementedError

    def calculate_energy_and_derivatives(self, D_sp, H_sp, a):
        raise NotImplementedError

    def add_smooth_xc_potential_and_energy_1d(self, vt_g):
        w_i = self.coefficients.get_coefficients_1d()
        u2_j = safe_sqr(self.ae.s_j) # s_j, not u_j!
        vt_g += self.weight * npy.dot(w_i * self.ae.f_j, u2_j) / (npy.dot(self.ae.f_j, u2_j) +1e-10)
        return 0.0 # Response part does not contribute to energy
        
    def add_extra_setup_data(self, dict):
        # GLLBScr has not any special data
        pass
        
