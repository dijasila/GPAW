import numpy as np


class LDA:
    hybrid = 0.0
    def __init__(self, xckernel):
        self.xckernel = xckernel
        self.name = xckernel.name
        self.gd = None
        
    def initialize(self, density, hamiltonian, wfs):
        pass

    def set_grid_descriptor(self, gd):
        self.gd = gd
        
    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        if gd is not self.gd:
            self.set_grid_descriptor(gd)
        if e_g is None:
            e_g = gd.empty()
        if v_sg is None:
            v_sg = np.zeros_like(n_sg)
        self.calculate_lda(e_g, n_sg, v_sg)
        return gd.integrate(e_g)

    def calculate_lda(self, e_g, n_sg, v_sg):
        self.xckernel.calculate(e_g, n_sg, v_sg)

    def calculate_radial(self, rgd, n_sLg, Y_L, v_sg):
        e_g = rgd.empty()
        n_sg = np.dot(Y_L, n_sLg)
        self.xckernel.calculate(e_g, n_sg, v_sg)
        return rgd.integrate(e_g)

    def add_non_local_terms(self, psit_xG, Htpsit_xG, kpt):
        pass

    def adjust_non_local_residual(self, psit_G, Htpsit_G, kpt, n):
        pass
