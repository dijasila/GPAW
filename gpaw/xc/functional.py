class XCNull:
    type = 'LDA'
    name = 'null'
    def calculate(self, n_sg, e_g, dedn_sg):
        e_g[:] = 0.0


class XCFunctional:
    hybrid = 0.0
    def __init__(self, name):
        self.name = name
        self.gd = None
        self.ekin = 0.0

    def initialize(self, density, hamiltonian, wfs):
        pass

    def set_grid_descriptor(self, gd):
        self.gd = gd

    def add_correction(self, kpt, psit_nG, R_nG):
        pass
    
    def add_paw_correction(self, kpt, c_ani):
        pass
    
    def add_correction2(self, kpt, psit_xG, R_xG, n_x=None):
        pass
    
    def add_paw_correction2(self, kpt, c_axi, n_x=None):
        pass
    
    def rotate(self, kpt, U_nn):
        pass

    def get_kinetic_energy_correction(self):
        return self.ekin

    def set_positions(self, spos_ac):
        pass
    
    def forces(self, F_av):
        pass

    def summary(self, fd):
        pass
