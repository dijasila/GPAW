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

    def add_correction(self, kpt, psit_xG, R_xG, approximate=False, n_x=None):
        pass
    
    def add_paw_correction(self, kpt, c_xi, approximate=False, n_x=None):
        pass
    
    def rotate(self, kpt, U_nn):
        pass

    def get_kinetic_energy_correction(self):
        return self.ekin
