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
        
    def initialize(self, density, hamiltonian, wfs):
        pass

    def set_grid_descriptor(self, gd):
        self.gd = gd

    def add_correction(self, psit_nG, R_nG, approximate=False):
        pass
    
    def add_paw_correction(self, c_ni, approximate=False):
        pass
    
    def rotate(self, U_nn):
        pass
