class XCFunctional:
    orbital_dependent = False
    def __init__(self, name):
        self.name = name
        self.gd = None
        self.ekin = 0.0
        
    def get_setup_name(self):
        return self.name
    
    def initialize(self, density, hamiltonian, wfs, occupations):
        pass

    def set_grid_descriptor(self, gd):
        self.gd = gd

    def correct_hamiltonian_matrix(self, kpt, psit_nG, H_nn, Htpsit_nG,
                                   dH_asp):
        pass

    def add_correction(self, kpt, psit_xG, R_xG, P_axi, c_axi, n_x=None,
                       calculate_change=False):
        pass
    
    def rotate(self, kpt, U_nn):
        pass

    def get_kinetic_energy_correction(self):
        return self.ekin

    def set_positions(self, spos_ac):
        pass
    
    def add_forces(self, F_av):
        pass

    def summary(self, fd):
        pass
