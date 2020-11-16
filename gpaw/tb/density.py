import numpy as np

from gpaw.density import Density


class TBDensity(Density):
    def set_positions(self, spos_ac, atom_partition):
        self.set_positions_without_ruining_everything(spos_ac, atom_partition)

    def initialize_density_from_atomic_densities(self, basis_functions, f_asi):
        pass

    def mix(self, comp_charge):
        self.error = 0.0

    def normalize(self, comp_charge):
        pass

    def calculate_pseudo_density(self, wfs):
        pass

    def calculate_dipole_moment(self):
        return np.zeros(3)
