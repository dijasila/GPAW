from gpaw.xc.functional import XCFunctional
import numpy as np
from gpaw import mpi


class WDA(XCFunctional):
    def __init__(self, mode=None, densitymode=None):
        XCFunctional.__init__(self, 'WDA', 'LDA')

        if densitymode is not None and densitymode.lower() == "valence":
            self.get_working_density = self.get_valence_density
        else:
            self.get_working_density = self.get_corrected_density

        if mode is not None and mode.lower() == "symmetric":
            self.get_Zs = self.symmetricmode_Zs
        else:
            self.get_Zs = self.standardmode_Zs
        

    def initialize(self, density, hamiltonian, wfs, occupations):
        self.wfs = wfs
        self.density = density
        pass

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        if len(n_sg) != 1:
            raise NotImplementedError
        spin = 0

        # Get full grid gd
        gd1 = gd.new_descriptor(comm=mpi.serial_comm)
        grid = gd1.get_grid_point_coordinates()
        
        # Get density: AE, Corrected or Valence
        wn_sg = self.get_working_density(n_sg, gd1)
    
        # Get ni_grid
        ni_grid, lower, upper = self.get_ni_grid(mpi.rank, mpi.size, wn_sg)


        # Get Zs
        Z_i, Z_lower, Z_upper = self.get_Zs(wn_sg, ni_grid, lower, upper, grid, spin, gd1)
        # Get alphas

        # Calculate Vs
        # Add correction if symmetric mode

        # Calculate energy
        # Add correction if symmetric mode

        

        pass

    
    def get_corrected_density(self, n_sg, gd):
        from gpaw.xc.WDAUtils import correct_density
        return correct_density(n_sg, gd, self.wfs.setups, self.wfs.spos_ac)

    def get_valence_density(self, n_sg):
        raise NotImplementedError

    def get_ni_grid(self, rank, size, n_sg):
        from gpaw.xc.WDAUtils import get_ni_grid
        pts_per_rank = self.get_pts_per_rank()
        return get_ni_grid(rank, size, n_sg, pts_per_rank)

    def get_pts_per_rank(self):
        return min(10, 100//mpi.size + 1)

    def get_pairdist_g(self, grid, ni, spin):
        grid_distances = np.linalg.norm(grid, axis=0)
        from scipy.special import gamma
        exc = self.get_lda_xc(ni, spin)
        
        lambd = 3 * gamma(3/5) / (2 * gamma(2/5) * exc)

        C = 3 / (4 * np.pi * gamma(2/5) * ni * lambd**3)

        g = np.zeros_like(grid_distances)
        g[grid_distances > 0] = 1 + C * (1 - np.exp(-(lambd / (grid_distances[grid_distances > 0]))**5))
        g[grid_distances == 0] = 1 + C
        
        return g

    def get_lda_xc(self, n, spin):
        from gpaw.xc.lda import lda_c, lda_x
        narr = np.array([n]).astype(np.float64)
        earr = np.array([n]).astype(np.float64)
        varr = np.array([n]).astype(np.float64)

        lda_x(spin, earr, narr, varr)
        zeta = 0
        lda_c(spin, earr, narr, varr, zeta)
        
        return earr[0] 

    def standardmode_Z_derivative(self, ni_value):
        raise NotImplementedError

    def symmetricmode_Z_derivative(self, ni_value, indicator_function):
        raise NotImplementedError

    def build_indicators(self, ni_grid):
        raise NotImplementedError

    def standardmode_Zs(self, n_sg, ni_grid, lower_ni, upper_ni, grid, spin, gd):
        # Get parametrized pair distribution functions
        # augmented_ni_grid = np.hstack([[lower], ni_grid, [upper]])
        
        #pairdist_ig = np.array([self.get_pairdist(grid, ni, spin) for ni in augmented_ni_grid])

        Z_i = np.zeros_like(ni_grid)

        pairdist_g = self.get_pairdist_g(grid, lower_ni, spin)
        Z_lower = gd.integrate(n_sg*(np.array([pairdist_g]) - 1))
        
        pairdist_g = self.get_pairdist_g(grid, upper_ni, spin)
        Z_upper = gd.integrate(n_sg*(np.array([pairdist_g]) - 1))

        for i, n in enumerate(ni_grid):
            pairdist_g = self.get_pairdist_g(grid, n, spin)
            Z_i[i] = gd.integrate(n_sg*(np.array([pairdist_g]) - 1))

        return Z_i, Z_lower, Z_upper

    def symmetricmode_Zs(self, ni_grid, n_sg):
        raise NotImplementedError


    def calculate_paw_corrections(self, setup, D_sp, dEdD_sp=None, a=None):
        return 0
