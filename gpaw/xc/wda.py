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
        Z_ig, Z_lower_g, Z_upper_g = self.get_Zs(wn_sg, ni_grid, lower, upper, grid, spin, gd1)

        # Get alphas
        # alpha_i = self.get_alphas(Z_i, Z_lower, Z_upper)

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
        if not np.allclose(np.max(n_sg), 0):
            maxval = max(np.max(n_sg), 100/np.mean(n_sg))
        else:
            maxval = 10
        
        return get_ni_grid(rank, size, maxval, pts_per_rank)

    def get_pts_per_rank(self):
        return min(10, 100//mpi.size + 1)

    def get_pairdist_g(self, grid, ni, spin):
        grid_distances = np.linalg.norm(grid, axis=0)
        from scipy.special import gamma
        exc = self.get_lda_xc(ni, spin)
        if np.allclose(exc, 0):
            return np.zeros_like(grid_distances) + 1

        lambd = - 3 * gamma(3/5) / (2 * gamma(2/5) * exc)

        C = -3 / (4 * np.pi * gamma(2/5) * ni * lambd**3)

        g = np.zeros_like(grid_distances)
        exp_val = (lambd / (grid_distances[grid_distances > 0]))**5
        g[grid_distances > 0] = 1 + C * (1 - np.exp(-exp_val))
        g[np.isclose(grid_distances, 0)] = 1 + C
        
        return g

    def get_lda_xc(self, n, spin):
        if np.allclose(n, 0):
            return 0

        from gpaw.xc.lda import lda_c, lda_x
        narr = np.array([n]).astype(np.float64)
        earr = np.array([0.]).astype(np.float64)
        varr = np.array([0.]).astype(np.float64)
        
        lda_x(spin, earr, n, varr)
        zeta = 0
        lda_c(spin, earr, narr, varr, zeta)
        
        return earr[0] 

    def standardmode_Z_derivative(self, grid, ni_value, spin):
        pairdist_g = self.get_pairdist_g(grid, ni_value, spin)
        return pairdist_g - 1

    def symmetricmode_Z_derivative(self, grid, ni_value, spin, indicator_function):
        raise NotImplementedError

        pairdist_g = self.get_pairdist_g(grid, ni_value, spin)
        return pairdist_g - 1

    def get_augmented_ni_grid(self, ni_grid, ni_lower, ni_upper):
        if ni_lower != ni_grid[0] and ni_upper != ni_grid[-1]:
            aug = np.hstack([[ni_lower], ni_grid, [ni_upper]])
            lower_off = 1
            upper_off = 1
        elif ni_lower != ni_grid[0]:
            aug = np.hstack([[ni_lower], ni_grid])
            lower_off = 1
            upper_off = 0
        elif ni_upper != ni_grid[-1]:
            aug = np.hstack([ni_grid, [ni_upper]])
            lower_off = 0
            upper_off = 1
        elif ni_lower == ni_grid[0] and ni_upper == ni_grid[-1]:
            aug = ni_grid
            lower_off = 0
            upper_off = 0
        else:
            raise ValueError("Could not get augmented grid, lower, upper, grid: {}, {}, {}".format(ni_lower, ni_upper, ni_grid))
        return aug, lower_off, upper_off

    def build_indicators(self, ni_grid, ni_lower, ni_upper):
        # We use linear indicators for simplicity
        # This makes it easy to ensure that f <= 1, f >= 0 and that they always sum to 1
        # The return value of this function is a list of scipy interpolation objects
        from scipy.interpolate import interp1d

        augmented_ni_grid, lower_off, upper_off = self.get_augmented_ni_grid(ni_grid, ni_lower, ni_upper)

        
        def build_an_indicator(target_index, val_grid):
            targets = np.zeros_like(val_grid)
            targets[target_index] = 1
            return interp1d(val_grid, targets, kind="linear", bounds_error=False, fill_value=0)
        ni_range = range(lower_off, len(augmented_ni_grid)-upper_off)

        return [build_an_indicator(j, augmented_ni_grid) for j in ni_range]

    def standardmode_Zs(self, n_sg, ni_grid, lower_ni, upper_ni, grid, spin, gd):
        # Get parametrized pair distribution functions
        # augmented_ni_grid = np.hstack([[lower], ni_grid, [upper]])
        
        #pairdist_ig = np.array([self.get_pairdist(grid, ni, spin) for ni in augmented_ni_grid])

        Z_isg = np.zeros(ni_grid.shape + n_sg.shape)

        pairdist_g = self.get_pairdist_g(grid, lower_ni, spin)
        Z_lower_sg = self.fold(n_sg, np.array([pairdist_g]) - 1)
        
        pairdist_g = self.get_pairdist_g(grid, upper_ni, spin)
        Z_upper_sg = self.fold(n_sg, np.array([pairdist_g]) - 1)

        for i, n in enumerate(ni_grid):
            pairdist_g = self.get_pairdist_g(grid, n, spin)
            Z_isg[i] = self.fold(n_sg, np.array([pairdist_g]) - 1)

        return Z_isg, Z_lower_sg, Z_upper_sg

    def fold(self, f, g):
        assert np.allclose(f, f.real)
        assert np.allclose(g, g.real)
        assert f.shape == g.shape
        F = np.fft.fftn(f)
        G = np.fft.fftn(g)

        res = np.fft.ifftn(F*G)
        return res.real

    def symmetricmode_Zs(self, n_sg, ni_grid, lower_ni, upper_ni, grid, spin, gd):
        augmented_ni_grid, lower_off, upper_off = self.get_augmented_ni_grid(ni_grid, lower_ni, upper_ni)

        pairdist_sg = np.array([self.get_pairdist_g(grid, lower_ni, spin)])
        ind_i = self.build_indicators(augmented_ni_grid, lower_ni, upper_ni)
        ind_sg = np.array([ind_i[0](n_sg[0])])
        def getZ(pairdist, ind):
            return self.fold(n_sg, pairdist - 1) + self.fold(n_sg*ind, pairdist)

        Z_lower_sg = getZ(pairdist_sg, ind_sg)
        Z_isg = []
        for i, n in enumerate(ni_grid):
            pd = np.array([self.get_pairdist_g(grid, n, spin)])
            ind = ind_i[i](n_sg[0])
            Z = getZ(pd, ind)
            Z_isg.append(Z)
        assert len(Z_isg) == len(ni_grid)
        pd = np.array([self.get_pairdist_g(grid, upper_ni, spin)])
        ind = ind_i[-1](n_sg[0])
        Z_upper_sg = getZ(pd, ind)

        return np.array(Z_isg), Z_lower_sg, Z_upper_sg


    def interpolate_this(self, val_i, lower, upper, target):
        inter_i = np.zeros_like(val_i)
        for i, val in enumerate(val_i):
            if i == 0:
                next = val_i[1]
                prev = lower
            elif i == len(val_i)-1:
                next = upper
                prev = val_i[i-1]
            else:
                next = val_i[i+1]
                prev = val_i[i-1]

            if (val <= target and next >= target) or (val >= target and next <= target):
                inter_i[i] = (next - target) / (next - val)
            elif (val <= target and prev >= target) or (val >= target and prev <= target):
                inter_i[i] = (prev - target) / (prev - val)

        return inter_i

    def get_alphas(self, Z_isg, Z_lower_sg, Z_upper_sg):
        alpha_isg = np.zeros_like(Z_isg) 

        for iz, Z_yxsi in enumerate(Z_isg.T):
            for iy, Z_xsi in enumerate(Z_yxsi):
                for ix, Z_si in enumerate(Z_xsi):
                    for inds, Z_i in enumerate(Z_si):
                        assert Z_i.ndim == 1
                        nlesser = (Z_i <= -1).astype(int).sum()
                        ngreater = (Z_i > -1).astype(int).sum()
                        alpha_isg[:, inds, ix, iy, iz] = self.interpolate_this(Z_i, Z_lower_sg[inds, ix, iy, iz], Z_upper_sg[inds, ix, iy, iz], -1)
        return alpha_isg

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None, a=None):
        return 0

