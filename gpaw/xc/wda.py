from gpaw.xc.functional import XCFunctional
from gpaw.xc.lda import PurePythonLDAKernel
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
            self.dZFunc = self.symmetric_dZ
        else:
            self.get_Zs = self.standardmode_Zs
            self.dZFunc = self.normal_dZ

        self.mode = mode
        self.densitymode = densitymode
        self.lda_kernel = PurePythonLDAKernel()
        print("LDA CORRELATION DISABLED")
        
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
        self.gd = gd1
        grid = gd1.get_grid_point_coordinates()
        
        # Get density: AE, Corrected or Valence
        wn_sg = self.get_working_density(n_sg, gd1)
    
        # Get ni_grid
        ni_grid, ni_lower, ni_upper = self.get_ni_grid(mpi.rank,
                                                       mpi.size, wn_sg)

        # Get Zs
        Z_isg, Z_lower_sg, Z_upper_sg = self.get_Zs(wn_sg,
                                                    ni_grid,
                                                    ni_lower,
                                                    ni_upper,
                                                    grid, spin,
                                                    gd1)

        # Get alphas
        alpha_isg = self.get_alphas(Z_isg, Z_lower_sg, Z_upper_sg)
        # Calculate Vs
        V_sg = self.calculate_V1(alpha_isg, wn_sg, grid, ni_grid)
        V_sg += self.calculate_V1p(alpha_isg, wn_sg, grid, ni_grid)

        dalpha_isg = self.get_dalpha_isg(alpha_isg, Z_isg, Z_lower_sg,
                                         Z_upper_sg, grid, ni_grid,
                                         ni_lower, ni_upper,
                                         len(n_sg), self.dZFunc)
        V_sg += self.calculate_V2(dalpha_isg, wn_sg, grid, ni_grid)

        # Add correction if symmetric mode
        if self.mode.lower() == "symmetric":
            V_sg += self.calculate_sym_pot_correction(alpha_isg,
                                                      wn_sg, grid, ni_grid)
        mpi.world.sum(V_sg)

        # Calculate energy
        eWDA_g = self.calculate_energy(alpha_isg, wn_sg, gd, grid, ni_grid)

        # Add correction if symmetric mode
        if self.mode is not None and self.mode.lower() == "symmetric":
            eWDA_g += self.calculate_sym_energy_correction(alpha_isg,
                                                           wn_sg, gd,
                                                           grid,
                                                           ni_grid)

        # Correct if we want to use WDA for valence density only
        if self.densitymode is not None:
            if self.densitymode.lower() == "valence":
                eWDA_g += self.calculate_energy_correction_valence_mode(wn_sg,
                                                                        n_sg)

        mpi.world.sum(eWDA_g)

        gd.distribute(eWDA_g, e_g)
        gd.distribute(V_sg, v_sg)

    def get_corrected_density(self, n_sg, gd):
        from gpaw.xc.WDAUtils import correct_density
        return correct_density(n_sg, gd, self.wfs.setups, self.wfs.spos_ac)

    def get_valence_density(self, n_sg):
        raise NotImplementedError

    def get_ni_grid(self, rank, size, n_sg):
        from gpaw.xc.WDAUtils import get_ni_grid
        pts_per_rank = self.get_pts_per_rank()
        if not np.allclose(np.max(n_sg), 0):
            maxval = max(np.max(n_sg), 100 * np.mean(n_sg))
        else:
            maxval = 10
        #maxval = np.mean(n_sg)*3
        maxval = 1.0
        return get_ni_grid(rank, size, maxval, pts_per_rank)

    def get_pts_per_rank(self):
        return 20
        # return min(10, 100 // mpi.size + 1)

    def get_pairdist_g(self, grid, ni, spin):
        grid_distances = np.linalg.norm(grid, axis=0)
        from scipy.special import gamma
        exc = self.get_lda_xc(ni, spin)
        assert exc <= 0
        if np.allclose(exc, 0):
            g = np.zeros_like(grid_distances)
            nni = 0.0000001
            # nni = 0.001
            exc = self.get_lda_xc(nni, spin)
            lambd = -3 * gamma(3/5) / (2 * gamma(2 / 5) * exc)
            C = -3 / (4 * np.pi * gamma(2 / 5) * nni * lambd**3)
            g[:] = C
            g = g + 1
            return g

        lambd = - 3 * gamma(3 / 5) / (2 * gamma(2 / 5) * exc)

        C = -3 / (4 * np.pi * gamma(2 / 5) * ni * lambd**3) # * 1000000
        g = np.zeros_like(grid_distances)
        exp_val = -(lambd / (grid_distances[grid_distances > 0]))**5
        g[grid_distances > 0] = 1 + C * (1 - np.exp(exp_val))
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
        # earr2 = np.zeros_like(earr)
        # lda_c(spin, earr, narr, varr, zeta)
        # assert (np.abs(earr2) < np.abs(earr)), "CORRE E: {}, EX E: {}".format(earr2, earr)
        
        return earr[0] / n

    def standardmode_Z_derivative(self, grid, ni_value, spin):
        pairdist_g = self.get_pairdist_g(grid, ni_value, spin)
        return pairdist_g - 1

    def symmetricmode_Z_derivative(self,
                                   grid,
                                   ni_value, spin, indicator_function):
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
            raise ValueError("Could not get augmented grid," +
                             "lower, upper, grid:" +
                             "{}, {}, {}".format(ni_lower, ni_upper, ni_grid))
        return aug, lower_off, upper_off

    def build_indicators(self, ni_grid, ni_lower, ni_upper):
        # We use linear indicators for simplicity
        # This makes it easy to ensure that f <= 1, f >= 0 and
        # that they always sum to 1
        # The return value of this function is a list of scipy
        # interpolation objects
        from scipy.interpolate import interp1d

        gang = self.get_augmented_ni_grid
        augmented_ni_grid, low_off, upp_off = gang(ni_grid,
                                                   ni_lower,
                                                   ni_upper)

        def build_an_indicator(target_index, val_grid):
            targets = np.zeros_like(val_grid)
            targets[target_index] = 1
            return interp1d(val_grid, targets, kind="linear",
                            bounds_error=False, fill_value=0)
        ni_range = range(low_off, len(augmented_ni_grid) - upp_off)

        return [build_an_indicator(j, augmented_ni_grid) for j in ni_range]

    def normal_dZ(self, i, s, G_isr, grid_vg, ni_lower, ni_upper, alpha_isg):
        if i < 0:
            return (self.get_pairdist_g(grid_vg, ni_lower, s) - 1).reshape(-1)
        elif i >= len(G_isr):
            return (self.get_pairdist_g(grid_vg, ni_upper, s) - 1).reshape(-1)
        else:
            return G_isr[i, s]

    def symmetric_dZ(self, i, s, G_isr, grid_vg,
                     ni_lower, ni_upper, alpha_isg):
        if i < 0:
            f1 = self.get_pairdist_g(grid_vg,
                                     ni_lower,
                                     s).reshape(-1)
            return f1 * (1 + alpha_isg[i, s].reshape(-1)) - 1
        elif i >= len(G_isr):
            f1 = (self.get_pairdist_g(grid_vg, ni_upper, s)).reshape(-1)
            return f1 * (1 + alpha_isg[i, s].reshape(-1)) - 1
        else:
            return (G_isr[i, s] + 1) * (1 + alpha_isg[i, s].reshape(-1)) - 1

    def standardmode_Zs(self, n_sg,
                        ni_grid, lower_ni,
                        upper_ni, grid, spin, gd):
        # Get parametrized pair distribution functions
        # augmented_ni_grid = np.hstack([[lower], ni_grid, [upper]])
        
        # pairdist_ig
        # = np.array([self.get_pairdi
        # st(grid, ni, spin) for ni in augmented_ni_grid])

        Z_isg = np.zeros(ni_grid.shape + n_sg.shape)

        pairdist_g = self.get_pairdist_g(grid, lower_ni, spin)
        Z_lower_sg = self.fold_w_pair(n_sg, np.array([pairdist_g]) - 1)
        
        pairdist_g = self.get_pairdist_g(grid, upper_ni, spin)
        Z_upper_sg = self.fold_w_pair(n_sg, np.array([pairdist_g]) - 1)

        for i, n in enumerate(ni_grid):
            pairdist_g = self.get_pairdist_g(grid, n, spin)
            Z_isg[i] = self.fold_w_pair(n_sg, np.array([pairdist_g]) - 1)
        # print("NORM ON THE INSIDE", gd.integrate(n_sg))

        return Z_isg, Z_lower_sg, Z_upper_sg


    def fftn(self, f_sg, axes=None):
        if axes is None:
            sqrtN = np.sqrt(np.array(f_sg.shape).prod())
        else:
            sqrtN = 1
            for ax in axes:
                sqrtN *= f_sg.shape[ax]
            sqrtN = np.sqrt(sqrtN)
            
        # return np.fft.fftn(f_sg, axes=axes)
        return np.fft.fftn(f_sg, axes=axes, norm="ortho") / sqrtN # This is correct: * sqrtN * (1 / (self.gd.dv * sqrtN**2)) * self.gd.dv # * self.gd.dv

    def ifftn(self, f_sg, axes=None):
        if axes is None:
            sqrtN = np.sqrt(np.array(f_sg.shape).prod())
        else:
            sqrtN = 1
            for ax in axes:
                sqrtN *= f_sg.shape[ax]
            sqrtN = np.sqrt(sqrtN)
          
        # return np.fft.ifftn(f_sg, axes=axes)
        return np.fft.ifftn(f_sg, axes=axes, norm="ortho") * sqrtN # / self.gd.dv

    def fold(self, f_sg, g_sg):
        assert np.allclose(f_sg, f_sg.real)
        assert np.allclose(g_sg, g_sg.real)
        assert f_sg.shape == g_sg.shape
        assert f_sg.ndim == 4


        N = np.array(f_sg[0].shape).prod()
        F = self.fftn(f_sg, axes=(1, 2, 3))
        G = self.fftn(g_sg, axes=(1, 2, 3))


        res = self.ifftn(F * G * N * self.gd.dv, axes=(1, 2, 3))
        assert np.allclose(res, res.real), "MEAN ABS DIFF: {}, MEAN ABS: {}, N: {}".format(np.mean(np.abs(res - res.real)), np.mean(np.abs(res.real)), N)
        # assert np.mean(np.abs(res.real)) < 1e2
        return res.real

    def fold_w_pair(self, f_sg, G_sg):
        assert np.allclose(f_sg, f_sg.real)
        assert np.allclose(G_sg, G_sg.real)
        assert f_sg.shape == G_sg.shape
        assert f_sg.ndim == 4


        N = np.array(f_sg[0].shape).prod()
        F = self.fftn(f_sg, axes=(1, 2, 3))
        G = self.fftn(G_sg, axes=(1, 2, 3))
        
        res = self.ifftn(F * G * N, axes=(1, 2, 3))
        assert np.allclose(res, res.real), "MEAN ABS DIFF: {}, MEAN ABS: {}, N: {}".format(np.mean(np.abs(res - res.real)), np.mean(np.abs(res.real)), N)

        return res.real

    def symmetricmode_Zs(self, n_sg, ni_grid,
                         lower_ni, upper_ni,
                         grid, spin, gd):
        augmented_ni_grid, _, _ = self.get_augmented_ni_grid(ni_grid,
                                                             lower_ni,
                                                             upper_ni)

        pairdist_sg = np.array([self.get_pairdist_g(grid, lower_ni, spin)])
        ind_i = self.build_indicators(augmented_ni_grid, lower_ni, upper_ni)
        ind_sg = np.array([ind_i[0](n_sg[0])])
        
        def getZ(pairdist, ind):
            return self.fold_w_pair(n_sg, pairdist - 1) + self.fold_w_pair(n_sg * ind,
                                                             pairdist)

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
        count = 0
        for i, val in enumerate(val_i):
            if i == 0:
                nextv = val_i[1]
                prev = lower
            elif i == len(val_i) - 1:
                nextv = upper
                prev = val_i[i - 1]
            else:
                nextv = val_i[i + 1]
                prev = val_i[i - 1]

            if (val <= target
                and nextv >= target) or (val >= target and nextv <= target):
                inter_i[i] = (nextv - target) / (nextv - val)
                count += 1
            elif (val <= target and
                  prev >= target) or (val >= target and prev <= target):
                inter_i[i] = (prev - target) / (prev - val)
                count += 1
        
        return inter_i

    def get_alphas(self, Z_isg, Z_lower_sg, Z_upper_sg):
        alpha_isg = np.zeros_like(Z_isg)

        for iz, Z_yxsi in enumerate(Z_isg.T):
            for iy, Z_xsi in enumerate(Z_yxsi):
                for ix, Z_si in enumerate(Z_xsi):
                    for inds, Z_i in enumerate(Z_si):
                        assert Z_i.ndim == 1
                        alpha_isg[:, inds, ix, iy, iz] = self.interpolate_this(
                            Z_i,
                            Z_lower_sg[inds, ix, iy, iz],
                            Z_upper_sg[inds, ix, iy, iz], -1)
        return alpha_isg

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None, a=None):
        return 0

    def _get_K_G(self, gd):
        assert gd.comm.size == 1 # Construct_reciprocal doesnt work in parallel
        k2_Q, _ = construct_reciprocal(gd)
        k2_Q[0,0,0] = 0
        return k2_Q**(1/2)

    def fold_with_vC(self, f_sg, g_sg, grid_vg):

        # K_G = self._get_K_G(gd)

        # assert np.allclose(f_sg, f_sg.real)
        # assert np.allclose(g_sg, g_sg.real)
        # assert f_sg.shape == g_sg.shape
        # assert f_sg.ndim == 4


        distances_sg = np.array([np.linalg.norm(grid_vg, axis=0)])
        cutoff = 1e-4
        distances_sg[distances_sg < cutoff] = cutoff
        assert (distances_sg >= cutoff).all()

        

        res_sg = self.fold_w_pair(f_sg, g_sg / distances_sg)

        return res_sg
    
    def get_G_isg(self, grid_vg, ni_j, numspins):
        return np.array(
            [[self.get_pairdist_g(grid_vg, ni, spin) - 1
              for spin in range(numspins)]
             for ni in ni_j])

    def calculate_V1(self, alpha_isg, n_sg, grid_vg, ni_j):
        # Convolve n_sg and G/v_C
        # Multiply alpha_ig to result
        G_isg = self.get_G_isg(grid_vg, ni_j, len(n_sg))
        folded_isg = np.array(
            [self.fold_with_vC(n_sg, G_sg, grid_vg)
             for G_sg in G_isg])

        res_isg = alpha_isg * folded_isg

        return res_isg.sum(axis=0)

    def calculate_V1p(self, alpha_ig, n_sg, grid_vg, ni_j):
        # Convolve n_sg*alpha_ig and G/v_C

        G_isg = self.get_G_isg(grid_vg, ni_j, len(n_sg))
        folded_isg = np.array(
            [self.fold_with_vC(n_sg * alpha_g,
                               G_isg[i],
                               grid_vg) for i, alpha_g in enumerate(alpha_ig)])
        
        return folded_isg.sum(axis=0)

    def get_dalpha_isg(self, alpha_isg, Z_isg,
                       Z_lower_sg, Z_upper_sg,
                       grid_vg, ni_j, ni_lower,
                       ni_upper, numspins, dZFunc):
        dalpha_isr = np.zeros_like(alpha_isg.reshape(len(alpha_isg),
                                                     len(alpha_isg[0]), -1))

        G_isr = self.get_G_isg(grid_vg,
                               ni_j,
                               numspins).reshape(*dalpha_isr.shape)

        npand = np.logical_and
        npor = np.logical_or
        Z_isr = Z_isg.reshape(len(Z_isg), len(Z_isg[0]), -1)
        Z_lower_sr = Z_lower_sg.reshape(len(Z_lower_sg), -1)
        Z_upper_sr = Z_upper_sg.reshape(*Z_lower_sr.shape)

        def gZ(i, s):
            if i < 0:
                return Z_lower_sr[s]
            elif i >= len(Z_isr):
                return Z_upper_sr[s]
            else:
                return Z_isr[i, s]

        def dZ(i, s):
            return dZFunc(i,
                          s,
                          G_isr,
                          grid_vg, ni_lower, ni_upper, alpha_isg)

        hitit = False
        for i, alpha_sr in enumerate(alpha_isg.reshape(*dalpha_isr.shape)):
            for s, alpha_r in enumerate(alpha_sr):
                if not np.allclose(gZ(i + 1, s), gZ(i, s)):
                    denom0 = gZ(i + 1, s) - gZ(i, s)
                    
                    # At +0 branch if
                    # Z[i] >= -1 and Z[i+1] < -1
                    # OR
                    # Z[i] <= -1 and Z[i+1] > -1
                    factor0_r = npor(npand(gZ(i, s) >= -1, gZ(i + 1, s) < -1),
                                     npand(gZ(i, s) <= -1, gZ(i + 1, s) > -1))
              
                    dalpha_isr[i, s] += factor0_r * (dZ(i + 1, s) / denom0
                                                     - alpha_r / denom0
                                                     * (dZ(i + 1, s)
                                                        - dZ(i, s)))
                    hitit = True

                if not np.allclose(gZ(i, s), gZ(i - 1, s)):
                    # At +1 branch if
                    # Z[i-1] >= -1 and Z[i] < -1
                    # OR
                    # Z[i-1] <= -1 and Z[i] > -1
                    factor1_r = npor(npand(gZ(i - 1, s) >= -1, gZ(i, s) < -1),
                                     npand(gZ(i - 1, s) <= -1, gZ(i, s) > -1))
                    denom1 = gZ(i, s) - gZ(i - 1, s)
                    dalpha_isr[i, s] += factor1_r * (-dZ(i - 1, s) / denom1
                                                     - alpha_r / denom1
                                                     * (dZ(i, s) - dZ(i, s)))
                    hitit = True

        # assert hitit
        return dalpha_isr.reshape(*alpha_isg.shape)
    
    def calculate_V2(self, dalpha_isg, n_sg, grid_vg, ni_j):
        # Convolve n_sg and G/v_C
        # Multiply n_sg to result -> res
        # Convolve res and dalpha/dn

        G_isg = self.get_G_isg(grid_vg, ni_j, len(n_sg))
        folded_isg = self.fold_multiple_with_vC(n_sg, G_isg, grid_vg)
        res_isg = n_sg[np.newaxis, ...] * folded_isg
        
        final_isg = np.array(
            [self.fold(dalpha_sg, res_isg[i])
             for i, dalpha_sg in enumerate(dalpha_isg)])

        return final_isg.sum(axis=0)

    def calculate_sym_pot_correction(self, alpha_isg, n_sg, grid_vg, ni_j):
        # Two corrections: dV1, dV1p
        # dV1: Convolve n_sg*alpha_ig and G/v_C
        G_isg = self.get_G_isg(grid_vg, ni_j, len(n_sg))
        dV1_isg = np.array(
            [self.fold_with_vC(n_sg * alpha_isg[i], G_sg, grid_vg)
             for i, G_sg, in enumerate(G_isg)])
        # dV1p: Convolve n_sg and G/v_c and multiply alpha_ig to result
        dV1p_isg = alpha_isg * np.array(
            [self.fold_with_vC(n_sg, G_sg, grid_vg) for G_sg in G_isg])
        
        return (dV1_isg + dV1p_isg).sum(axis=0)

    def calculate_energy(self, alpha_isg, n_sg, gd, grid_vg, ni_j):
        # Convolve n_sg with G/v_C
        # Multiply alpha_ig*n_sg to result
        # Integrate over space

        G_isg = self.get_G_isg(grid_vg, ni_j, len(n_sg))
        folded_isg = self.fold_multiple_with_vC(n_sg, G_isg, grid_vg)
        integrand_isg = alpha_isg * folded_isg
        
        return integrand_isg.sum(axis=0).sum(axis=1)

    def fold_multiple_with_vC(self, f_sg, F_isg, grid_vg):
        return np.array(
            [self.fold_with_vC(f_sg, F_sg, grid_vg) for F_sg in F_isg])

    def calculate_sym_energy_correction(self,
                                        alpha_ig, n_sg, gd, grid_vg, ni_j):
        # Convolve n_sg*alpha_ig with g/v_C (small g!)
        # Multiply result with n_sg and integrate
        G_isg = self.get_G_isg(grid_vg, ni_j, len(n_sg))
        
        r_isg = self.fold_multiple_with_vC(n_sg, G_isg, grid_vg)
        
        return r_isg.sum(axis=0).sum(axis=0)

    def calculate_energy_correction_valence_mode(self, nae_sg, n_sg):
        # Calculate ELDA for ae dens and subtract ELDA for other dens
        eae_g = np.zeros(nae_sg.shape[1:])
        e_g = np.zeros_like(eae_g)
        self.lda_kernel.calculate(eae_g, nae_sg, np.zeros_like(nae_sg))
        self.lda_kernel.calculate(e_g, n_sg, np.zeros_like(n_sg))

        return eae_g - e_g
