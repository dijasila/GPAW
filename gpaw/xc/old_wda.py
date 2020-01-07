from gpaw.xc.functional import XCFunctional
from gpaw.xc.lda import PurePythonLDAKernel
from gpaw.utilities.tools import construct_reciprocal
from gpaw.blacs import BlacsGrid, BlacsDescriptor, Redistributor
import numpy as np
from gpaw import mpi
from ase.parallel import parprint

##TODOS
# Calculate G(k) on radial grid and implement using this instead
# Fix indicators/ni-interpolation to handle Z_is not crossing -1

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
        print("LDA CORRELATION DISABLED", flush=True)
        
    def initialize(self, density, hamiltonian, wfs, occupations):
        self.wfs = wfs
        self.density = density
        pass

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        if len(n_sg) != 1:
            raise NotImplementedError
        spin = 0
        parprint("Started", flush=True)
        # Get full grid gd
        gd1 = gd.new_descriptor(comm=mpi.serial_comm)
        self.gd = gd1
        grid = gd1.get_grid_point_coordinates()
        
        # Get density: AE, Corrected or Valence
        wn_sg = self.get_working_density(n_sg, gd)
        parprint("Got working density", flush=True)
        # Get ni_grid
        ni_grid, ni_lower, ni_upper, numni, fulln_i = self.get_ni_grid(mpi.rank,
                                                              mpi.size, wn_sg)
        self.fulln_i = fulln_i
        parprint("Got ni grid", flush=True)
        # Get Zs
        Z_isg, Z_lower_sg, Z_upper_sg = self.get_Zs(wn_sg,
                                                    ni_grid,
                                                    ni_lower,
                                                    ni_upper,
                                                    grid, spin,
                                                    gd1, mpi.rank,
                                                    mpi.size)
        parprint("Calculated Z_isg", flush=True)
        alpha_isg = self.get_alphas(Z_isg, len(ni_grid), numni, mpi.rank, mpi.size)

        # dCalculate Vs
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
            parprint(f"eWDA_g shape: {eWDA_g.shape}")
            parprint(f"Grid shape: {grid.shape}")
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
        res = correct_density(n_sg, gd, self.wfs.setups, self.wfs.spos_ac)
        return res

    def get_valence_density(self, n_sg):
        raise NotImplementedError

    def get_ni_grid(self, rank, size, n_sg):
        from gpaw.xc.WDAUtils import get_ni_grid_w_min
        pts_per_rank = self.get_pts_per_rank()
        if not np.allclose(np.max(n_sg), 0):
            maxval = max(np.max(n_sg), 100 * np.mean(n_sg))
        else:
            maxval = 10
        #maxval = np.mean(n_sg)*3
        maxval = self.gd.integrate(n_sg)[0]*2 #1.0
        minval = 0.5 * np.min(n_sg)
        
        # return np.arange(minval, maxval, pts_per_rank * size)
        return get_ni_grid_w_min(rank, size, minval, maxval, pts_per_rank, return_full_size=True)

    def get_pts_per_rank(self):
        # return 20
        return min(10, 100 // mpi.size + 1)

    def get_pairdist_g(self, grid, ni, spin):
        from ase.units import Bohr
        grid_distances = np.linalg.norm(grid, axis=0) * Bohr
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

    def build_indicators(self, ni_grid, ni_lower, ni_upper, rank, size):
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
            inter = interp1d(val_grid, targets, kind="linear",
                            bounds_error=False, fill_value=0)
            return inter
            # def func(vals_g):
            #     applied = inter(vals_g)
        ni_range = range(low_off, len(augmented_ni_grid) - upp_off)

        simple_indicators = [build_an_indicator(j, augmented_ni_grid) for j in ni_range]
        # adv_indicators = []
        def build_adv(j, indicator, augmented_ni_grid):
            def f(vals):
                less_thans = vals < np.min(augmented_ni_grid)
                great_thans = vals > np.max(augmented_ni_grid)
                applied = indicator(vals)
                if j == 0 and rank == 0:
                    applied[less_thans] = 1
                elif j == len(augmented_ni_grid) - 1 and rank == size - 1:
                    applied[great_thans] = 1
                return applied
            return f

        adv_indicators = [build_adv(j, ind, augmented_ni_grid) for j, ind in enumerate(simple_indicators)]

        return adv_indicators
        #return simple_indicators

        #return [build_an_indicator(j, augmented_ni_grid) for j in ni_range]

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
                        upper_ni, grid, spin, gd,
                        rank, size):
        # Get parametrized pair distribution functions
        # augmented_ni_grid = np.hstack([[lower], ni_grid, [upper]])
        
        # pairdist_ig
        # = np.array([self.get_pairdi
        # st(grid, ni, spin) for ni in augmented_ni_grid])
        # Z_isg = np.zeros(ni_grid.shape + n_sg.shape)

        # pairdist_g = self.get_pairdist_g(grid, lower_ni, spin)
        # Z_lower_sg = self.fold_w_pair(n_sg, np.array([pairdist_g]) - 1)
        Z_lower_sg = self.fold_w_G(n_sg, [lower_ni])[0]
        
        # pairdist_g = self.get_pairdist_g(grid, upper_ni, spin)
        # Z_upper_sg = self.fold_w_pair(n_sg, np.array([pairdist_g]) - 1)
        Z_upper_sg = self.fold_w_G(n_sg, [upper_ni])[0]

        Z_isg = self.fold_w_G(n_sg, ni_grid)
        # for i, n in enumerate(ni_grid):
        #     pairdist_g = self.get_pairdist_g(grid, n, spin)
        #     Z_isg2[i] = self.fold_w_pair(n_sg, np.array([pairdist_g]) - 1)
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

    def fold_w_G(self, f_sg, ni_j):
        K_G = self._get_K_G(self.gd)
        G_iG = self.calc_G_ik(ni_j, K_G)

        F = self.fftn(f_sg, axes=(1, 2, 3))
        FG_isG = F[np.newaxis, :, :, :, :] * G_iG[:, np.newaxis, :, :, :]

        res_isg = self.ifftn(FG_isG, axes=(2, 3, 4))
        assert np.allclose(res_isg, res_isg.real)
        return res_isg.real

    def fold_w_g(self, f_isg, ni_j):
        assert len(f_isg) == len(ni_j), "{}, {}".format(len(f_isg), len(ni_j))
        K_G = self._get_K_G(self.gd)
        assert (K_G >= 0).all()
        zero_index = np.argmin(K_G)
        G_iG = self.calc_G_ik(ni_j, K_G)
        # How do we handle delta function for g(K) at K = 0?
        if f_isg.ndim == 4:
            f_isg = f_isg[np.newaxis, ...]

        F_isG = self.fftn(f_isg, axes=(2, 3, 4))
        FG_isG = F_isG * G_iG[:, np.newaxis, :, :, :]

        res_isg = self.ifftn(FG_isG, axes=(2, 3, 4)) + F_isG[:, :, zero_index]
        assert np.allclose(res_isg, res_isg.real)
        return res_isg.real

    def fold_w_pair(self, f_sg, G_sg):
        assert np.allclose(f_sg, f_sg.real)
        assert np.allclose(G_sg, G_sg.real)
        assert f_sg.shape == G_sg.shape, f"Shape of f_sg: {f_sg.shape}, shape of G_sg: {G_sg.shape}"
        assert f_sg.ndim == 4


        N = np.array(f_sg[0].shape).prod()
        F = self.fftn(f_sg, axes=(1, 2, 3))
        G = self.fftn(G_sg, axes=(1, 2, 3)) * N # * self.gd.dv
        
        res = self.ifftn(F * G, axes=(1, 2, 3))
        assert np.allclose(res, res.real), "MEAN ABS DIFF: {}, MEAN ABS: {}, N: {}".format(np.mean(np.abs(res - res.real)), np.mean(np.abs(res.real)), N)

        return res.real

    def fold_w_pair2(self, f_sg, G_sG):
        assert np.allclose(f_sg, f_sg.real)
        assert np.allclose(G_sg, G_sg.real)
        assert f_sg.shape == G_sg.shape
        assert f_sg.ndim == 4


        N = np.array(f_sg[0].shape).prod()
        F = self.fftn(f_sg, axes=(1, 2, 3))
        
        res = self.ifftn(F * G_sG, axes=(1, 2, 3))
        assert np.allclose(res, res.real), "MEAN ABS DIFF: {}, MEAN ABS: {}, N: {}".format(np.mean(np.abs(res - res.real)), np.mean(np.abs(res.real)), N)

        return res.real

    def get_G_sG(inter_iG, i, gd, spin):
        K_G = self._get_K_G(gd)

        G_G = inter_iG(K_G, i)

        

    def symmetricmode_Zs(self, n_sg, ni_grid,
                         lower_ni, upper_ni,
                         grid, spin, gd,
                         rank, size):
        from time import time
        augmented_ni_grid, _, _ = self.get_augmented_ni_grid(ni_grid,
                                                             lower_ni,
                                                             upper_ni)

        ind_i = self.build_indicators(augmented_ni_grid, lower_ni, upper_ni, rank, size)
        ind2_i = self.build_indicators(ni_grid, lower_ni, upper_ni, rank, size)
        # def getZ(indicator_sg, ):
        #    return self.fold_w_G(n_sg, indicator_sg) + self.fold_w_g(n_sg * indicator_sg
        ind_sg = np.array([ind_i[0](n_g) for n_g in n_sg])
        # print("lower")
        Z_lower_sg = self.fold_w_G(n_sg, [lower_ni])[0] \
                     + self.fold_w_g(n_sg * ind_sg, [lower_ni])[0]

        ind_isg = np.array([[ind(n_g) for n_g in n_sg] for ind in ind2_i])
        # print("all")
        Z_isg = self.fold_w_G(n_sg, ni_grid) \
                + self.fold_w_g(n_sg[np.newaxis, ...] * ind_isg, 
                                ni_grid)
        ind_sg = np.array([ind_i[-1](n_g) for n_g in n_sg])
        # print("upper")
        Z_upper_sg = self.fold_w_G(n_sg, [upper_ni])[0] \
                     + self.fold_w_g(n_sg * ind_sg, [upper_ni])[0]
        

        return Z_isg, Z_lower_sg, Z_upper_sg

        # pairdist_sg = np.array([self.get_pairdist_g(grid, lower_ni, spin)])

        # ind_sg = np.array([ind_i[0](n_sg[0])])
        
        # def getZ(pairdist, ind):
        #     return self.fold_w_pair(n_sg, pairdist - 1) + self.fold_w_pair(n_sg * ind,
        #                                                      pairdist)

        # Z_lower_sg = getZ(pairdist_sg, ind_sg)
        # Z_isg = []
        # for i, n in enumerate(ni_grid):
        #     pd = np.array([self.get_pairdist_g(grid, n, spin)])
        #     ind = ind_i[i](n_sg[0])
        #     Z = getZ(pd, ind)
        #     Z_isg.append(Z)

        # assert len(Z_isg) == len(ni_grid)
        # pd = np.array([self.get_pairdist_g(grid, upper_ni, spin)])
        # ind = ind_i[-1](n_sg[0])
        # Z_upper_sg = getZ(pd, ind)

        # return np.array(Z_isg), Z_lower_sg, Z_upper_sg

    def interpolate_this(self, val_i, lower, upper, target):
        inter_i = np.zeros_like(val_i)

        # if (val_i <= target).all():
        #     max_pos = np.unravel_index(np.argmax(val_i), val_i.shape)
        #     inter_i[max_pos] = 1
        #     assert np.allclose(inter_i.sum(), 1)
        #     return inter_i
        # elif (val_i >= target).all():
        #     min_pos = np.unravel_index(np.argmin(val_i), val_i.shape)
        #     inter_i[min_pos] = 1
        #     assert np.allclose(inter_i.sum(), 1)
        #     return inter_i
        
        if np.allclose(val_i[0], lower) and np.allclose(val_i[-1], upper):
            
            for i, val in enumerate(val_i):
                if i == 0:
                    continue
                
                if (val >= target and val_i[i - 1] <= target) or (val <= target and val_i[i - 1] >= target):
                    forward_weight = (target - val_i[i - 1]) / (val - val_i[i - 1])
                    backward_weight = (val - target) / (val - val_i[i - 1])
                    inter_i[i] = forward_weight
                    inter_i[i - 1] = backward_weight
                    return inter_i

        elif np.allclose(val_i[0], lower):

            found = False
            for i, val in enumerate(val_i):
                if i == 0:
                    continue

                if (val >= target and val_i[i - 1] <= target) or (val <= target and val_i[i - 1] >= target):
                    forward_weight = (target - val_i[i - 1]) / (val - val_i[i - 1])
                    backward_weight = (val - target) / (val - val_i[i - 1])
                    inter_i[i] = forward_weight
                    inter_i[i - 1] = backward_weight
                    found = True
                    break

            if ((upper >= target and val_i[-1] <= target) or (upper <= target and val_i[-1] >= target)) and not found:
                forward_weight = (target - val_i[-1]) / (upper - val_i[i - 1])
                backward_weight = (upper - target) / (upper - val_i[i - 1])
                inter_i[-1] = backward_weight
                
            return inter_i

        elif np.allclose(val_i[0], upper):
            found = False
            if (val_i[0] >= target and lower <= target) or (val_i[0] <= target and lower >= target):
                forward_weight = (target - lower) / (val_i[0] - lower)
                inter_i[0] = forward_weight
                found = True
                return inter_i

            for i, val in enumerate(val_i):
                if i == 0:
                    continue
                
                if (val >= target and val_i[i - 1] <= target) or (val <= target and val_i[i - 1] >= target):
                    forward_weight = (target - val_i[i - 1]) / (val - val_i[i - 1])
                    backward_weight = (val - target) / (val - val_i[i - 1])
                    inter_i[i] = forward_weight
                    inter_i[i - 1] = backward_weight
                    return inter_i

        else:
            found = False
            if (val_i[0] >= target and lower <= target) or (val_i[0] <= target and lower >= target):
                forward_weight = (target - lower) / (val_i[0] - lower)
                inter_i[0] = forward_weight
                found = True
                return inter_i

            for i, val in enumerate(val_i):
                if i == 0:
                    continue
                
                if (val >= target and val_i[i - 1] <= target) or (val <= target and val_i[i - 1] >= target):
                    forward_weight = (target - val_i[i - 1]) / (val - val_i[i - 1])
                    backward_weight = (val - target) / (val - val_i[i - 1])
                    inter_i[i] = forward_weight
                    inter_i[i - 1] = backward_weight
                    return inter_i

            if (upper >= target and val_i[-1] <= target) or (upper <= target and val_i[-1] >= target):
                backward_weight = (upper - target) / (upper - val_i[i - 1])
                inter_i[-1] = backward_weight
                return inter_i
                    

        return inter_i

        # if True:
        #     print("HEJ")

        # elif np.allclose(val_i[0], lower):
        #     aug_i = np.array(list(val_i) + [upper])
        # elif np.allclose(val_i[-1], upper):
        #     aug_i = np.array([lower] + list(val_i))
        # else:
        #     aug_i = np.array([lower] + list(val_i) + [upper])

        # # count = 0


        # count = 0
        # for i, val in enumerate(val_i):
        #     if i == 0:
        #         nextv = val_i[i + 1]
        #         prev = lower
        #     elif i == len(val_i) - 1:
        #         nextv = upper
        #         prev = val_i[i - 1]
        #     else:
        #         nextv = val_i[i + 1]
        #         prev = val_i[i - 1]

        #     if ((val <= target
        #         and nextv >= target) or (val >= target and nextv <= target)) and (not np.allclose(val, nextv)):
        #         inter_i[i] = (nextv - target) / (nextv - val)
        #         assert inter_i[i] >= 0
        #         assert inter_i[i] <= 1
        #         count += 1
        #     elif ((val <= target and
        #           prev >= target) or (val >= target and prev <= target)) and (not np.allclose(val, prev)):
        #         inter_i[i] = (prev - target) / (prev - val)
        #         assert inter_i[i] >= 0
        #         assert inter_i[i] <= 1
        #         count += 1
        #     if count >= 2:
        #         break

        # assert np.allclose(inter_i.sum(), 1), "Actual sum: {}\nVal_i: {}\nLower: {}\nUpper: {}\nTarget: {}".format(inter_i.sum(), val_i, lower, upper, target)
        # assert count == 2

        # if count != 2:
        #     print("Count is:", count)
        #     import matplotlib.pyplot as plt
        #     plt.plot(val_i, '-', marker=".")
        #     plt.show()

        # return inter_i
        
    def get_alphas(self, Z_isg, myni, numni, rank, size):
        nZ_ix, mynx, nx = self.redistribute_i_to_g(Z_isg, myni, numni, rank, size)

        alpha_ix = self._get_alphas(nZ_ix)
        alpha_isg = self.redistribute_g_to_i(alpha_ix,
                                             np.zeros_like(Z_isg),
                                             myni, numni,
                                             mynx, nx,
                                             rank, size)
    
        return alpha_isg


    def _get_alphas(self, Z_ix):
        # Reimplement this to take Z_ix as input
        # Rewrite tests...
        alpha_ix = np.zeros_like(Z_ix)
        for ix, Z_i in enumerate(Z_ix.T):
            for i, Z in enumerate(Z_i):
                if i == len(Z_i) - 1:
                    icross = None
                    continue
                else:
                    if Z <= -1 and Z_i[i + 1] > -1:
                        icross = i
                        break
                    elif Z > -1 and Z_i[i + 1] <= -1:
                        icross = i
                        break

            if icross is None:
                if (Z_i <= -1).all():
                    alpha_ix[np.argmax(Z_i), ix] = 1
                else:
                    assert (Z_i > -1).all()
                    alpha_ix[np.argmin(Z_i), ix] = 1
            else:
                valleft = Z_i[icross]
                valright = Z_i[icross + 1]
                alpha_ix[icross, ix] = (valright - (-1)) / (valright - valleft)
                alpha_ix[icross + 1, ix] = ((-1) - valleft) / (valright - valleft)
                        
        return alpha_ix


    def DEPRECATEDget_alphas(self, Z_isg, Z_lower_sg, Z_upper_sg):
        alpha_isg = np.zeros_like(Z_isg)

        for iz, Z_yxsi in enumerate(Z_isg.T):
            for iy, Z_xsi in enumerate(Z_yxsi):
                for ix, Z_si in enumerate(Z_xsi):
                    for inds, Z_i in enumerate(Z_si):
                        assert Z_i.ndim == 1
                        if not ((Z_i >= -1).any() and (Z_i <= -1).any()):
                            mindex = np.argmin(Z_i + 1)
                            alpha_isg[mindex, inds, ix, iy, iz] = 1
                        else:
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

    def standard_dalpha_isgg(self, alpha_isg, Z_isg,
                             Z_lower_sg, Z_upper_sg,
                             grid_vg, ni_j, ni_lower,
                             ni_upper, numspins,
                             numni, rank, size):
        # return dalpha(r') / dn(r) 

        # if alpha_i = 0, then dalpha = 0

        # Switch to parallelization over spin \otimes position because it is much
        # simpler to calculate

        alpha_ix, mynx, nx = self.redistribute_i_to_g(alpha_isg, len(ni_j), numni, rank, size)

        Z_ix, mynx2, nx2 = self.redistribute_i_to_g(Z_isg, len(ni_j), numni, rank, size)
        assert mynx == mynx2
        assert nx == nx2
        qw =  lpha_ix[np.logical_not(np.isclose(alpha_ix, 0))].sum()
        assert qw == 1 or qw == 2

        G_i
        dZ_ixg = 
        
    
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
        assert len(integrand_isg.shape) == 5
        return integrand_isg.sum(axis=0).sum(axis=0)

    def fold_multiple_with_vC(self, f_sg, F_isg, grid_vg):
        return np.array(
            [self.fold_with_vC(f_sg, F_sg, grid_vg) for F_sg in F_isg])

    def calculate_sym_energy_correction(self,
                                        alpha_ig, n_sg, gd, grid_vg, ni_j):
        # Convolve n_sg*alpha_ig with g/v_C (small g!)
        # Multiply result with n_sg and integrate
        G_isg = self.get_G_isg(grid_vg, ni_j, len(n_sg))
        
        r_isg = self.fold_multiple_with_vC(n_sg, G_isg, grid_vg)
        
        assert len(r_isg.shape) == 5, f"Shape of r_isg: {r_isg.shape}"
        return r_isg.sum(axis=0).sum(axis=0)

    def calculate_energy_correction_valence_mode(self, nae_sg, n_sg):
        # Calculate ELDA for ae dens and subtract ELDA for other dens
        eae_g = np.zeros(nae_sg.shape[1:])
        e_g = np.zeros_like(eae_g)
        self.lda_kernel.calculate(eae_g, nae_sg, np.zeros_like(nae_sg))
        self.lda_kernel.calculate(e_g, n_sg, np.zeros_like(n_sg))

        return eae_g - e_g

    def construct_G_splines(self, ni_grid):
        r_j = np.arange(0.01, 200, 0.01)
        dr_j = r_j[1:] - r_j[:-1]
        
        G_ir = self.get_G_ir(ni_grid, r_j, 0)

        ks = np.linspace(0.01, 200, 300, endpoint=False)
    
        G_iG = np.zeros((len(ni_grid), len(ks)))

        na = np.newaxis
        for ik, k in enumerate(ks):
            int_ir = G_ir * 4 * np.pi * np.sin(k * r_j[na, :]) / k * r_j[na, :]
            int_ir = (int_ir[:, 1:] + int_ir[:, :-1]) / 2
            val_i = np.sum(int_ir * dr_j[na, :], axis=1)
            G_iG[:, ik] = val_i
            
        from scipy.interpolate import spline

        inter_iG = lambda i, K_G: spline(ks, G_iG[i, :], K_G)

        return inter_iG

        
    def DEPRECATEDget_G_ir(self, ni_grid, r_j, spin):
        from scipy.special import gamma

        def get_G_r(ni):
            exc = self.get_lda_xc(ni, spin)
            lambd = -3 * gamma(3 / 5) / (2 * gamma(2 / 5) * exc)
            C = -3 / (4 * np.pi * gamma(2 / 5) * ni * lambd**3)
        
            G_r = C * (1 - np.exp(-(lambd / r_j)**5))
            return G_r

        G_ir = np.array([get_G_r(ni) for ni in ni_grid])

        return G_ir

    def redistribute_i_to_g(self, in_isg, myni, ni, rank, size): # out_isg, 
        # Take an array that is distributed over i (WD index)
        # and return an array distributed over g (position)
        comm = mpi.world


        in_ix = in_isg.reshape(myni, -1)
        nx = in_ix.shape[1]
        mynx = (nx + size - 1) // size

        # out_ix = out_isg.reshape(ni, -1)
        out_ix = np.zeros((ni, mynx))

        bg1 = BlacsGrid(comm, size, 1)
        bg2 = BlacsGrid(comm, 1, size)
        md1 = BlacsDescriptor(bg1, ni, nx, myni, nx)
        md2 = BlacsDescriptor(bg2, ni, nx, ni, mynx)

        r = Redistributor(comm, md1, md2)

        # r.redistribute(in_isg.reshape(md1.shape),
        #               out_isg.reshape(md2.shape))
        r.redistribute(in_isg.reshape(md1.shape),
                       out_ix)

        return out_ix, mynx, nx

    def redistribute_g_to_i(self, in_ix, out_isg, myni, ni, mynx, nx,
                            rank, size):
        # Take an array that is distributed over g (position)
        # and return an array distributed over i (WD index)
        comm = mpi.world

        # in_ix = in_isg.reshape(ni, -1)
        out_ix = out_isg.reshape(myni, -1)

        bg1 = BlacsGrid(comm, 1, size)
        bg2 = BlacsGrid(comm, size, 1)
        md1 = BlacsDescriptor(bg1, ni, nx, ni, mynx)
        md2 = BlacsDescriptor(bg2, ni, nx, myni, nx)

        r = Redistributor(comm, md1, md2)
        
        r.redistribute(in_ix,
                       out_isg.reshape(md2.shape))
        
        return out_isg


    def get_G_r(self, ni, r_g, spin=0):
        from ase.units import Bohr
        from scipy.special import gamma
        exc = self.get_lda_xc(ni, spin) # What should spin be here?
        assert exc <= 0
        if np.allclose(exc, 0):
            g = np.zeros_like(r_g)
            nni = 0.0000001
            # nni = 0.001
            exc = self.get_lda_xc(nni, spin)
            lambd = -3 * gamma(3/5) / (2 * gamma(2 / 5) * exc)
            C = -3 / (4 * np.pi * gamma(2 / 5) * nni * lambd**3)
            g[:] = C
            g = g
            return g

        lambd = - 3 * gamma(3 / 5) / (2 * gamma(2 / 5) * exc)

        C = -3 / (4 * np.pi * gamma(2 / 5) * ni * lambd**3) # * 1000000
        g = np.zeros_like(r_g)
        exp_val = -(lambd / (r_g[r_g > 0]))**5
        g[r_g > 0] = C * (1 - np.exp(exp_val))
        g[np.isclose(r_g, 0)] = C
        
        return g
        

    def calc_G_ik(self, ni_j, k_k):
        dr = 0.001
        r_g = np.arange(dr / 2, 1, dr) # Units: Bohr??
        G_ig = np.array([self.get_G_r(ni, r_g) for ni in ni_j])
        na = np.newaxis
        k_k[np.isclose(k_k, 0)] = 1 # How to handle |k| = 0?
        integrand_ikg = 4 * np.pi * r_g[na, na, na, na, :] / k_k[na, :, :, :, na] * np.sin(k_k[na, :, :, :, na] * r_g[na, na, na, na, :]) * G_ig[:, na, na, na, :]

        G_ik = np.sum(integrand_ikg * dr, axis=-1)


        return G_ik
