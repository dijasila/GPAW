from testframework import BaseTester
import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw import PW
from gpaw import mpi


atoms = Atoms("H2", positions=[[0,0,0], [0,0,2]], cell=5*np.identity(3))
calc = GPAW(mode=PW(200), xc="WDA_standard", txt=None)
calc.initialize(atoms=atoms)
calc.set_positions(atoms)
xc = calc.hamiltonian.xc

assert xc is not None





class Tester(BaseTester):
    def __init__(self):
        self.gd = xc.gd.new_descriptor(comm=mpi.serial_comm)
        self.grid = self.gd.get_grid_point_coordinates()
    

    def get_a_density(self):
        gd = xc.gd.new_descriptor(comm=mpi.serial_comm)
        grid = gd.get_grid_point_coordinates()
        dens = np.zeros(grid.shape[1:])
        densf = np.fft.fftn(dens)
        densf = np.random.rand(*densf.shape) + 0.1
        densf[0,0,0] = densf[0,0,0] + 1.0

        dens = np.fft.ifftn(densf).real
        dens = dens + np.min(dens)
        dens[dens < 1e-7] = 1e-8
        norm = self.gd.integrate(dens)

        nelec = 5
        dens = dens / norm * nelec
        
        assert np.allclose(self.gd.integrate(dens), nelec)
        
        res = np.array([dens])
        #res[res < 1e-7] = 1e-8
        
        assert (res >= 0).all()
        assert res.ndim == 4
        assert not np.allclose(res, 0)
        return res

    def get_initial_stuff(self):
        n_sg = self.get_a_density()
        ni_j, nilower, niupper = xc.get_ni_grid(0, 1, n_sg)
        grid = self.gd.get_grid_point_coordinates()
        Z_ig, Zlower_g, Zupper_g = xc.get_Zs(n_sg, ni_j, nilower, niupper, grid, 0, self.gd)
        alpha_ig = xc.get_alphas(Z_ig, Zlower_g, Zupper_g)
        
        return alpha_ig, Z_ig, Zlower_g, Zupper_g, ni_j, nilower, niupper, n_sg

    def get_a_sym_density(self, dir=0):
        n_sg = self.get_a_density()
        if dir == 0:
            n_sg = (n_sg[:, :, :, :] + n_sg[:, ::-1, :, :]) / 2
        elif dir == 1:
            n_sg = (n_sg[:, :, :, :] + n_sg[:, :, ::-1, :]) / 2
        elif dir == 2:
            n_sg = (n_sg[:, :, :, :] + n_sg[:, :, :, ::-1]) / 2
        else:
            raise ValueError("Direction '{}' not recognized. Must be 0, 1, or 2.".format(dir))

        return n_sg

    def get_grid(self):
        return xc.gd.get_grid_point_coordinates()

    def test_01_getstandardZis(self):
        n_sg = self.get_a_density()
        ni_grid, lower_ni, upper_ni = xc.get_ni_grid(0, 1, n_sg)
        grid = self.get_grid()
        spin = 0
        gd = xc.gd
        Z_i = xc.standardmode_Zs(n_sg, ni_grid, lower_ni, upper_ni, grid, spin, gd)

    def test_02_getLDAxc(self):
        return "skipped"
        # Test for spin = 0 and spin = 1
        n = np.random.rand() * 100
        e_x = - 3 / 4 * (3 / np.pi)**(1/3) * n**(4/3)
        from gpaw.xc.lda import lda_c, lda_x
        exp_x = np.array([0.])
        lda_x(0, exp_x, n, np.array([0.]))
        e_c = np.array([0.])
        lda_c(0, e_c, n, np.array([0.]), 0)
        
        actual_xc = xc.get_lda_xc(n, 0) * n

        assert np.allclose(e_x + e_c, actual_xc), "Error in xc: {}".format(np.max(np.abs(e_x+e_c - actual_xc)))


    def test_03_indicators_sumtoone(self):
        ni_grid, ni_lower, ni_upper = xc.get_ni_grid(0, 1, self.get_a_density())
        ind_i = xc.build_indicators(ni_grid, ni_lower, ni_upper)

        num_pts = np.random.randint(1000) + 100
        fine_grid = np.linspace(min(ni_grid), max(ni_grid), num_pts)
        
        evaluated_inds = np.array([ind(fine_grid) for ind in ind_i])
        assert np.allclose(evaluated_inds.sum(axis=0), 1)

    def test_04_indicators_greaterthanzero(self):
        ni_grid, ni_lower, ni_upper = xc.get_ni_grid(0, 1, self.get_a_density())
        ind_i = xc.build_indicators(ni_grid, ni_lower, ni_upper)
        
        num_pts = np.random.randint(1000) + 100
        fine_grid = np.linspace(min(ni_grid), max(ni_grid), num_pts)
        eval_inds = np.array([ind(fine_grid) for ind in ind_i])
        assert (eval_inds >= 0).all()

    def test_05_indicators_oneattarget(self):
        ni_grid, ni_lower, ni_upper = xc.get_ni_grid(0, 1, self.get_a_density())
        ind_i = xc.build_indicators(ni_grid, ni_lower, ni_upper)

        num_pts = np.random.randint(1000) + 100
        
        eval_inds = np.array([ind(ni_grid) for ind in ind_i])

        assert np.allclose(eval_inds, np.eye(len(ni_grid)))
        

    def test_06_getsymmetricZis(self):
        n_sg = self.get_a_density()
        ni_grid, lower_ni, upper_ni = xc.get_ni_grid(0, 1, n_sg)
        grid = self.get_grid() 
        spin = 0
        gd = xc.gd
        Z_i = xc.symmetricmode_Zs(n_sg, ni_grid, lower_ni, upper_ni, grid, spin, gd)

    def good_num(self, num):
        a = num != np.inf
        b = num != -np.inf
        c = num != np.nan
        return np.logical_and(a, np.logical_and(b, c))

    def test_07_standardZderiv(self):
        ni = np.random.rand() * 2.0
        grid = self.get_grid()
        spin = 0

        Zderiv = xc.standardmode_Z_derivative(grid, ni, spin)
        
        assert self.good_num(Zderiv).all()

    def test_08_symmetricZderiv(self):
        return "skipped"
        raise NotImplementedError

    def execute_test_for_random_mpiworld(self, test_fct, global_test=None):
        mpi_size = np.random.randint(20) + 2
        global_data = []
        for rank in range(mpi_size):
            global_data.append(test_fct(rank, mpi_size))
        if global_test is not None:
            global_test(mpi_size, global_data)

    def test_09_indicators_parallel(self):
        n_sg = self.get_a_density()
        def local_test_fct(rank, size):
            ni_grid, lower_ni, upper_ni = xc.get_ni_grid(rank, size, n_sg)
            ind_i = xc.build_indicators(ni_grid, lower_ni, upper_ni)
            ind_ig = np.array([ind(n_sg[0]) for ind in ind_i])
            assert (ind_ig >= 0).all()
            return ind_ig
        
        def global_test(size, global_data):
            ind_sum_rg = np.array([g.sum(axis=0) for g in global_data])
            assert np.allclose(ind_sum_rg.sum(axis=0), 1, atol=1e-3, rtol=1), "MEAN: {}, STD: {}".format(ind_sum_rg.sum(axis=0).mean(), ind_sum_rg.sum(axis=0).std())
            
        self.execute_test_for_random_mpiworld(local_test_fct, global_test)
            
    def test_10_indicators_parallel2(self):
        n_sg = self.get_a_density()
        num_pts = len(n_sg.reshape(-1))
        n_sg = np.array(np.linspace(0, 1, num_pts)).reshape(*n_sg.shape)
        def local_test_fct(rank, size):
            ni_grid, lower_ni, upper_ni = xc.get_ni_grid(rank, size, n_sg)
            ind_i = xc.build_indicators(ni_grid, lower_ni, upper_ni)
            ind_ig = np.array([ind(n_sg[0]) for ind in ind_i])
            assert (ind_ig >= 0).all()
            return ind_ig
        
        def global_test(size, global_data):
            ind_sum_rg = np.array([g.sum(axis=0) for g in global_data])
            assert np.allclose(ind_sum_rg.sum(axis=0), 1)
            
        self.execute_test_for_random_mpiworld(local_test_fct, global_test)

        # Need some tests that fourier transforms + folding works as expected
        # 1. xc.fftn(xc.ifftn) = 1, xc.ifftn(xc.fftn) = 1.
        # 2. Fold any function with a weight function with w(k=0) = 1, get function with same norm
        # 3. 


    def test_11_fftnifftn_isid(self):
        n_sg = self.get_a_density()
        assert np.allclose(xc.fftn(xc.ifftn(n_sg)), n_sg)
        assert np.allclose(xc.ifftn(xc.fftn(n_sg)), n_sg)


        n2_sg = np.ones(n_sg.shape)

        fn2_sg = xc.ifftn(n2_sg)

        assert np.allclose(fn2_sg[0,0,0,0], np.sum(n2_sg)), fn2_sg[0,0,0,0]

    def test_12_fftfold_normunchanged(self):
        n_sg = self.get_a_density()

        norm = xc.gd.integrate(n_sg)
        
        nnorm = np.random.rand() * 20
        n_sg = n_sg * nnorm / norm

        assert np.allclose(xc.gd.integrate(n_sg), nnorm)

        w_sg = np.random.rand(*n_sg.shape)

        w_sG = xc.fftn(w_sg)
        N = np.array(n_sg.shape[1:]).prod()
        w_sG[:, 0, 0, 0] = 1 / N

        w_sg = xc.ifftn(w_sG)

        nn_sg = xc.fold_w_pair(w_sg, n_sg)

        norm1 = xc.gd.integrate(n_sg)
        norm2 = xc.gd.integrate(nn_sg)
        assert np.allclose(norm1, norm2), "Norm before fold: {}, norm after fold: {}".format(norm1, norm2)
        
    def test_13_standardZs_lessgreatm1(self):
        # Test that the Zs have values that are both greater than
        # and less than -1.
        # -1 is the target values
        mag = np.random.rand()*100
        n_sg = self.get_a_density() / 5
        while np.allclose(np.max(n_sg), 0):
            mag = np.random.rand()*100
            n_sg = self.get_a_density()*mag
        
        ni_grid, lower_ni, upper_ni = xc.get_ni_grid(0, 1, n_sg)
        grid = self.get_grid()
        Z_i, _, _ = xc.standardmode_Zs(n_sg, ni_grid, lower_ni, upper_ni, grid, 0, xc.gd)

        Z_i = Z_i.reshape(len(Z_i), -1)
       
        countNoGood = np.array([not (Z >= -1).any() for Z in Z_i.T]).sum()
        print("No good count: {}. Total count: {}".format(countNoGood, len(Z_i[0])))
        # con = 0
        # for index, vals in enumerate(Z_i.T):
        #     if not (vals >= -1).any():
        #         con += 1
        #         #print("BAD INDEX: ", index)
        # print("MAN COUNT:", con)
        assert (np.array([(Z >= -1).any() for Z in Z_i.T])).all(), "Z_0 mean: {}, Z_last mean: {}".format(Z_i[0].mean(), Z_i[-1].mean())
        assert (np.array([(Z <= -1).any() for Z in Z_i.T])).all()

    def test_14_standardZs_lessgreatm1_vsmallmag(self):
        # Test that the Zs have values that are both greater than
        # and less than -1.
        # -1 is the target values
        mag = np.random.rand()*0.001
        n_sg = self.get_a_density()*mag
        while np.allclose(np.max(n_sg), 0):
            mag = np.random.rand()*100
            n_sg = self.get_a_density()*mag
        
        ni_grid, lower_ni, upper_ni = xc.get_ni_grid(0, 1, n_sg)
        grid = self.get_grid()
        Z_i, _, _ = xc.standardmode_Zs(n_sg, ni_grid, lower_ni, upper_ni, grid, 0, xc.gd)


        Z_i = Z_i.reshape(len(Z_i), -1)

        assert (np.array([(Z >= -1).any() for Z in Z_i.T])).all()
        assert (np.array([(Z <= -1).any() for Z in Z_i.T])).all()

    def test_15_standardZs_lessgreatm1_numelec(self):
        # Test that the Zs have values that are both greater than
        # and less than -1.
        # -1 is the target values
        num_e = np.random.randint(50, 100) + 1
        n_sg = self.get_a_density()
        n_sg = n_sg * num_e / xc.gd.integrate(n_sg)
        
        ni_grid, lower_ni, upper_ni = xc.get_ni_grid(0, 1, n_sg)
        grid = self.get_grid()
        Z_i, _, _ = xc.standardmode_Zs(n_sg, ni_grid, lower_ni, upper_ni, grid, 0, xc.gd)
        
        Z_i = Z_i.reshape(len(Z_i), -1)

        assert (np.array([(Z >= -1).any() for Z in Z_i.T])).all()
        assert (np.array([(Z <= -1).any() for Z in Z_i.T])).all()
                          


    def test_16_symmetricZs_lessgreatm1(self):
        num_e = np.random.randint(100) + 1
        n_sg = self.get_a_density()
        n_sg = n_sg * num_e / xc.gd.integrate(n_sg)
        
        ni_grid, lower_ni, upper_ni = xc.get_ni_grid(0, 1, n_sg)
        grid = self.get_grid()
        Z_i, _, _ = xc.symmetricmode_Zs(n_sg, ni_grid, lower_ni, upper_ni, grid, 0, xc.gd)

        Z_i = Z_i.reshape(len(Z_i), -1)

        assert (np.array([(Z >= -1).any() for Z in Z_i.T])).all()
        assert (np.array([(Z <= -1).any() for Z in Z_i.T])).all()
                     
    def test_17_standardZs_parallel(self):
        num_e = np.random.randint(100) + 1
        n_sg = self.get_a_density()
        n_sg = n_sg * num_e / xc.gd.integrate(n_sg)

        def local_test(rank, size):
            ni_grid, lower_ni, upper_ni = xc.get_ni_grid(rank, size, n_sg)
            grid = self.get_grid()
            Z_i, _, _ = xc.standardmode_Zs(n_sg, ni_grid, lower_ni, upper_ni, grid, 0, xc.gd)
            return Z_i

        def global_test(size, global_data):
            Z_i = np.vstack(global_data)
            assert len(Z_i) == len(global_data) * len(global_data[0])

            Z_i = Z_i.reshape(len(Z_i), -1)
            assert (np.array([(Z >= -1).any() for Z in Z_i.T])).all()
            assert (np.array([(Z <= -1).any() for Z in Z_i.T])).all()
            
        self.execute_test_for_random_mpiworld(local_test, global_test)

    def test_18_symmetricZs_parallel(self):
        num_e = np.random.randint(100) + 1
        n_sg = self.get_a_density()
        n_sg = n_sg * num_e / xc.gd.integrate(n_sg)

        def local_test(rank, size):
            ni_grid, lower_ni, upper_ni = xc.get_ni_grid(rank, size, n_sg)
            grid = self.get_grid()
            Z_i, _, _ = xc.symmetricmode_Zs(n_sg, ni_grid, lower_ni, upper_ni, grid, 0, xc.gd)
            return Z_i

        def global_test(size, global_data):
            Z_i = np.vstack(global_data)
            assert len(Z_i) == len(global_data) * len(global_data[0])

            Z_i = Z_i.reshape(len(Z_i), -1)
            assert (np.array([(Z >= -1).any() for Z in Z_i.T])).all()
            assert (np.array([(Z <= -1).any() for Z in Z_i.T])).all()
            
        self.execute_test_for_random_mpiworld(local_test, global_test)

    def get_some_Zs(self, Z_fct):
        num_e = np.random.randint(100) + 1
        n_sg = self.get_a_density()
        n_sg = n_sg * num_e / xc.gd.integrate(n_sg)
        assert np.allclose(xc.gd.integrate(n_sg), num_e)
        ni_grid, lower, upper = xc.get_ni_grid(0, 1, n_sg)
        grid = self.get_grid()
        Z_ig, Z_lowerg, Z_upperg = Z_fct(n_sg, ni_grid, lower, upper, grid, 0, xc.gd)
        return Z_ig, Z_lowerg, Z_upperg
        
    def test_19_get_alphas(self):
        Z_ig, Z_lowerg, Z_upperg = self.get_some_Zs(xc.standardmode_Zs)
        alpha_ig = xc.get_alphas(Z_ig, Z_lowerg, Z_upperg)

        alpha_ir = alpha_ig.reshape(len(alpha_ig), -1)
        assert np.allclose(alpha_ir.sum(axis=0), 1)
        assert (alpha_ir >= 0).all()
        not_zeros = np.logical_not(np.isclose(alpha_ir, 0))
        count_notzeros = not_zeros.sum(axis=0)
        assert np.logical_or(np.isclose(count_notzeros, 1), np.isclose(count_notzeros, 2)).all()

    def test_20_get_symalphas(self):
        Z_ig, Z_lowerg, Z_upperg = self.get_some_Zs(xc.symmetricmode_Zs)
        alpha_ig = xc.get_alphas(Z_ig, Z_lowerg, Z_upperg)
        
        alpha_ir = alpha_ig.reshape(len(alpha_ig), -1)
        assert np.allclose(alpha_ir.sum(axis=0), 1)
        assert (alpha_ir >= 0).all()
        not_zeros = np.logical_not(np.isclose(alpha_ir, 0))
        count_notzeros = not_zeros.sum(axis=0)
        assert np.logical_or(np.isclose(count_notzeros, 1), np.isclose(count_notzeros, 2)).all()

    def test_21_alphaval(self):
        Z_isg = np.array([[[[np.linspace(-1, 0, 10)]]]]).T
        Z_lower = np.array([[[[-1]]]])
        Z_upper = np.array([[[[0]]]])
        alpha_i = xc.get_alphas(Z_isg, Z_lower, Z_upper)
        expected = np.zeros_like(Z_isg)
        expected[0, 0, 0, 0, 0] = 1
        
        assert np.allclose(alpha_i, expected)
        notzero = np.logical_not(np.isclose(alpha_i, 0))
        assert notzero.sum() == 1 or notzero.sum() == 2

    def test_22_alphaval2_nonmonotonic(self):
        return "skipped"
        Z_i = np.zeros((100, 1, 1, 1, 1))
        index = np.random.randint(99)
        delta = 0.1
        relweight = np.random.rand()
        Z_i[index, 0, 0, 0, 0] = -1 - delta * relweight
        Z_i[index+1, 0, 0, 0, 0] = -1 + delta * (1 - relweight)
        Z_lower = np.array([[[[Z_i[0, 0, 0, 0, 0]]]]])
        Z_upper = np.array([[[[Z_i[-1, -1, -1, -1, -1]]]]])
        assert Z_lower.shape == Z_i.shape[1:]
        alpha_i = xc.get_alphas(Z_i, Z_lower, Z_upper)
        expected = np.zeros_like(Z_i)
        expected[index, 0, 0, 0, 0] = 1 - relweight
        expected[index+1, 0, 0, 0, 0] = relweight
        
        assert not np.allclose(expected, 0)
        notzeros = np.logical_not(np.isclose(alpha_i, 0))
        countnotzero = notzeros.sum()
        assert countnotzero == 1 or countnotzero == 2, countnotzero
        assert np.allclose(alpha_i, expected)

    def test_23_alphaval2(self):
        Z_i = np.zeros((100, 1, 1, 1, 1))
        index = np.random.randint(99)
        delta = 0.1
        relweight = np.random.rand()
        Z_i[:index, 0, 0, 0, 0] = -2
        Z_i[index, 0, 0, 0, 0] = -1 - delta * relweight
        Z_i[index+1:, 0, 0, 0, 0] = -1 + delta * (1 - relweight)
        Z_lower = np.array([[[[Z_i[0, 0, 0, 0, 0]]]]])
        Z_upper = np.array([[[[Z_i[-1, -1, -1, -1, -1]]]]])
        assert Z_lower.shape == Z_i.shape[1:]
        alpha_i = xc.get_alphas(Z_i, Z_lower, Z_upper)
        expected = np.zeros_like(Z_i)
        expected[index, 0, 0, 0, 0] = 1 - relweight
        expected[index+1, 0, 0, 0, 0] = relweight
        
        assert not np.allclose(expected, 0)
        notzeros = np.logical_not(np.isclose(alpha_i, 0))
        countnotzero = notzeros.sum()
        assert countnotzero == 1 or countnotzero == 2, countnotzero
        assert np.allclose(alpha_i, expected)

    def test_24_alphapara(self):
        n_sg = self.get_a_density()
        grid = self.gd.get_grid_point_coordinates()
        def local_test(rank, size):
            
            ni_j, nilower, niupper = xc.get_ni_grid(rank, size, n_sg)
            Z_ig, Zlower_g, Zupper_g = xc.get_Zs(n_sg, ni_j, nilower, niupper, grid, 0, self.gd)
            alpha_ig = xc.get_alphas(Z_ig, Zlower_g, Zupper_g)

            return alpha_ig

        def global_test(size, global_data):
            alpha_ig = np.vstack(global_data)
            alpha_ir = alpha_ig.reshape(len(alpha_ig), -1)
            assert np.allclose(alpha_ir.sum(axis=0), 1)
            notzeros = np.logical_not(np.isclose(alpha_ir, 0))
            countnotzeros_r = notzeros.sum(axis=0)
            assert (countnotzeros_r >= 1).all()
            assert (countnotzeros_r <= 2).all()
        self.execute_test_for_random_mpiworld(local_test, global_test)

    def test_25_calculateV1(self):
        alpha_ig, Z_ig, Zlower_g, Zupper_g, ni_j, nilower, niupper, n_sg = self.get_initial_stuff()
        
        V1_sg = xc.calculate_V1(alpha_ig, n_sg, self.grid, ni_j)
        self.isgoodnum(V1_sg)

    def isgoodnum(self, arr):
        assert np.allclose(arr, arr.real)
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()

    def test_26_calculateV1p(self):
        alpha_ig, Z_ig, Zlower_g, Zupper_g, ni_j, nilower, niupper, n_sg = self.get_initial_stuff()
        
        V1p_sg = xc.calculate_V1p(alpha_ig, n_sg, self.grid, ni_j)
        self.isgoodnum(V1p_sg)

    def test_27_calculateV2_normal(self):
        alpha_isg, Z_isg, Zlower_sg, Zupper_sg, ni_j, nilower, niupper, n_sg = self.get_initial_stuff()
        dalpha_isg = xc.get_dalpha_isg(alpha_isg, Z_isg, Zlower_sg, Zupper_sg, self.grid, ni_j, nilower, niupper, len(n_sg), xc.normal_dZ)
        V2_sg = xc.calculate_V2(dalpha_isg, n_sg, self.grid, ni_j)
        self.isgoodnum(V2_sg)
    
    def test_28_calculate_sympot(self):
        alpha_ig, Z_ig, Zlower_g, Zupper_g, ni_j, nilower, niupper, n_sg = self.get_initial_stuff()
        
        Vsympot_sg = xc.calculate_sym_pot_correction(alpha_ig, n_sg, self.grid, ni_j)
        self.isgoodnum(Vsympot_sg)

    def test_29_calculate_energy(self):
        alpha_ig, Z_ig, Zlower_g, Zupper_g, ni_j, nilower, niupper, n_sg = self.get_initial_stuff()
        
        e_g = xc.calculate_energy(alpha_ig, n_sg, self.gd, self.grid, ni_j)
        self.isgoodnum(e_g)

    def test_30_calculate_symene(self):
        alpha_ig, Z_ig, Zlower_g, Zupper_g, ni_j, nilower, niupper, n_sg = self.get_initial_stuff()
        
        esym_g = xc.calculate_sym_energy_correction(alpha_ig, n_sg, self.gd, self.grid, ni_j)
        self.isgoodnum(esym_g)

    def test_31_calculate_valence_corr(self):
        n1_sg = self.get_a_density()
        n2_sg = self.get_a_density()
        
        eval_g = xc.calculate_energy_correction_valence_mode(n1_sg, n2_sg)
        self.isgoodnum(eval_g)

        # expected = None
        # assert np.allclose(eval_g, expected)

    def test_32_calculateV2_normal(self):
        alpha_isg, Z_isg, Zlower_sg, Zupper_sg, ni_j, nilower, niupper, n_sg = self.get_initial_stuff()
        dalpha_isg = xc.get_dalpha_isg(alpha_isg, Z_isg, Zlower_sg, Zupper_sg, self.grid, ni_j, nilower, niupper, len(n_sg), xc.symmetric_dZ)
        V2_sg = xc.calculate_V2(dalpha_isg, n_sg, self.grid, ni_j)
        self.isgoodnum(V2_sg)

    def test_33_get_dalpha(self):
        # dalpha(r')/dn(r) = 0 for alpha(r') = 0
        # What to do at left edge i = N - 1? If alpha_i = 0, then return 0.
        ### If alpha_i != 0, then we are always (hopefully) in +1 branch. Then return +1 expression
        # What to do at the right edge i = 0? If alpha_i = 0, then return 0.
        ### If alpha_i != 0, then we are always (hopefully) in +0 branch. Then return +0 expression

        # We can substitute in any dZ function we want and any Z_i(r) functions we want, so we can control the results
        n_sg = self.get_a_density()
        print("NORM HERE: ", self.gd.integrate(n_sg))
        
        ni_j, ni_lower, ni_upper = xc.get_ni_grid(0, 1, n_sg)
        
        nnis = len(ni_j)

        alpha_isg = np.random.rand(*((nnis,) + n_sg.shape))*1000#np.zeros((nnis,) + n_sg.shape)
        
        def my_dZ(i, s, G_isr, grid_vg, ni_lowe2r, ni_uppe2r, alpha_isg):
            return np.random.rand(*G_isr[i, s].shape)

        Z_isg, Z_lower_sg, Z_upper_sg = xc.get_Zs(n_sg, ni_j, ni_lower, ni_upper, self.grid, 0, self.gd)
        
        print("MEAN Z", np.mean(Z_isg))
        import matplotlib.pyplot as plt

        # for j, ni in enumerate(ni_j):
        #     g_g = xc.get_pairdist_g(self.grid, ni, 0)
        #     plt.plot(g_g[:, 0, 0], label=str(ni))
        # plt.legend()
        # plt.show()

        print("")
        print("DV = ", self.gd.dv)
        print("1/DV = ", 1 / self.gd.dv)
        npts = np.array(Z_isg[0,0].shape).prod()
        print("NPTS: ", npts)
        print("SQRT NPTS: ", np.sqrt(npts))

        print("A Z VAL:" , Z_isg[0, 0, 0, 0, 0])

        D = Z_isg[:, 0, ...]
        D = D.reshape(len(Z_isg), -1)
        plt.plot(ni_j, np.mean(D, axis=1))
        plt.show()

        assert (Z_isg < -1.0).any()
        assert (Z_isg >= -1.0).any()

        dalpha_isg = xc.get_dalpha_isg(alpha_isg, Z_isg, Z_lower_sg, Z_upper_sg, self.grid, ni_j, ni_lower, ni_upper, 1, my_dZ)
        
        assert np.allclose(dalpha_isg, 0)

        
        ii = np.random.randint(nnis)
        
        delta = np.random.rand() + 0.01
        Z_isg[ii, :, ...] = -1.0
        Z_isg[:ii, :, ...] = -1.0 + delta
        Z_isg[ii+1:, :, ...] = -1.0 - delta

    
        Z_lower_sg[:, ...] = -1.0 + delta
        Z_upper_sg[:, ...] = -1.0 - delta

        alpha_isg[ii, 0] = 1.0

        def my_dZ2(i, s, G_isr, grid, ni_low, ni_upp, alpha_isg):
            return np.ones(G_isr[0, s].shape)

        dalpha_isg = xc.get_dalpha_isg(alpha_isg, Z_isg, Z_lower_sg, Z_upper_sg, self.grid, ni_j, ni_lower, ni_upper, 1, my_dZ2)

        assert np.allclose(dalpha_isg[ii, 0], 1 / (-1.0 - delta - (-1.0))), "MEAN: {}, STD: {}, ii: {}".format(dalpha_isg[ii, 0].mean(), dalpha_isg[ii+1, 0].std(), ii)
        assert np.allclose(dalpha_isg[:ii, 0], 0)

        if ii < nnis - 1:
            assert np.allclose(dalpha_isg[ii+1,0], -1 / (-1.0 - delta - (-1.0))), "MEAN: {}, STD: {}".format(dalpha_isg[ii+1,0].mean(), dalpha_isg[ii+1,0].std())
            if ii < nnis - 2:
                das = dalpha_isg[ii+2,0]
                assert np.allclose(dalpha_isg[ii+2:, 0], 0), "MEAN: {}, STD: {}".format(das.mean(), das.std())

        # raise NotImplementedError

    def test_34_get_dZ_normal(self):
        raise NotImplementedError

    def test_35_get_dZ_symmetric(self):
         raise NotImplementedError

    def test_36_calculateV2_symmetric(self):
         raise NotImplementedError

if __name__ == "__main__":
    import sys
    tester = Tester()
    if len(sys.argv) > 1:
        if sys.argv[1] == "multi":
            if len(sys.argv) > 3:
                its = int(sys.argv[2])
                number = sys.argv[3]
                tester.run_tests(multi=True, its=its, number=number)
            else:
                its = int(sys.argv[2])
                tester.run_tests(multi=True, its=its)
        else:
            tester.run_tests(number=sys.argv[1])
    else:
        tester.run_tests()
