from testframework import BaseTester
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, mpi


atoms = Atoms("H2", positions=[[0,0,0], [0,0,2]], cell=5*np.identity(3), pbc=True)
calc = GPAW(mode=PW(400), xc="WLDA_altmethod", txt=None)
calc.initialize(atoms=atoms)
calc.set_positions(atoms)
xc = calc.hamiltonian.xc

assert xc is not None

class Tester(BaseTester):
    def _init__(self):
        pass
        
    def get_a_density(self):
        gd = xc.gd.new_descriptor(comm=mpi.serial_comm)
        grid = gd.get_grid_point_coordinates()
        dens = np.zeros(grid.shape[1:])
        densf = np.fft.fftn(dens)
        densf = np.random.rand(*densf.shape) + 0.1
        densf[0,0,0] = densf[0,0,0] + 1.0
        res = np.array([np.fft.ifftn(densf).real])
        res = res + np.min(res)
        res[res < 1e-7] = 1e-8
        assert (res >= 0).all()
        assert res.ndim == 4
        assert not np.allclose(res, 0)
        return res

    def get_a_density_fromgd(self, gdd):
        gd = gdd.new_descriptor(comm=mpi.serial_comm)
        grid = gd.get_grid_point_coordinates()
        dens = np.zeros(grid.shape[1:])
        densf = np.fft.fftn(dens)
        densf = np.random.rand(*densf.shape) + 0.1
        densf[0,0,0] = densf[0,0,0] + 1.0
        res = np.array([np.fft.ifftn(densf).real])
        res = res + np.min(res)
        res[res < 1e-7] = 1e-8
        assert (res >= 0).all()
        assert res.ndim == 4
        assert not np.allclose(res, 0)
        return res

    def get_a_localized_density_fromgd(self, gdd):
        gd = gdd.new_descriptor(comm=mpi.serial_comm)
        grid = gd.get_grid_point_coordinates()
        dists = np.linalg.norm(grid, axis=0)
        res = np.array([np.exp(-dists**2)])
        norm = gd.integrate(res)
        res = res / norm * 1.0
        assert (res >= 0).all()
        assert res.ndim == 4
        assert not np.allclose(res, 0)
        return res


    def test_01_ldaX(self):
        # Calc E_X with standard kernel, then div/mult to get WLDA
        # Calc with n = n* in WLDA implementation
        # 
        my_as = xc.distribute_alphas(xc.nindicators, 0, 1)

        from gpaw.xc.lda import PurePythonLDAKernel, lda_x
        lda_kernel = PurePythonLDAKernel()

        n_sg = self.get_a_density()
        nstar_sg = self.get_a_density()
        assert len(n_sg) == 1
        
        
        ldaEX = np.zeros_like(nstar_sg[0])
        v = np.zeros_like(nstar_sg)
        lda_x(0, ldaEX, nstar_sg[0], v)

        expected = ldaEX * n_sg / nstar_sg

        wldaEX = np.zeros_like(nstar_sg[0])
        vWLDA = np.zeros_like(nstar_sg)

        xc.lda_x(0, wldaEX, n_sg[0], nstar_sg[0], vWLDA, my_as)
        
        assert np.allclose(expected, wldaEX)

    def test_02_ldaC(self):
        # Calc with n = n* and check against standard kernel
        # Check E_C with div/mult method
        my_as = xc.distribute_alphas(xc.nindicators, 0, 1)

        from gpaw.xc.lda import PurePythonLDAKernel, lda_c
        lda_kernel = PurePythonLDAKernel()

        n_sg = self.get_a_density()
        nstar_sg = self.get_a_density()
        assert len(n_sg) == 1
        
        
        ldaEC = np.zeros_like(nstar_sg[0])
        v = np.zeros_like(nstar_sg)
        lda_c(0, ldaEC, nstar_sg[0], v, 0)

        expected = ldaEC * n_sg / nstar_sg

        wldaEC = np.zeros_like(nstar_sg[0])
        vWLDA = np.zeros_like(nstar_sg)
        
        xc.lda_c(0, wldaEC, n_sg[0], nstar_sg[0], vWLDA, 0, my_as)
        
        assert np.allclose(expected, wldaEC)

    def test_03_indicator_math(self):
        # Test mathematical properties of indicators
        # Should sum = 1, f_alpha >= 0
        # Should test the function get_indicator_alpha and get_indicator_g

        alphas = xc.alphas

        val_array = np.random.rand(100)*(max(alphas) - min(alphas)) + min(alphas)
        sum_array = np.zeros_like(val_array)


        ind_i0 = xc.get_indicator_alpha(0)
        assert np.allclose(ind_i0(0), 1)
        assert np.allclose(ind_i0((alphas[0] + alphas[1])/2), 0.5)

        for ia, a in enumerate(alphas):
            if ia == len(alphas) - 1:
                continue
            ind_ia = xc.get_indicator_alpha(ia)
            assert np.allclose(ind_ia(alphas[ia]), 1), "ia: {}, ind_ia(alphas[ia]): {}".format(ia, ind_ia(alphas[ia]))
            assert np.allclose(ind_ia((alphas[ia] + alphas[ia+1])/2), 0.5)
    
        n_sg = self.get_a_density().reshape(-1)
        res_sg = np.zeros_like(n_sg)
        for ia, a in enumerate(alphas):
            ind_ia = xc.get_indicator_alpha(ia)
            for iv, val in enumerate(val_array):
                sum_array[iv] += ind_ia(val)

            ind_sg = xc.get_indicator_sg(n_sg, ia)
            res_sg += ind_sg

        
        assert np.allclose(res_sg, 1)
        assert np.allclose(sum_array, 1), np.mean(sum_array)

    def test_04_weightednormisunchanged(self):
        # Input density and weighted density have same norm
        wn_sg = self.get_a_density()
        myas = xc.distribute_alphas(xc.nindicators, 0, 1)
        nstar_sg = xc.alt_weight(wn_sg, myas, xc.gd)

        norm1 = xc.gd.integrate(wn_sg)
        norm2 = xc.gd.integrate(nstar_sg)

        assert np.allclose(norm1, norm2), "norm1 = {}, norm2: {}".format(norm1, norm2)

    def test_05_indicators_parallel(self):
        # Test that range is distributed correctly
        size = np.random.randint(100) + 2
        
        ninds = np.random.randint(100, 1000)
        xc.alphas = xc.setup_indicator_grid(ninds)
        xc.setup_indicators(xc.alphas)

        alpha_indices = []
        for rank in range(size):
            my_alpha_indices = xc.distribute_alphas(ninds, rank, size)
            alpha_indices.append(my_alpha_indices)

        joined = []

        for i, inds in enumerate(alpha_indices):
            if i == 0:
                joined = joined + list(inds)
                continue

            assert inds[0] > alpha_indices[i - 1][-1]

            joined = joined + list(inds)
        assert np.allclose(joined, range(len(xc.alphas)))

    def test_06_indicator_math_par(self):
        # Parallel version of above math test
        size = np.random.randint(100) + 2
        ninds = np.random.randint(500, 1000)
        xc.alphas = xc.setup_indicator_grid(ninds)
        xc.setup_indicators(xc.alphas)

        for rank in range(size):
            alphas = xc.alphas
            my_inds = xc.distribute_alphas(ninds, rank, size)
            mymin = alphas[my_inds[0]]
            mymax = alphas[my_inds[-1]]

            ind_i0 = xc.get_indicator_alpha(my_inds[0])
            assert np.allclose(ind_i0(alphas[my_inds[0]]), 1)
            assert np.allclose(ind_i0((alphas[my_inds[0]] + alphas[my_inds[1]])/2), 0.5), "{}\n{}".format(len(my_inds), len(alphas))
            for iia, ia in enumerate(my_inds):
                if iia == len(my_inds) - 1:
                    continue
                ind_ia = xc.get_indicator_alpha(ia)
                assert np.allclose(ind_ia(alphas[ia]), 1), "ia: {}, ind_ia(alphas[ia]): {}".format(ia, ind_ia(alphas[ia]))
                assert np.allclose(ind_ia((alphas[ia] + alphas[ia+1])/2), 0.5)


            val_array = np.random.rand(100)*(mymax - mymin) + mymin
            sum_array = np.zeros_like(val_array)

            for ia in my_inds:
                ind_ia = xc.get_indicator_alpha(ia)
                for iv, val in enumerate(val_array):
                    sum_array[iv] += ind_ia(val)

            assert np.allclose(sum_array, 1), np.mean(sum_array)

    def test_07_weightednormisunchanged_par(self):
        # Input density and weighted density have same norm
        size = np.random.randint(10) + 2
        ninds = np.random.randint(20, 25)
        xc.alphas = xc.setup_indicator_grid(ninds)
        xc.setup_indicators(xc.alphas)

        wn_sg = self.get_a_density()
        res_sg = np.zeros_like(wn_sg)
        for rank in range(size):
            myas = xc.distribute_alphas(ninds, rank, size)
            res_sg += xc.alt_weight(wn_sg, myas, xc.gd)

        norm1 = xc.gd.integrate(wn_sg)
        norm2 = xc.gd.integrate(res_sg)

        assert np.allclose(norm1, norm2), "norm1 = {}, norm2: {}".format(norm1, norm2)

        
    def test_08_weighteddens_noneg(self):
        # Test that weighted density is non-negative
        # if input density is also non-neg
        size = np.random.randint(5) + 2
        ninds = np.random.randint(10, 15)
        xc.alphas = xc.setup_indicator_grid(ninds)
        xc.setup_indicators(xc.alphas)
        
        wn_sg = self.get_a_density()
        assert (wn_sg >= 0).all()
        res_sg = np.zeros_like(wn_sg)

        for rank in range(size):
            myas = xc.distribute_alphas(ninds, rank, size)
            res_sg += xc.alt_weight(wn_sg, myas, xc.gd)

        assert (res_sg >= 0).all()

    def test_09_weighteddens_sizeconsist(self):
        # Test that adding more vacuum eventually doesnt matter
        current = 10000
        for i, cellsize in enumerate([5, 5.5, 6]):#, 200]:
            latoms = Atoms("H2", positions=[[0,0,0], [0,0,2]], cell=cellsize*np.identity(3), pbc=True)
            lcalc = GPAW(mode=PW(100), xc="WLDA_altmethod", txt=None)
            lcalc.initialize(atoms=latoms)
            lcalc.set_positions(latoms)
            lxc = lcalc.hamiltonian.xc

            grid = lxc.gd.get_grid_point_coordinates()



            wn_sg = self.get_a_localized_density_fromgd(lxc.gd)

            
            myas = lxc.distribute_alphas(lxc.nindicators, 0, 1)
            nstar_sg = lxc.alt_weight(wn_sg, myas, lxc.gd)
            mid = nstar_sg.shape[1]//2
            assert nstar_sg[0, mid, mid, mid] < current
            current = nstar_sg[0, mid, mid, mid]

    def test_10_indicator_valueoutsidegrid(self):
        # Test that behaviour for indicators is correct
        # when the input value lies outside the alpha-grid
        # We should have that f_0(x) = 1, if x <= 0
        # f_MAX(x) = 1 if x >= MAX

        ind_i0 = xc.get_indicator_alpha(0)
        ind_im = xc.get_indicator_alpha(len(xc.alphas) - 1)

        lowval = np.random.rand() * (3 * xc.alphas[0]) - 4*xc.alphas[0]
        highval = np.random.rand() * (3 * xc.alphas[-1]) + xc.alphas[-1]

        assert ind_i0(lowval) == 1
        assert ind_im(highval) == 1
        assert ind_i0(highval) == 0
        assert ind_im(lowval) == 0

    def isgoodnum(self, arr):
        assert np.allclose(arr, arr.real)
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()

    def test_11_foldwderiv_goodnum(self):
        # Check that result is goodnum for variety of input
        ninds = np.random.randint(10, 20)
        xc.alphas = xc.setup_indicator_grid(ninds)
        xc.setup_indicators(xc.alphas)


        n1_g = self.get_a_density()[0]
        n2_g = self.get_a_density()[0]

        
        myas = xc.distribute_alphas(ninds, 0, 1)

        res_g = xc.fold_with_derivative(n1_g, n2_g, myas)
        
        self.isgoodnum(res_g)

    def test_12_foldwderiv(self):
        # Maybe take a function with f(G) = 1
        # and check that result is eq. 16 in overleaf notes
        raise NotImplementedError

    def test_12_getweight(self):
        # is goodnum
        # is normalized, also in realspace
        raise NotImplementedError

    def test_13_sizeKGvsalphas(self):
        # Test that alphas cover K_G grid
        raise NotImplementedError

    def test_14_alphasnooverlap(self):
        # Check for variety of MPI worlds
        # that there is no overlap between assigned alphas
        raise NotImplementedError

    def test_15_indicators_nonuniform(self):
        # Check that indicators work for non-uniform grid
        raise NotImplementedError

    def test_16_unaffectedbyfilter(self):
        # Input a function with only low momenta into weight-function
        # Output should be unaffected
        raise NotImplementedError

    def test_17_indicator_deriv(self):
        # Test that indicator implementation matches with deriv impl
        # via finite difference
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



