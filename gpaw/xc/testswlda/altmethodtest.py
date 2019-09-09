from testframework import BaseTester
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, mpi


atoms = Atoms("H2", positions=[[0,0,0], [0,0,2]], cell=5*np.identity(3))
calc = GPAW(mode=PW(200), xc="WLDA_altmethod", txt=None)
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


    def test_01_ldaX(self):
        # Calc E_X with standard kernel, then div/mult to get WLDA
        # Calc with n = n* in WLDA implementation
        # 
        my_as = xc.distribute_alphas(xc.nindicators)

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
        my_as = xc.distribute_alphas(xc.nindicators)

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

    def test_03_weightednormisunchanged(self):
        # Input density and weighted density have same norm
        wn_sg = self.get_a_density()
        myas = xc.distribute_alphas(xc.nindicators)
        nstar_sg = xc.alt_weight(wn_sg, myas, xc.gd)

        norm1 = xc.gd.integrate(wn_sg)
        norm2 = xc.gd.integrate(nstar_sg)

        assert np.allclose(norm1, norm2)

    def test_04_indicators_parallel(self):
        # Test that range is distributed correctly
        raise NotImplementedError

    def test_05_indicator_math(self):
        # Test mathematical properties of indicators
        # Should sum = 1, f_alpha >= 0
        # Should test the function get_indicator_alpha and get_indicator_g
        raise NotImplementedError

    def test_06_indicator_math_par(self):
        # Parallel version of above test
        raise NotImplementedError
        
    def test_07_weighteddens_noneg(self):
        # Test that weighted density is non-negative
        # if input density is also non-neg
        raise NotImplementedError

    def test_08_weighteddens_sizeconsist(self):
        # Test that adding more vacuum eventually doesnt matter
        raise NotImplementedError

    def test_09_indicator_valueoutsidegrid(self):
        # Test that behaviour for indicators is correct
        # when the input value lies outside the alpha-grid
        # We should have that f_0(x) = 1, if x <= 0
        # f_MAX(x) = 1 if x >= MAX
        raise NotImplementedError

    def test_10_foldwderiv_goodnum(self):
        # Check that result is goodnum for variety of input
        raise NotImplementedError

    def test_11_foldwderiv(self):
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



