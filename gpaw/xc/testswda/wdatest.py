from testframework import BaseTester
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, mpi


atoms = Atoms("H2", positions=[[0,0,0], [0,0,2]], cell=5*np.identity(3))
calc = GPAW(mode=PW(200), xc="WDA_standard", txt=None)
calc.initialize(atoms=atoms)
calc.set_positions(atoms)
xc = calc.hamiltonian.xc

assert xc is not None





class Tester(BaseTester):
    def __init__(self):
        self.gd = xc.gd
    

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

    def test_01_getstandardZis(self):
        n_sg = self.get_a_density()
        raise NotImplementedError

    def test_02_getLDAxc(self):
        # Test for spin = 0 and spin = 1
        raise NotImplementedError

    def test_03_getsymmetricZis(self):
        raise NotImplementedError

    def test_04_standardZderiv(self):
        raise NotImplementedError

    def test_05_symmetricZderiv(self):
        raise NotImplementedError

    def test_06_indicators_sumtoone(self):
        ni_grid, _, _ = xc.get_ni_grid(0, 1, self.get_a_density())
        ind_ig = xc.build_indicators(ni_grid)
        assert np.allclose(ind_ig.sum(axis=0), 1)

    def test_07_indicators_greaterthanzero(self):
        ni_grid, _, _ = xc.get_ni_grid(0, 1, self.get_a_density())
        ind_ig = xc.build_indicators(ni_grid)
        assert (ind_ig >= 0).all()

    def execute_test_for_random_mpiworld(self, test_fct, global_test=None):
        mpi_size = np.random.randint(10) + 2
        global_data = []
        for rank in range(mpi_size):
            global_data.append(test_fct(rank, mpi_size))
        if global_test is not None:
            global_test(mpi_size, global_data)

    def test_08_indicators_parallel(self):
        n_sg = self.get_a_density()
        def local_test_fct(rank, size):
            ni_grid, _, _ = xc.get_ni_grid(rank, size, n_sg)
            ind_ig = xc.build_indicators(ni_grid)
            assert (ind_ig >= 0).all()
            return ind_ig
        
        def global_test(size, global_data):
            ind_sum_rg = np.array([g.sum(axis=0) for g in global_data])
            assert np.allclose(ind_sum_rg.sum(axis=0), 1)
            
        self.execute_test_for_random_mpiworld(local_test_fct, global_test)
            
    






if __name__ == "__main__":
    import sys
    tester = Tester()
    if len(sys.argv) > 1:
        if sys.argv[1] == "multi":
            its = int(sys.argv[2])
            tester.run_tests(multi=True, its=its)
        else:
            tester.run_tests(number=sys.argv[1])
    else:
        tester.run_tests()
