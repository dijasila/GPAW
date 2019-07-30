from testframework import BaseTester
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, mpi


atoms = Atoms("H2", positions=[[0,0,0],[0,0,2]], cell=5*np.identity(3))
calc = GPAW(mode=PW(200), xc="WLDA_renorm", txt=None)#, convergence={'density': 1e-1})
calc.initialize(atoms=atoms)
calc.set_positions(atoms)
xc = calc.hamiltonian.xc

class Tester(BaseTester):
    def __init__(self):
        self.gd = xc.gd
        self.default_filter = xc._theta_filter
    

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

    def test_01_cangetaedensity(self):
        n_sg = self.get_a_density()
        nae_sg = xc.get_ae_density(self.gd, n_sg)

    def test_02_somechangeaedens(self):
        n_sg = self.get_a_density()
        ncopy = n_sg.copy()
        nae_sg = xc.get_ae_density(self.gd, n_sg)
        assert not np.allclose(nae_sg, n_sg)

    def test_03_getnigrid(self):
        n_sg = self.get_a_density()
        n_i, lower, upper = xc.get_ni_grid(0, 1, n_sg)

    def test_04_nomissingpts(self):
        n_sg = np.ones(calc.wfs.gd.get_grid_point_coordinates().shape[1:])
        n_i0, lower0, upper0 = xc.get_ni_grid(0, 2, n_sg)
        n_i1, lower1, upper1 = xc.get_ni_grid(1, 2, n_sg)
        
        assert lower0 == 0
        assert upper0 == n_i1[0]
        assert lower1 == n_i0[-1]
        assert upper1 == 1
        
        full = np.linspace(0, 1, 20)
        full_recon = np.hstack([n_i0, n_i1])
        assert np.allclose(full, full_recon)

    def test_05_randomworld_nomissing(self):
        n_sg = np.ones(calc.wfs.gd.get_grid_point_coordinates().shape[1:])
        size = np.random.randint(10) + 1
        n_i_grids = []
        for i in range(size):
            n_i, lower, upper = xc.get_ni_grid(i, size, n_sg)
            n_i_grids.append((n_i, lower, upper))
        
        assert n_i_grids[0][0][0] == 0
        assert n_i_grids[-1][0][-1] == 1

        full_recon = [n_i_grids[0][0]]
        for i in range(1, size-1):
            assert n_i_grids[i][1] == n_i_grids[i-1][0][-1]
            assert n_i_grids[i][2] == n_i_grids[i+1][0][0]
            full_recon.append(n_i_grids[i][0])
        if size > 1:
            full_recon.append(n_i_grids[-1][0])
        full_recon = np.hstack(full_recon)
        
        assert np.allclose(full_recon, np.linspace(0, 1, size*10)), "Size: {}, reconshape: {}".format(size, full_recon.shape)

    def test_06_nigridcontainsdensvals(self):
        n_sg = self.get_a_density()
        n_sg = n_sg * (np.random.rand() + 0.1) * 10
        ni_j, lower, upper = xc.get_ni_grid(0, 1, n_sg)

        assert np.max(ni_j) >= np.max(n_sg)
        assert np.min(ni_j) <= np.min(n_sg)

    def test_07_f_isumstoone(self):
        n_sg = self.get_a_density()
        ni_j, lower, upper = xc.get_ni_grid(0, 1, n_sg)
        assert np.min(n_sg) >= lower, "Min: {}, lower: {}".format(np.min(n_sg), lower)
        assert np.max(n_sg) <= upper, "Max: {}, upper: {}".format(np.max(n_sg), upper)
        f_isg = xc.get_f_isg(ni_j, lower, upper, n_sg)

        assert np.allclose(f_isg.sum(axis=0), 1), "Max abs diff: {}".format(np.max(np.abs(f_isg.sum(axis=0) -1)))
        
    def test_08_fisg_correctforconstantdens(self):
        n_sg = np.ones(self.get_a_density().shape)
        ones = np.ones(n_sg.shape)
        ni_j, lower, upper = xc.get_ni_grid(0, 1, ones)
        factor = (ni_j[5] + ni_j[4]) / 2
        n_sg *= factor
        f_isg = xc.get_f_isg(ni_j, lower, upper, n_sg)
        expected = np.zeros(len(ni_j))
        expected[4:6] = 0.5
        na = np.newaxis
        assert np.allclose(f_isg - expected[:, na, na, na, na], 0), "f_isg: {}. ni_j: {}".format(f_isg[:, 0, 0, 0, 0], ni_j)
        
    def test_09_fisg_correctforconstantdens2(self):
        n_sg = np.ones(self.get_a_density().shape)
        ones = np.ones(n_sg.shape)
        ni_j, lower, upper = xc.get_ni_grid(0, 1, ones)
        factor = ni_j[5]*0.8 + ni_j[4]*0.2
        n_sg *= factor
        f_isg = xc.get_f_isg(ni_j, lower, upper, n_sg)
        expected = np.zeros(len(ni_j))
        expected[4] = 0.2
        expected[5] = 0.8
        na = np.newaxis
        assert np.allclose(f_isg - expected[:, na, na, na, na], 0), "f_isg: {}. ni_j: {}".format(f_isg[:, 0, 0, 0, 0], ni_j)

    def test_10_fisg_correct_random(self):
        n_sg = np.ones(self.get_a_density().shape)
        ni_j, lower, upper = xc.get_ni_grid(0, 1, n_sg)
        expected_isg = np.zeros((len(ni_j),) + n_sg.shape)

        for ix, n_yz in enumerate(n_sg[0]):
            for iy, n_z in enumerate(n_yz):
                for iz, n in enumerate(n_z):
                    index = np.random.randint(len(ni_j)-1)
                    dist = np.random.rand()
                    expected_isg[index, 0, ix, iy, iz] = dist
                    expected_isg[index + 1, 0, ix, iy, iz] = 1 - dist
                    n_sg[0, ix, iy, iz] = dist*ni_j[index] + (1-dist)*ni_j[index+1]

        f_isg = xc.get_f_isg(ni_j, lower, upper, n_sg)
        assert np.allclose(f_isg, expected_isg)

    def test_11_fisg_sumstoonerandomparallel(self):
        n_sg = self.get_a_density()
        size = np.random.randint(10) + 2
        f_isgs = []
        for i in range(size):
            ni_j, lower, upper = xc.get_ni_grid(i, size, n_sg)
            f_isg = xc.get_f_isg(ni_j, lower, upper, n_sg)
            f_isgs.append(f_isg)

        f_isg = np.vstack(f_isgs)
        assert np.allclose(f_isg.sum(axis=0), 1), "max abs sum: {}".format(np.max(np.abs(f_isg.sum(axis=0))))

    def test_12_fisg_randomparallel(self):
        n_sg = np.ones(self.get_a_density().shape)
        size = np.random.randint(10) + 2
        ni_grids = []
        for i in range(size):
            ni_j, lower, upper = xc.get_ni_grid(i, size, n_sg)
            ni_grids.append((ni_j, lower, upper))
        
        ni_j = np.linspace(0, 1, 10*size)
        expected_isg = np.zeros((len(ni_j),) + n_sg.shape)

        for ix, n_yz in enumerate(n_sg[0]):
            for iy, n_z in enumerate(n_yz):
                for iz, n in enumerate(n_z):
                    index = np.random.randint(len(ni_j)-1)
                    dist = np.random.rand()
                    expected_isg[index, 0, ix, iy, iz] = dist
                    expected_isg[index + 1, 0, ix, iy, iz] = 1 - dist
                    n_sg[0, ix, iy, iz] = dist*ni_j[index] + (1-dist)*ni_j[index+1]
        
        f_isg_r = []
        for i in range(size):
            my_ni_j, my_lower, my_upper = ni_grids[i]
            f = xc.get_f_isg(my_ni_j, my_lower, my_upper, n_sg)
            f_isg_r.append(f)
        f_isg = np.vstack(f_isg_r)
        diff = np.unravel_index(np.argmax((f_isg - expected_isg).sum(axis=(0,1))), f_isg.shape[2:])
        assert np.allclose(f_isg, expected_isg), "f_isg:\n {}\n\n exp_isg:\n {} \n\n max abs diff {}".format(f_isg[:, 0, diff[0], diff[1], diff[2]], expected_isg[:, 0, diff[0], diff[1], diff[2]] , np.max(np.abs(f_isg - expected_isg)))
        
    def test_13_get_wisg(self):
        n_sg = self.get_a_density()
        ni_j, _, _, = xc.get_ni_grid(0, 1, n_sg)
        gd = xc.gd.new_descriptor(comm=mpi.serial_comm)
        w_isg = xc.get_w_isg(ni_j, n_sg, gd, self.default_filter) 

    def calc_wisg(self, gd, ni, n_sg):
        # Wouldnt work, need to integrate over all of space if we work in real space
        raise ValueError("Dont use this function")

    def test_14_wisg_value(self):
        for i in range(3):
            n_sg = 0
            count = 0
            maxcount = 10
            while np.allclose(n_sg, 0) and count < maxcount:
                n_sg = self.get_a_sym_density(dir=i)
            assert not np.allclose(n_sg, 0)
            ni_j, lower, upper = xc.get_ni_grid(0, 1, n_sg)
            w_isg = xc.get_w_isg(ni_j, n_sg, xc.gd, self.default_filter)
            assert np.allclose(w_isg, np.flip(w_isg, i + 2))

    def test_15_wisg_zerokF(self):
        n_sg = self.get_a_density()
        if np.allclose(n_sg.sum(), 0):
            n_sg += 1.0
        else:
            n_sg /= n_sg.sum()
        ni_j, lower, upper = xc.get_ni_grid(0, 1, n_sg)
        assert np.allclose(ni_j[0], 0)
        w_isg = xc.get_w_isg(ni_j, n_sg, xc.gd, self.default_filter)
        
        # Assert constant but not killed
        assert not np.allclose(w_isg[0, ...], 0)
        assert np.allclose(w_isg[0, :, 0, 0, 0], w_isg[0, :, :, :, :]), "Max abs diff: {}, Mean val w_isg: {}".format(np.max(np.abs(w_isg[0]-w_isg[0, :, 0, 0, 0])), np.mean(w_isg[0]))
        
        # Assert sum conserved
        assert np.allclose(1.0, w_isg[0].sum(axis=(1,2,3)))

    def test_16_wisg_largekF(self):
        # Should be no change
        n_sg = self.get_a_density()
        K_G = xc._get_K_G(xc.gd)
        ni_j, _, _ = xc.get_ni_grid(0, 1, n_sg*100*np.max(K_G)/np.max(n_sg))
        kernels = [xc._theta_filter, xc._fermi_kinetic]
        for i, kernel in enumerate(kernels):
            w_isg = xc.get_w_isg(ni_j, n_sg, xc.gd, kernel)
        
            assert np.allclose(w_isg[-1, :, :, :, :], n_sg), "Failed for kernel {}".format(i)

    def test_17_wisg_positivityconserved(self):
        # Filtered should always be positive
        # This was not automatically assured. Interesting.
        n_sg = self.get_a_density()*10
        assert (n_sg >= 0).all()
        ni_j, _, _ = xc.get_ni_grid(0, 1, n_sg)
        
        
        w_isg = xc.get_w_isg(ni_j, n_sg, xc.gd, xc._gaussian_filter)
        
        assert (w_isg >= 0).all(), "Min: {}, mean: {}".format(np.min(w_isg), np.mean(w_isg))

    def test_18_wisg_parallelget(self):
        n_sg = self.get_a_density()

        size = np.random.randint(10) + 2
        for i in range(size):
            ni_j, lower, upper = xc.get_ni_grid(i, size, n_sg)
            w_isg = xc.get_w_isg(ni_j, n_sg, xc.gd, self.default_filter)

    def test_19_wisg_parallel_tests(self):
        n_sg = self.get_a_density()
        n_sum = n_sg.sum()
        size = np.random.randint(10) + 2
        for i in range(size):
            ni_j, _, _ = xc.get_ni_grid(i, size, n_sg)
            w_isg = xc.get_w_isg(ni_j, n_sg, xc.gd, self.default_filter)

            # Only rank 0 has the constant weighted density
            if i == 0:
                assert not np.allclose(w_isg[0, ...], 0)
                assert np.allclose(w_isg[0, :, 0,0,0], w_isg[0, :, :, :, :])
            else:
                assert not np.allclose(w_isg[0, :, 0,0,0], w_isg[0, :, :, :, :])
            
            # No weighted densities are all zero
            for j in range(len(ni_j)):
                assert not np.allclose(w_isg[j], 0)
            # Sums are conserved
            assert np.allclose(w_isg.sum(axis=(1,2,3,4)), n_sum)


            # No loss of positivity for gaussian filter
            w_isg = xc.get_w_isg(ni_j, n_sg, xc.gd, xc._gaussian_filter)
            assert (w_isg >= 0).all()






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
