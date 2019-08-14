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
        self.gd = xc.gd.new_descriptor(comm=mpi.serial_comm)

    def get_a_density(self):
        num_el = np.random.randint(100) + 1
        gd = xc.gd.new_descriptor(comm=mpi.serial_comm)
        grid = gd.get_grid_point_coordinates()
        dens = np.zeros(grid.shape[1:])
        densf = np.fft.fftn(dens)
        densf = np.random.rand(*densf.shape) + 0.1
        densf[0,0,0] = densf[0,0,0] + 1.0
        res = np.array([np.fft.ifftn(densf).real])
        res = res + np.min(res)
        res = res * num_el / gd.integrate(res)
        res[res < 1e-7] = 1e-8
        assert (res >= 0).all()
        assert res.ndim == 4
        assert not np.allclose(res, 0)
        return res

    def get_initial_stuff(self):
        n_sg = self.get_a_density()
        nae_sg = xc.get_ae_density(xc.gd, n_sg)
        ni_j, nilower, niupper = xc.get_ni_grid(0, 1, nae_sg)
        f_isg = xc.get_f_isg(ni_j, nilower, niupper, nae_sg)
        
        w_isg = xc.get_w_isg(ni_j, nae_sg, self.gd, xc._theta_filter)
        
        nu_sg = xc.weight_density(f_isg, w_isg)
        assert np.allclose(nu_sg, nu_sg.real)

        EWLDA_g, vLDA_sg = xc.calculate_lda_energy_and_potential(nu_sg)
        
        return n_sg, nae_sg, ni_j, nilower, niupper, f_isg, w_isg, nu_sg, EWLDA_g, vLDA_sg

    def test_01_unnormedpot_get(self):
        n_sg, nae_sg, ni_j, nilower, niupper, f_isg, w_isg, nu_sg, EWLDA_g, vLDA_sg = self.get_initial_stuff()
        
        vWLDA_sg = xc.calculate_unnormed_wlda_pot(f_isg, w_isg, vLDA_sg)
        
    def test_02_unnormedpot_isgoodnum(self):
        n_sg, nae_sg, ni_j, nilower, niupper, f_isg, w_isg, nu_sg, EWLDA_g, vLDA_sg = self.get_initial_stuff()
        
        vWLDA_sg = xc.calculate_unnormed_wlda_pot(f_isg, w_isg, vLDA_sg)
        assert np.allclose(vWLDA_sg, vWLDA_sg.real)
        assert not np.isnan(vWLDA_sg).any()
        assert not np.isinf(vWLDA_sg).any()









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




