import numpy as np
from testframework import BaseTester
from ase import Atoms
from gpaw import GPAW, PW

d = 2
atoms = Atoms("H2", positions=[[0,0,0], [0, 0, d]], cell=20*np.identity(3))
calc = GPAW(mode=PW(200), xc="WLDA_renorm", txt=None)
atoms.set_calculator(calc)
calc.initialize(atoms=atoms)
setup_H = calc.setups[0]
spos_ac = [[0,0,0]]
        


class Tester(BaseTester):
    pass

    def get_pseud(self):
        return setup_H.calculate_pseudized_atomic_density(spos_ac)

    def get_zeros(self):
        return np.zeros(calc.wfs.gd.get_grid_point_coordinates().shape[1:])
        

    def test_01_getpseudizeddens(self):
        pseud = setup_H.calculate_pseudized_atomic_density(spos_ac)

    def test_02_pseudizedisnotzero(self):
        pseud = setup_H.calculate_pseudized_atomic_density(spos_ac)
        dens = self.get_zeros()
        pseud.add(dens)
        assert not np.allclose(dens, 0)

    def test_03_noteqatomic(self):
        pseud = self.get_pseud()
        atomic = setup_H.calculate_atomic_density()
        dens = self.get_zeros()
        pseud.add(dens)
        nx, ny, nz = dens.shape
        assert not np.allclose(dens[:, 0, 0], atomic[:nx])





if __name__ == "__main__":
    import sys
    tester = Tester()
    if len(sys.argv) > 1:
        tester.run_tests(number=sys.argv[1])
    else:
        tester.run_tests()
