from gpaw.westinterface import GPAWServer
from testframework import test_method, BaseTester
from ase import Atoms
from gpaw import GPAW, PW

class DummyAtoms:
    def __init__(self):
        self.has_run = False
        pass

    def get_potential_energy(self):
        self.has_run = True
        return 0

class DummyCalc:
    def __init__(self, external=None):
        self.vext = external
        pass
    def get_potential_energy(self, smth):
        return 0


atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=(4,4,4))
calc = GPAW(mode=PW(100), setups="ae", txt="testout.txt")
atoms.set_calculator(calc)
server = GPAWServer("servertest.xml", "serverout", atoms, calc)

class Tester(BaseTester):
    def __init__(self):
        pass

    def test_01_mainloopmaxruns(self):
        maxruns = 1
        server.main_loop(maxruns=maxruns)
        
    def test_02_readsfromfile(self):
        lserver = GPAWServer("servertest.xml", "tmpserverout", atoms, calc)
        maxruns = 1
        lserver.main_loop(maxruns=maxruns)
        try:
            lserver.input_file = "doesnexists.xml"
            lserver.main_loop(maxruns=maxruns)
            raise NameError("Didnt read file")
        except ValueError:
            pass

    def test_03_dummycalculation(self):
        latoms = DummyAtoms()
        lserver = GPAWServer("servertest.xml", "tmpserverout", latoms, calc)
        lserver.main_loop(maxruns=1)
        assert latoms.has_run

    def test_04_setsexternalpotential(self):
        server.main_loop(maxruns=1)
        assert server.calc.hamiltonian.vext is not None

    def test_05_externalpotvalues(self):
        import numpy as np
        lserver = GPAWServer("servervaluetest.xml", "tmpserverout", atoms, calc)
        lserver.main_loop(maxruns=1)
        ext_g = lserver.calc.hamiltonian.vext.vext_g
        expected = np.zeros_like(ext_g)
        nx, ny, nz = ext_g.shape
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    expected[ix, iy, iz] = ix*ny*nz + iy*nz + iz
        
        assert np.allclose(expected, ext_g)

    # Returns/Writes density to xml
    # Can read kill signal, should come from supercontroller if supercontroller finds that WEST client has finished

if __name__ == "__main__":
    tester = Tester()
    tester.run_tests()
