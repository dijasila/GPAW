from gpaw.westinterface import GPAWServer
from testframework import BaseTester
from ase import Atoms
from gpaw import GPAW, PW
import numpy as np
class DummyAtoms:
    def __init__(self):
        self.has_run = False
        pass

    def get_potential_energy(self):
        self.has_run = True
        return 0

    def get_cell(self):
        return np.zeros((3,3))

class DummyParams:
    def __init__(self):
        pass

class DummyCalc:
    def __init__(self, external=None):
        self.vext = external
        self.parameters = DummyParams() 
        pass
    def get_potential_energy(self, smth):
        return 0
    
    def initialize(self, atoms=None, reading=False):
        self.vext = self.parameters.external.values
        pass

    def get_all_electron_density(self):
        return np.zeros_like(self.vext.array)

    def get_pseudo_density(self):
        return np.zeros_like(self.vext)

    def set(self, **kwargs):
        pass


atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=(4,4,4))
calc = GPAW(mode=PW(100), txt="testout.txt")
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
        lcalc = DummyCalc()
        lserver = GPAWServer("servertest.xml", "tmpserverout", latoms, lcalc)
        lserver.main_loop(maxruns=1)
        assert latoms.has_run

    def test_04_setsexternalpotential(self):
        server.main_loop(maxruns=1)
        assert server.calc.hamiltonian.vext is not None

    def test_05_externalpotvalues(self):
        import numpy as np
        lserver = GPAWServer("servervaluetest.xml", "tmpserverout", atoms, calc)
        data = lserver.xmlrw.read("servervaluetest.xml")
        lserver.init_hamiltonian(data.array, data.domain)
        lserver.atoms.get_potential_energy()
        ext_g = lserver.calc.hamiltonian.vext.vext_g
       
        nx, ny, nz = ext_g.shape
        expected = np.arange(nx*ny*nz).reshape(nx, ny, nz)
        assert np.allclose(expected, ext_g)

    def test_06_writesoutfile(self):
        server.main_loop(maxruns=1)
        import os
        filename = server.output_file
        assert os.path.exists(filename)

    def test_07_outdenszeroforpotzero(self):
        lserver = GPAWServer("zerovalserver.xml", "zero", atoms, calc)
        data = lserver.xmlrw.read("zerovalserver.xml")
        assert np.allclose(data.array, 0)
        lserver.main_loop(maxruns=1)
        ldata = lserver.xmlrw.read("zero.xml")
        assert np.allclose(ldata.array, 0)

    def test_08_iskillable(self):
        fname = server.input_file.split(".")[0]
        with open(fname + ".kill", "w+") as f:
            f.write("Kill")
        server.main_loop(maxruns=2)
        assert server.count == 0
        import os
        os.remove(fname + ".kill")

    def test_09_centrallog(self):
        import os
        logfile = "GPAWServerLog.txt"
        if os.path.exists(logfile):
            os.remove(logfile)
        server.main_loop(maxruns=1)
        assert os.path.exists(logfile)
        with open(logfile, "r") as f:
            ldata = f.read()
        expected = "GPAW Server run 1 complete\n"
        assert ldata == expected
        os.remove(logfile)

    def test_10_gpawlogs(self):
        logfolder = "Testlogfolder"
        lserver = GPAWServer("servertest.xml", "tmpout.xml", atoms, calc, log_folder=logfolder)
        import os
        if os.path.exists(logfolder):
            for p in os.listdir(logfolder):
                os.remove(logfolder + "/" + p)
            os.rmdir(logfolder)
        numruns = 2
        lserver.main_loop(maxruns=numruns)
        assert os.path.exists(logfolder)
        files = os.listdir(logfolder)
        expecteds = ["gpaw_{}V_{}.txt".format(s, c) for s in ["plus", "minus"] for c in range(1, numruns+1)]
        for expected in expecteds:
            assert expected in files, expected + " was not found in {}".format(files)
        
    def test_11_disablelogging(self):
        logfolder = "NewTestlogfolder"
        lserver = GPAWServer("servertest.xml", "tmpout.xml", atoms, calc, should_log=False, log_folder=logfolder)
        import os
        if os.path.exists(logfolder):
            for p in os.listdir(logfolder):
                os.remove(logfolder + "/" + p)
            os.rmdir(logfolder)
        if os.path.exists("GPAWServerLog.txt"):
            os.remove("GPAWServerLog.txt")
        lserver.main_loop(maxruns=1)
        assert not os.path.exists(logfolder)
        assert not os.path.exists("GPAWServerLog.txt")


    def test_12_canpause(self):
        import os
        lserver = GPAWServer("servertest.xml", "tmpout.xml", atoms, calc, should_log=False)
        lockfile = lserver.input_file.split(".")[0] + ".lock"
        if not os.path.exists(lockfile):
            with open(lockfile, "w+") as f:
                f.write("Locked")
        lserver.sleep_period = 0.5
        lserver.main_loop(maxruns=2, maxloops=2)
        assert lserver.count == 0
        os.remove(lockfile)

    def test_13_densityhassym(self):
        lserver = GPAWServer("serversymtest.xml", "tmpout.xml", atoms, calc, should_log=False)
        lserver.main_loop(maxruns=1)
        data = lserver.xmlrw.read("tmpout.xml")

        array = data.array

        nx, ny, nz = array.shape

        for i in range(nx//2):
            assert np.allclose(array[i, :, :], array[-i, :, :])
        
    def test_14_densitysym2(self):
        lserver = GPAWServer("serversymtest2.xml", "tmpout.xml", atoms, calc, should_log=False)
        lserver.main_loop(maxruns=1)
        data = lserver.xmlrw.read("tmpout.xml")
        array = data.array
        nx, ny, nz = array.shape

        for i in  range(nx//2):
            assert np.allclose(array[i, :, :], -array[-i, :, :])

# TODO Calculations with non-zero potentials; check output

# TODO? Parallel test?

if __name__ == "__main__":
    tester = Tester()
    tester.run_tests()
