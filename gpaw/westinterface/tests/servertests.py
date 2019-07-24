from gpaw.westinterface import GPAWServer
from testframework import BaseTester
from ase import Atoms
from gpaw import GPAW, PW
import numpy as np


class CustomFail(Exception):
    pass

class DummyAtoms:
    def __init__(self):
        self.has_run = False
        pass

    def get_potential_energy(self):
        self.has_run = True
        return 0

    def get_cell(self):
        return np.zeros((3,3))

class DummyFailAtoms:
    def __init__(self):
        self.has_run = False
        pass

    def get_potential_energy(self):
        if self.has_run:
            raise CustomFail()
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

    def get_all_electron_density(self, skip_core=True):
        return np.zeros_like(self.vext)

    def get_pseudo_density(self):
        return np.zeros_like(self.vext)

    def set(self, **kwargs):
        pass


atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=(4,4,4))
calc = GPAW(mode=PW(200), txt="testout.txt")
atoms.set_calculator(calc)
atoms.get_potential_energy()
calcfilename = "servercalc.gpw"
calc.write(calcfilename)
server = GPAWServer("servertest.xml", "serverout", calcfilename, lock_disabled=True)

class Tester(BaseTester):
    def __init__(self):
        pass

    def test_01_mainloopmaxruns(self):
        maxruns = 1
        server.main_loop(maxruns=maxruns)
        
    def test_02_readsfromfile(self):
        lserver = GPAWServer("servertest.xml", "tmpserverout", calcfilename, lock_disabled=True)
        maxruns = 1
        lserver.main_loop(maxruns=maxruns)
        import os
        fname = lserver.input_file.split(".")[0] + ".lock"
        if os.path.exists(fname):
            os.remove(fname)
        try:
            lserver.input_file = "doesnexists.xml"
            lserver.main_loop(maxruns=maxruns)
            raise NameError("Didnt read file")
        except ValueError:
            pass

    def test_03_dummycalculation(self):
        latoms = DummyAtoms()
        lcalc = DummyCalc()
        lserver = GPAWServer("servertest.xml", "tmpserverout", None, atoms=latoms, calc=lcalc, lock_disabled=True)
        lserver._get_gpaw_shape = lambda : (28, 28, 28)
        lserver.main_loop(maxruns=1)
        assert latoms.has_run

    def test_04_setsexternalpotential(self):
        server.main_loop(maxruns=1)
        assert server.calc.hamiltonian.vext is not None

    def test_05_externalpotvalues(self):
        import numpy as np
        lserver = GPAWServer("servervaluetest.xml", "tmpserverout", calcfilename, lock_disabled=True)
        data = lserver.xmlrw.read("servervaluetest.xml")
        lserver.init_hamiltonian(data.array, data.domain)
        lserver.atoms.get_potential_energy()
        ext_g = lserver.calc.hamiltonian.vext.vext_g
       
        nx, ny, nz = ext_g.shape
        expected = np.arange(nx*ny*nz).reshape(nx, ny, nz)*0.001
        assert np.allclose(expected, ext_g)
        assert not np.allclose(ext_g, 0)

    def test_06_writesoutfile(self):
        server.main_loop(maxruns=1)
        import os
        filename = server.output_file
        assert os.path.exists(filename)
        os.remove(filename)

    def test_07_outdenszeroforpotzero(self):
        lserver = GPAWServer("zerovalserver.xml", "zero", calcfilename, lock_disabled=True)
        data = lserver.xmlrw.read("zerovalserver.xml")
        assert np.allclose(data.array, 0), "Expected zero. Maxabs: {}".format(np.max(np.abs(data.array)))
        lserver.main_loop(maxruns=1)
        ldata = lserver.xmlrw.read("zero.xml")
        assert np.allclose(ldata.array, 0, atol=1e-7), "Expected zero. Maxabs: {}".format(np.max(np.abs(ldata.array)))
        import os
        os.remove("zero.xml")


    def test_08_iskillable(self):
        fname = server.input_file.split(".")[0]
        with open(fname + ".KILL", "w+") as f:
            f.write("Kill")
        server.main_loop(maxruns=2)
        assert server.count == 0
        import os
        os.remove(fname + ".KILL")

        with open(fname + ".DONE", "w+") as f:
            f.write("DONE")
        server.main_loop(maxruns=2)
        assert server.count == 0
        os.remove(fname + ".DONE")

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
        lserver = GPAWServer("servertest.xml", "tmpout.xml", calcfilename, log_folder=logfolder, lock_disabled=True)
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
        lserver = GPAWServer("servertest.xml", "tmpout.xml", calcfilename, should_log=False, log_folder=logfolder, lock_disabled=True)
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
        lserver = GPAWServer("servertest.xml", "tmpout.xml", calcfilename, should_log=False, lock_disabled=True)
        lockfile = lserver.input_file.split(".")[0] + ".lock"
        if not os.path.exists(lockfile):
            with open(lockfile, "w+") as f:
                f.write("Locked")
        lserver.sleep_period = 0.5
        lserver.main_loop(maxruns=2, maxloops=2)
        assert lserver.count == 0
        os.remove(lockfile)

    def test_13_densityhassym(self):
        lserver = GPAWServer("serversymtest.xml", "tmpout.xml", calcfilename, should_log=False, lock_disabled=True)
        lserver.main_loop(maxruns=1)
        data = lserver.xmlrw.read("tmpout.xml")

        array = data.array

        nx, ny, nz = array.shape

        for i in range(nx//2):
            assert np.allclose(array[i, :, :], array[-i, :, :])
        
    def test_14_densitysym2(self):
        return "Skipped"
        lserver = GPAWServer("serversymtest2.xml", "tmpout2.xml", calcfilename, should_log=False, lock_disabled=True)
        larray = lserver.xmlrw.read("serversymtest2.xml").array
        lserver.main_loop(maxruns=1)
        ldata = lserver.xmlrw.read("tmpout2.xml")
        array = ldata.array
        nx, ny, nz = array.shape

        for i in range(nx//2):
            assert np.allclose(array[i, :, :], -array[-i, :, :], atol=1e-5, rtol=1e-2), "Failed at i = {}. {}".format(i, np.max(np.abs(array[i, :, :] + array[-i, :, :])))

    def test_15_retrylogic(self):
        latoms = DummyFailAtoms()
        lserver = GPAWServer("servertest.xml", "tmpout.xml", None, atoms=latoms, calc=calc, should_log=False, lock_disabled=True)
        try:
            lserver.main_loop(maxruns=1)
        except CustomFail:
            pass
        import os
        failname = "servertest.FAILED"
        assert os.path.exists(failname)
        os.remove(failname)

    def test_16_writeslock(self):
        lserver = GPAWServer("servertest.xml", "tmpout.xml", "servercalc.gpw")
        lserver.main_loop(maxruns=1, maxloops=1)
        import os
        lock_file = lserver.input_file.split(".")[0] + ".lock"
        assert os.path.exists(lock_file)

    def test_17_interpolationisid(self):
        array = np.random.rand(10,10,10)
        new_array = server.fft_interpolation(array, (10,10,10))

        assert np.allclose(array, new_array)

    def test_18_getindices_iswithinrange(self):
        reg_in = [tuple([np.random.randint(100)]*3) for _ in range(10)]
        reg_out = [tuple([np.random.randint(100)]*3) for _ in range(10)]
        rand_in = [tuple(np.random.randint(100, size=3)) for _ in range(10)]
        rand_out = [tuple(np.random.randint(100, size=3)) for _ in range(10)]
        
        def ltest(pair):
            in_shape, out_shape = pair

            in_xs, in_ys, in_zs, out_xs, out_ys, out_zs = server._get_interpolation_indices(in_shape, out_shape)
            
            def a(a,b):
                assert len(a) == len(b)

            a(in_xs, in_ys)
            a(in_xs, in_ys)
            a(in_xs, in_zs)
            a(in_xs, out_xs)
            a(in_xs, out_ys)
            a(in_xs, out_zs)

            for j in range(len(in_xs)):
                ix = in_xs[j]
                iy = in_ys[j]
                iz = in_zs[j]
                
                assert ix >= 0 and ix < in_shape[0]
                assert iy >= 0 and iy < in_shape[1]
                assert iz >= 0 and iz < in_shape[2]
                
                ix = out_xs[j]
                iy = out_ys[j]
                iz = out_zs[j]
                assert ix >= 0 and ix < out_shape[0]
                assert iy >= 0 and iy < out_shape[1]
                assert iz >= 0 and iz < out_shape[2]
                

        for p in zip(reg_in, reg_out):
            ltest(p)

        for p in zip(rand_in, rand_out):
            ltest(p)

    def test_19_normispreserved(self):
        dx = 0.1
        dv = dx**3
        array = np.ones((10,10,10))
        vol = np.sum(array * dv)
        new_array = server.fft_interpolation(array, (20, 20, 20))
        ndx = 0.05
        ndv = ndx**3
        nvol = np.sum(new_array*ndv)
        
        assert np.allclose(vol, nvol), "Old vol: {}, new vol: {}".format(vol, nvol)

    def test_20_normispreserved_nonreggrid(self):
        dx = 0.1
        dy = 0.2
        dv = dx**2*dy
        arr = np.ones((10, 5, 10))
        vol = np.sum(arr*dv)
        new_arr = server.fft_interpolation(arr, (10, 10, 10))
        ndv = dx**3
        nvol = np.sum(new_arr*ndv)
        
        assert np.allclose(vol, nvol), "Old vol: {}, new vol: {}".format(vol, nvol)

    def test_21_normispreserved_tosmaller(self):
        dx = 0.1
        dv = dx**3
        vals = np.arange(0, 8000).reshape(20,20,20)
        arr = np.sin(0.001 * vals)
        norm = np.sum(dv * arr)

        new_arr = server.fft_interpolation(arr, (10,10,10))
        ndx = 0.2
        ndv = ndx**3
        new_norm = np.sum(ndv * new_arr)
        
        assert np.allclose(norm, new_norm)

    def test_22_fftinter_valuesareclose(self):
        grid_size = 100
        xs = np.linspace(0, 1, grid_size)
        X, Y, Z = np.meshgrid(xs, xs, xs)

        for _ in range(3):
            a = np.random.rand()*2+0.1
            b = np.random.rand()*2+0.1
            c = np.random.rand()*2+0.1
            func = np.sin(a*np.pi*X + b*np.pi*Y + c*np.pi*Z)
            assert not np.allclose(func, 0)
            n_func = server.fft_interpolation(func, (2*grid_size, 2*grid_size, 2*grid_size))
            assert not np.allclose(n_func, 0)
            o_func = n_func[::2, ::2, ::2]
            assert np.allclose(func-o_func, 0)

    def test_23_fftinter_valuesareclosenonreg(self):
        grid_size = 100
        xs = np.linspace(0, 1, grid_size)
        X, Y, Z = np.meshgrid(xs, xs, xs)

        for _ in range(3):
            a = np.random.rand()*2+0.1
            b = np.random.rand()*2+0.1
            c = np.random.rand()*2+0.1
            func = np.sin(a*np.pi*X + b*np.pi*Y + c*np.pi*Z)
            assert not np.allclose(func, 0)
            n_func = server.fft_interpolation(func, (2*grid_size, 2*grid_size, 3*grid_size))
            assert not np.allclose(n_func, 0)
            o_func = n_func[::2, ::2, ::3]
            assert np.allclose(func-o_func, 0)

        

    def cleanup_lockfile(self):
        import os
        locks = [fname for fname in os.listdir() if fname.endswith(".lock")]
        for lock in locks:
            os.remove(lock)
    
    def cleanup_failfile(self):
        import os
        fails = [fname for fname in os.listdir() if fname.endswith(".FAILED")]
        for fail in fails:
            os.remove(fail)
        
if __name__ == "__main__":
    import sys
    tester = Tester()
    if len(sys.argv) > 1:
        number = sys.argv[1]
        tester.run_tests(number=number)
    else:
        tester.run_tests()
