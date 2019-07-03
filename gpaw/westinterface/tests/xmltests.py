from gpaw.westinterface.xmlreaderwriter import XMLReaderWriter, XMLData
from functools import wraps
import numpy as np
def colored(msg, color):
    end = "\033[0m"
    if color == "ok":
        pre = "\033[92m"
    elif color == "fail":
        pre = "\033[91m"
    else:
        raise ValueError("Color option not recognized")
    return pre + msg + end

def test_method(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        msg = f"Running {f.__name__}".ljust(50) + "..."
        print(msg, end="")
        try:
            res = f(*args, **kwargs)
            print(colored("success", "ok"))
            return res
        except Exception as e:
            print(colored("failure", "fail"))
            raise e
    return wrapped

xmlrw = XMLReaderWriter()
data = xmlrw.read("test.xml")




class Tester:
    def __init__(self):
        pass
    
    def run_tests(self):
        my_test_methods = [m for m in dir(self) if callable(getattr(self, m)) and m.startswith("test_")]
        for mname in my_test_methods:
            m = getattr(self, mname)
            m()
    @test_method
    def test_1_data_isnotnone(self):
        assert data is not None

    @test_method
    def test_2_data_isnumpyarray(self):
        assert isinstance(data, XMLData)

    @test_method
    def test_3_data_hasrightshape(self):
        assert data.array.shape == (64, 64, 64)

    @test_method
    def test_4_variousshapes(self):
        import os
        maxcount = 10
        count = 0
        for fname in os.listdir():
            if not fname.endswith(".xml"):
                continue
            if not fname.count("_") == 3:
                continue
            count += 1
            if count > maxcount:
                break
            ldata = xmlrw.read(fname).array
            _, nx, ny, nz = fname.split("_")
            nz, _ = nz.split(".")
            expected = (int(nx), int(ny), int(nz))
            assert ldata.shape == expected

    @test_method
    def test_5_datahasrightorder(self):
        ldata = xmlrw.read("ordered.xml").array
        nx, ny, nz = ldata.shape
        expected = np.zeros_like(ldata)
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    expected[ix, iy, iz] = ix * ny * nz + iy * nz + iz
        assert np.allclose(expected, ldata)

    @test_method
    def test_6_domainisread(self):
        expected = np.array([[16.0, 0., 0.],
                             [0., 16.0, 0.],
                             [0., 0., 16.0]])
        assert np.allclose(expected, data.domain)

    @test_method
    def test_7_varietyofdomains(self):
        import os
        testruns = 0
        for fname in os.listdir():
            if not fname.endswith(".xml") or not fname.startswith("domtest"):
                continue
            ldata = xmlrw.read(fname)
            _, num = fname.split("_")
            num, _ = num.split(".")
            if num == "1":
                testruns += 1
                expected = np.array([[16.0, 0.0, 5.0],
                                     [0.0, 16.0, 0.0],
                                     [0., 0., 16.0]])
                assert np.allclose(expected, ldata.domain)
            elif num == "2":
                testruns += 1
                expected = np.array([[16.0, 0.0, 3.0],
                                     [0.0, 16.0, 0.0],
                                     [0., 0.05, 16.0]])
                assert np.allclose(expected, ldata.domain)
            elif num == "3":
                testruns += 1
                expected = np.array([[16.0, 0.0, 3.0],
                                     [0.0, 13.0, 3.0],
                                     [17., 0., 12.0]])
                assert np.allclose(expected, ldata.domain)
            
        assert testruns == 3

    @test_method
    def test_8_writtenequalsmanual(self):
        data = np.arange(10*10*10).reshape(10,10,10)
        domain = np.array([[16.0, 0.0, 0.0],
                           [0., 16.0, 0.0],
                           [0., 0., 16.0]])
        fname = "unittest.xml"
        xmlrw.write(data, domain, fname)
        
        with open(fname, "r") as f:
            actual = f.read()
        with open("ordered.xml", "r") as f:
            expected = f.read()
        
        assert actual == expected

    @test_method
    def test_9_readwriteisidentity(self):
        ldata = xmlrw.read("test.xml")
        xmlrw.write(ldata.array, ldata.domain, "readwrite.xml")
        with open("readwrite.xml", "r") as f:
            actual = f.read()
        with open("test.xml", "r") as f:
            expected = f.read()
        assert actual == expected

    @test_method
    def test_x10_writereadisidentity(self):
        array = np.random.rand(10, 10, 12)
        domain = np.random.rand(3, 3)
        xmlrw.write(array, domain, "writeread.xml")
        ldata = xmlrw.read("writeread.xml")
        
        assert np.allclose(ldata.array, array)
        assert np.allclose(ldata.domain, domain)
        
if __name__ == "__main__":
    tester = Tester()
    tester.run_tests()
