from gpaw.westinterface import XMLReaderWriter, XMLData
import numpy as np
from testframework import test_method, BaseTester

xmlrw = XMLReaderWriter()
data = xmlrw.read("test.xml")




class Tester(BaseTester):
    def __init__(self):
        pass
    
    
    def test_01_data_isnotnone(self):
        assert data is not None

    def test_02_data_isnumpyarray(self):
        assert isinstance(data, XMLData)

    def test_03_data_hasrightshape(self):
        assert data.array.shape == (64, 64, 64)

    def test_04_variousshapes(self):
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

    def test_05_datahasrightorder(self):
        ldata = xmlrw.read("ordered.xml").array
        nx, ny, nz = ldata.shape
        expected = np.zeros_like(ldata)
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    expected[ix, iy, iz] = ix * ny * nz + iy * nz + iz
        assert np.allclose(expected, ldata)

    def test_06_domainisread(self):
        bohr2A = 0.529177
        expected = np.array([[16.0, 0., 0.],
                             [0., 16.0, 0.],
                             [0., 0., 16.0]]) * bohr2A
        assert np.allclose(expected, data.domain)

    def test_07_varietyofdomains(self):
        import os
        testruns = 0
        bohr2A = 0.529177
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
                                     [0., 0., 16.0]]) * bohr2A
                assert np.allclose(expected, ldata.domain)
            elif num == "2":
                testruns += 1
                expected = np.array([[16.0, 0.0, 3.0],
                                     [0.0, 16.0, 0.0],
                                     [0., 0.05, 16.0]]) * bohr2A
                assert np.allclose(expected, ldata.domain)
            elif num == "3":
                testruns += 1
                expected = np.array([[16.0, 0.0, 3.0],
                                     [0.0, 13.0, 3.0],
                                     [17., 0., 12.0]]) * bohr2A
                assert np.allclose(expected, ldata.domain)
            
        assert testruns == 3

    def test_08_writtenequalsmanual(self):
        data = np.arange(10*10*10).reshape(10,10,10)
        bohr2A = 0.529177
        domain = np.array([[16.0, 0.0, 0.0],
                           [0., 16.0, 0.0],
                           [0., 0., 16.0]]) * bohr2A
        fname = "unittest.xml"
        xmlrw.write(data, domain, fname)
        
        with open(fname, "r") as f:
            actual = f.read()
        with open("ordered.xml", "r") as f:
            expected = f.read()
        
        assert actual == expected

    def test_09_readwriteisidentity(self):
        ldata = xmlrw.read("test.xml")
        xmlrw.write(ldata.array, ldata.domain, "readwrite.xml")
        with open("readwrite.xml", "r") as f:
            actual = f.read()
        with open("test.xml", "r") as f:
            expected = f.read()
        assert actual == expected

    def test_10_writereadisidentity(self):
        array = np.random.rand(10, 10, 12)
        domain = np.random.rand(3, 3)
        xmlrw.write(array, domain, "writeread.xml")
        ldata = xmlrw.read("writeread.xml")
        
        assert np.allclose(ldata.array, array)
        assert np.allclose(ldata.domain, domain)
        
    def test_11_canreadrealdata(self):
        ldata = xmlrw.read("vext.xml")

if __name__ == "__main__":
    tester = Tester()
    import sys
    if len(sys.argv) > 1:
        number = sys.argv[1]
        tester.run_tests(number=number)
    else:
        tester.run_tests()
    
