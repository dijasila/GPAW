import numpy as np
from xml.etree import cElementTree as elTree

class XMLData:
    def __init__(self, array, domain):
        self.array = array
        self.domain = domain

class XMLReaderWriter:
    
    def __init__(self):
        pass

    def _inttryparse(self, value):
        try:
            intval = int(value)
            return intval
        except Exception as e:
            print("Could not parse value: {} as int.".format(value))
            raise e

    def read(self, xml_file_path):
        import os
        if not xml_file_path.endswith(".xml"):
            xml_file_path = xml_file_path + ".xml"
        if not os.path.exists(xml_file_path):
            raise ValueError("File does not exist: {}".format(xml_file_path))
       
        for event, elem in elTree.iterparse(xml_file_path):
            if elem.tag == "grid":
                nx, ny, nz = self._process_grid(elem)
            elif elem.tag == "grid_function":
                func_vals = self._process_grid_function(elem)
            elif elem.tag == "domain":
                a, b, c = self._process_domain(elem)
            elem.clear()
        
        # Final processing of data
        # Data from WEST is in Fortran style therefore shape: nz, ny, nx + transpose
        array = np.array(func_vals).reshape(nz, ny, nx).T
        dom = np.vstack([a,b,c])
        
        data = XMLData(array, dom)

        return data

    def _process_grid(self, grid):
        nx = self._inttryparse(grid.attrib["nx"])
        ny = self._inttryparse(grid.attrib["ny"])
        nz = self._inttryparse(grid.attrib["nz"])
        return nx, ny, nz

    def _process_grid_function(self, grid_function):
        import base64
        import struct
        func_vals_bytes = grid_function.text.encode("utf-8")#func_vals_str.encode("utf-8")
        
        func_vals_decoded = base64.decodebytes(func_vals_bytes)
        assert len(func_vals_decoded) % 8 == 0, "Length was {}".format(len(func_vals_decoded))
        func_vals = [struct.unpack("d", func_vals_decoded[8*k:8*k+8]) for k in range(len(func_vals_decoded)//8)]
        return func_vals

    def _process_domain(self, domain):
        result = []
        for tag in ["a", "b", "c"]:
            partial_res = self._process_vec(domain, tag)
            result.append(partial_res)
        return result

    def _process_vec(self, elem, tag):
        vec_string = elem.attrib[tag]
        vec = vec_string.split(" ")
        vec = [d for d in vec if d != ""]
        try:
            res = np.array([float(y) for y in vec])
            return res
        except Exception as e:
            print("Failed: ", vec)
            raise e

    def write(self, data, domain, file_name):
        if not file_name.endswith(".xml"):
            file_name = file_name + ".xml"
        import base64
        import struct
        nx, ny, nz = data.shape
        a, b, c = domain[0], domain[1], domain[2]

        v = data.T.flatten()
        
        with open(file_name, "w+") as f:
            print('<?xml version="1.0" encoding="UTF-8"?>', file=f)
            print('<fpmd:function3d xmlns:fpmd="http://www.quantum-simulation.org/ns/fpmd/fpmd-1.0"', file=f)
            print(' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"', file=f)
            print(' xsi:schemaLocation="http://www.quantum-simulation.org/ns/fpmd/fpmd-1.0 function3d.xsd"', file=f)
            print(' name="delta_v">', file=f)
            print('<domain a="{} {} {}"'.format(a[0], a[1], a[2]), file=f)
            print('        b="{} {} {}"'.format(b[0], b[1], b[2]), file=f)
            print('        c="{} {} {}"/>'.format(c[0], c[1], c[2]), file=f)
            print('<grid nx="{}" ny="{}" nz="{}"/>'.format(nx, ny, nz), file=f)
            print('<grid_function type="double" nx="{}" ny="{}" nz="{}" encoding="base64">'.format(nx, ny, nz), file=f)
            
            bytestr = b"".join(struct.pack('d',t) for t in v)
            s=base64.encodebytes(bytestr).strip().decode("utf-8")
            print(s, file=f)
            print('</grid_function>', file=f)
            print('</fpmd:function3d>', file=f)

# TODO? Graceful failure?

        


