from xml.dom import minidom
import numpy as np


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

    def _read_domain_vec(self, tag, vec_id):
        vec_string = tag.attributes[vec_id].value
        x = vec_string.split(" ")
        return np.array([float(y) for y in x])

    def read(self, xml_file_path):
        import os
        if not xml_file_path.endswith(".xml"):
            xml_file_path = xml_file_path + ".xml"
        if not os.path.exists(xml_file_path):
            raise ValueError("File does not exist: {}".format(xml_file_path))
        with open(xml_file_path, "r") as f:
            data = f.read()
        xmldoc = minidom.parse(xml_file_path)
        
        # Get shape
        grid = xmldoc.getElementsByTagName("grid")
        assert len(grid) == 1, "More than one grid tag was found!"
        grid = grid[0]
        nx = self._inttryparse(grid.attributes["nx"].value)
        ny = self._inttryparse(grid.attributes["ny"].value)
        nz = self._inttryparse(grid.attributes["nz"].value)
        
        # Get values
        import base64
        import struct
        func_vals_str = xmldoc.getElementsByTagName("grid_function")
        assert len(func_vals_str) == 1, "More than one grid_function tag was found!"
        func_vals_str = func_vals_str[0].firstChild.data
        func_vals_bytes = func_vals_str.encode("utf-8")
        
        func_vals_decoded = base64.decodebytes(func_vals_bytes)
        assert len(func_vals_decoded) % 8 == 0, "Length was {}".format(len(func_vals_decoded))
        func_vals = [struct.unpack("d", func_vals_decoded[8*k:8*k+8]) for k in range(len(func_vals_decoded)//8)]
        
        array = np.array(func_vals).reshape(nz, ny, nx).T
        
        # Get domain
        domain_tag = xmldoc.getElementsByTagName("domain")
        assert len(domain_tag) == 1, "More than one domain tag was found!"
        domain_tag = domain_tag[0]
        a = self._read_domain_vec(domain_tag, "a")
        b = self._read_domain_vec(domain_tag, "b")
        c = self._read_domain_vec(domain_tag, "c")
        
        dom = np.vstack([a,b,c])
        
        
        data = XMLData(array, dom)

        return data

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

        


