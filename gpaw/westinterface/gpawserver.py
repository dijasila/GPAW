from gpaw.external import ExternalPotential

class ExternalField(ExternalPotential):
    def __init__(self, data):
        '''
        Data is type gpaw.westinterface.xmlreaderwriter.XMLData
        '''
        self.values = data.array
        self.domain = data.domain

    def calculate_potential(self, gd):
        assert gd.get_grid_point_coordinates().shape[1:] == self.values.shape, "Grid shape {} does not match potential data shape {}".format(gd.get_grid_point_coordinates().shape[1:], self.values.shape)
        self.vext_g = self.values

    def get_name(self):
        return "WESTExternalField"



class GPAWServer:
    '''
    Class that handles the calculation-loop with the WEST code and the GPAW calculation itself

    '''
    def __init__(self, input_file, output_file, atoms, calc):
        from gpaw.westinterface import XMLReaderWriter
        self.xmlrw = XMLReaderWriter()
        self.input_file = input_file
        self.output_file = output_file
        self.atoms = atoms
        self.calc = calc
        pass

    def main_loop(self, maxruns=-1):
        # Init stuff in Calc
        self.atoms.get_potential_energy()

        count = 0
        while True:
            count += 1
            if count > maxruns and maxruns >= 0:
                break

            data = self.xmlrw.read(self.input_file)
            
            # Set External potential
            external = ExternalField(data)
            self.calc.hamiltonian = None
            self.calc.parameters.external = external
            self.calc.initialize()
            
            self.atoms.get_potential_energy()

