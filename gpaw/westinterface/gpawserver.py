from gpaw.external import ExternalPotential
from gpaw import mpi

class ExternalField(ExternalPotential):
    def __init__(self, array, domain):
        '''
        Data is type gpaw.westinterface.xmlreaderwriter.XMLData
        '''
        self.values = array
        self.domain = domain

    def calculate_potential(self, gd):
        assert gd.get_grid_point_coordinates().shape[1:] == self.values.shape, "Grid shape {} does not match potential data shape {}".format(gd.get_grid_point_coordinates().shape[1:], self.values.shape)
        self.vext_g = self.values

    def get_name(self):
        return "WESTExternalField"



class GPAWServer:
    '''
    Class that handles the calculation-loop with the WEST code and the GPAW calculation itself

    '''
    def __init__(self, input_file, output_file, atoms, calc, should_log=True, log_folder="GPAWServerLogsFolder"):
        from gpaw.westinterface import XMLReaderWriter
        self.xmlrw = XMLReaderWriter()
        self.input_file = input_file
        self.output_file = output_file if output_file.endswith(".xml") else output_file + ".xml"
        self.atoms = atoms
        self.calc = calc
        self.count = 0
        self.should_log = should_log
        self.log_folder = log_folder
        self.loopcount = 0
        self.sleep_period = 3

    def main_loop(self, maxruns=-1, maxloops=-1):
        # Init stuff in Calc
        self.atoms.get_potential_energy()
        self.calc.set(convergence={"density":1e-8})
        self.init_logging()
        self.count = 0
        self.loopcount = 0
        while True:
            if self.should_kill():
                break
            self.loopcount += 1
            if maxloops >= 0 and self.loopcount > maxloops:
                break

            if self.is_locked():
                import time
                time.sleep(self.sleep_period)
                continue


            self.count += 1
            if maxruns >= 0 and self.count > maxruns:
                break

            data = self.xmlrw.read(self.input_file)
            
            
            # +V calculation
            if self.should_log:
                self.calc.set(txt=self.log_folder + "/" + "gpaw_plusV_{}.txt".format(self.count))
            self.init_hamiltonian(data.array, data.domain)
            self.atoms.get_potential_energy()

            domain = self.atoms.get_cell()
            densityp = self.calc.get_pseudo_density()

            # -V calculation
            if self.should_log:
                self.calc.set(txt=self.log_folder + "/" + "gpaw_minusV_{}.txt".format(self.count))
            self.init_hamiltonian(data.array, data.domain)
            self.atoms.get_potential_energy()
            densitym = self.calc.get_pseudo_density()

            density = (densityp - densitym) / 2.0

            
            # Write to output file
            self.xmlrw.write(density, domain, self.output_file)
            
            self.log("GPAW Server run {} complete".format(self.count))

    def init_logging(self):
        import os
        if not self.should_log:
            self.calc.set(txt=None)
            return
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)

    def init_hamiltonian(self, array, domain):
        external = ExternalField(array, domain)
        self.calc.hamiltonian = None
        self.calc.density = None
        self.calc.parameters.external = external
        self.calc.initialize()

    def should_kill(self):
        from pathlib import Path
        fname = self.input_file.split(".")[0] + ".kill"
        p = Path(fname)
        shouldkill = p.exists()
        return shouldkill

    def is_locked(self):
        from pathlib import Path
        fname = self.input_file.split(".")[0] + ".lock"
        p = Path(fname)
        return p.exists()
    def log(self, msg):
        if not self.should_log:
            return
        if not mpi.rank == 0:
            return

        from pathlib import Path
        fname = "GPAWServerLog.txt"
        with open(fname, "a") as f:
            f.write(msg + "\n")
