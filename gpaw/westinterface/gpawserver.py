from gpaw.external import ExternalPotential
from gpaw import mpi
import numpy as np
from ase.parallel import parprint

class ExternalField(ExternalPotential):
    def __init__(self, array, domain):
        '''
        Data is type gpaw.westinterface.xmlreaderwriter.XMLData
        '''
        self.values = array
        self.domain = domain

    def calculate_potential(self, gd):
        gd1 = gd.new_descriptor(comm=mpi.serial_comm)
        grid_shape = gd1.get_grid_point_coordinates().shape[1:]
        data_shape = self.values.shape
        assert grid_shape == data_shape, "Could not calculate potential. Grid shape {} does not match data shape: {}".format(grid_shape, data_shape)
        # assert gd.get_grid_point_coordinates().shape[1:] == self.values.shape, "Grid shape {} does not match potential data shape {}. Gd with all pts has shape: {}".format(gd.get_grid_point_coordinates().shape[1:], self.values.shape, gd1.get_grid_point_coordinates().shape[1:])
        self.vext_g = np.zeros(gd.get_grid_point_coordinates().shape[1:])
        if mpi.rank != 0:
            self.values = None
        gd.distribute(self.values, self.vext_g)

    def get_name(self):
        return "WESTExternalField"


# TODO: Add interpolation of potential and density?
# TODO: OR calculate h-parameter for given cutoff?
class GPAWServer:
    '''
    Class that handles the calculation-loop with the WEST code and the GPAW calculation itself

    '''
    def __init__(self, input_file, output_file, calc_file, should_log=True, log_folder="GPAWServerLogsFolder", atoms=None, calc=None, lock_disabled=False):
        from gpaw import GPAW
        if calc_file is not None:
            try:
                gpw_infofile = calc_file.split(".")[0] + "_info.txt"
                assert gpw_infofile != input_file and gpw_infofile != output_file
                calc = GPAW(calc_file, txt=gpw_infofile)
            except Exception as e:
                self.send_signal("FAILED", str(e))
                raise e
            atoms = calc.atoms
            atoms.set_calculator(calc)
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
        self.lock_disabled = lock_disabled

    def main_loop(self, maxruns=-1, maxloops=-1):
        # Init stuff in Calc
        success = self.get_potential_energy()
        if not success:
            return

        # Get full grid shape for interpolation - WEST and GPAW may use slightly different grids.
        gpaw_shape = self._get_gpaw_shape()
        self.calc.set(convergence={"density":1e-7})
        self.calc.set(fixdensity=True)
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
            mpi.world.barrier()

            # Read and transform data to GPAW-compatible shape
            data = self.xmlrw.read(self.input_file)
            west_shape = data.array.shape
            data.array = self.fft_interpolation(data.array, gpaw_shape)
            data = self.downscale_potential(data)

            # V = 0 calculation
            # self.init_hamiltonian(np.zeros_like(data.array), data.domain)
            # success = self.get_potential_energy()
            # if not success:
            #     break
            # density0t = self.calc.get_pseudo_density()
            # density0 = self.calc.get_all_electron_density()
            # density0wocore = self.calc.get_all_electron_density(skip_core=True)
            # only_core = density0 - density0wocore

            # -V calculation
            if self.should_log and mpi.rank == 0:
                self.calc.set(txt=self.log_folder + "/" + "gpaw_minusV_{}.txt".format(self.count))
            self.init_hamiltonian(-data.array, data.domain)

            success = self.get_potential_energy()
            if not success:
                break

            densitymt = self.calc.get_pseudo_density()
            #densitym = self.calc.get_all_electron_density(skip_core=False)
            #densitymwocore = self.calc.get_all_electron_density(skip_core=True)


            # Calculate density integrals
            #dv = self.atoms.get_volume() / self.calc.get_number_of_grid_points().prod()
            #It = densitymt.sum() * dv
            #I2 = densitymwocore.sum() * dv / 2**3
            #I = densitym.sum() * dv / 2**3
            #parprint("Integrated Minus density tilde: ", It)
            #parprint("Integrated Minus density w/o core: ", I2)
            #parprint("Integrated Minus density: ", I)

            # +V calculation
            if self.should_log and mpi.rank == 0:
                self.calc.set(txt=self.log_folder + "/" + "gpaw_plusV_{}.txt".format(self.count))
            self.init_hamiltonian(data.array, data.domain)
            
            success = self.get_potential_energy()
            if not success:
                break

            densitypt = self.calc.get_pseudo_density()
            #densityp = self.calc.get_all_electron_density(skip_core=False)
            #densitypwocore = self.calc.get_all_electron_density(skip_core=True)

            # Calculate density integrals
            #dv = self.atoms.get_volume() / self.calc.get_number_of_grid_points().prod()
            #It = densitypt.sum() * dv
            #I2 = densitypwocore.sum() * dv / 2**3
            #I = densityp.sum() * dv / 2**3
            #parprint("Integrated Plus density tilde: ", It)
            #parprint("Integrated Plus density w/o core: ", I2)
            #parprint("Integrated Plus density: ", I)
            

            # Scale densities to have same maxabs
            # mdpt = np.max(np.abs(densitypt))
            # mdmt = np.max(np.abs(densitymt))
            # mdp = np.max(np.abs(densityp))
            # mdm = np.max(np.abs(densitym))
            # mdpwoc = np.max(np.abs(densitypwocore))
            # mdmwoc = np.max(np.abs(densitymwocore))
            
            # densitypt *= mdmt / mdpt
            # densityp *= mdm / mdp
            # densitypwocore *= mdmwoc / mdpwoc

            # Density response
            #density = (densityp - densitym) / 2.0
            #density = self.upscale_density(density)
            
            #densitywocore = (densitypwocore - densitymwocore) / 2.0
            #densitywocore = self.upscale_density(densitywocore)

            densityt = (densitypt - densitymt) / 2.0
            densityt = self.upscale_density(densityt)
            
            
            # Write to output file
            domain = self.atoms.get_cell()
            #density = self.fft_interpolation(density, west_shape)
            #self.xmlrw.write(density, domain, self.output_file)
            
            # Write other calc types
            #self.xmlrw.write(densitywocore, domain, "AEwocore" + self.output_file)
            self.xmlrw.write(densityt, domain, "Pseudo" + self.output_file)

            self.log("GPAW Server run {} complete".format(self.count))


            if not self.lock_disabled:
                self.send_signal("lock")
            
            # Barrier to ensure some ranks dont start next loop
            mpi.world.barrier()

    def init_logging(self):
        import os
        if mpi.rank != 0:
            return

        if not self.should_log:
            self.calc.set(txt=None)
            return
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)

    def _get_gpaw_shape(self):
        gd1 = self.calc.wfs.gd.new_descriptor(comm=mpi.serial_comm)
        gd1 = gd1.refine()
        return gd1.get_grid_point_coordinates().shape[1:]

    def init_hamiltonian(self, array, domain):
        external = ExternalField(array, domain)
        self.calc.hamiltonian = None
        #self.calc.density = None
        self.calc.parameters.external = external
        self.calc.initialize()

    def get_potential_energy(self):
        # Apply retry logic to GPAW calculation
        try:
            self.atoms.get_potential_energy()
            return True
        except Exception as e:
            self.send_signal("FAILED", "Could not get potential energy:\n {}".format(str(e)))
            raise e
            # return False
            

    def should_kill(self):
        from pathlib import Path
        base = self.input_file.split(".")[0]
        fname = base + ".KILL"
        p = Path(fname)
        shouldkill = p.exists()
        
        fname2 = base + ".DONE"
        p2 = Path(fname2)
        isdone = p2.exists()
        
        return shouldkill or isdone

    def is_locked(self):
        from pathlib import Path
        fname = self.input_file.split(".")[0] + ".lock"
        p = Path(fname)
        return p.exists()

    def log(self, msg):
        if not self.should_log:
            return
        if mpi.rank != 0:
            return

        from pathlib import Path
        fname = "GPAWServerLog.txt"
        with open(fname, "a") as f:
            f.write(msg + "\n")
    
    def send_signal(self, msg, extra=None):
        if mpi.rank != 0:
            return
        extra = extra or msg
        msg = msg if msg.startswith(".") else "." + msg
        fname = self.input_file.split(".")[0] + msg
        with open(fname, "w+") as f:
            f.write(extra)

    def downscale_potential(self, xmldata):
        maxval = 1e-3
        maxval_data = np.max(np.abs(xmldata.array))
        self.scale_factor = maxval / maxval_data if not np.allclose(maxval_data, 0) else 1
        xmldata.array = xmldata.array * self.scale_factor
        return xmldata

    def upscale_density(self, density):
        density /= self.scale_factor
        return density

    def fft_interpolation(self, array, out_shape):
        in_shape = array.shape
        if in_shape == out_shape:
            return array
        f_array = np.fft.fftn(array)
        f_array = np.fft.fftshift(f_array)


        out_array = np.zeros(out_shape, dtype=np.complex128)

        in_xs, in_ys, in_zs, out_xs, out_ys, out_zs = self._get_interpolation_indices(in_shape, out_shape)
    
        out_array[out_xs, out_ys, out_zs] = f_array[in_xs, in_ys, in_zs]
        
        #x_indices = range(xstart-xout//2, xstart+xout//2)
        #out_array[x_indices, ystart - yout//2:ystart + yout//2, zstart - zout//2:zstart+zout//2] = f_array[x_indices, ystart - yout//2:ystart + yout//2, zstart - zout//2:zstart+zout//2]

        factor = (np.array(out_shape)/np.array(in_shape)).prod()

        return np.fft.ifftn(np.fft.ifftshift(out_array)).real * factor

    def _get_interpolation_indices(self, in_shape, out_shape):
        xw, yw, zw = [min(in_shape[i], sh) for i, sh in enumerate(out_shape)]
        mids = lambda sha: [sh//2 for sh in sha]
        xm, ym, zm = mids(out_shape)
        ixm, iym, izm = mids(in_shape)

        def a(w, m):
            if w == 1:
                return [m]
            else:
                return np.arange(m - w//2, m + w//2)
        
        from itertools import product
        in_inds = np.array(list(product(a(xw, ixm), a(yw, iym), a(zw, izm))))
        out_inds = np.array(list(product(a(xw, xm), a(yw, ym), a(zw, zm))))
        
        if len(in_inds) == 0 or len(out_inds) == 0:
            return [], [], [], [], [], []
        
        in_xs, in_ys, in_zs = in_inds[:, 0], in_inds[:, 1], in_inds[:, 2]
        out_xs, out_ys, out_zs = out_inds[:, 0], out_inds[:, 1], out_inds[:, 2]

        return in_xs, in_ys, in_zs, out_xs, out_ys, out_zs

if __name__ == "__main__":
    import sys
    infile, outfile, calcfile = sys.argv[1:4]
    server = GPAWServer(infile, outfile, calcfile)
    server.main_loop()
