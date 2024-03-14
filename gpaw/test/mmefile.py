from pathlib import Path

from gpaw.mpi import serial_comm, world
from gpaw.new.ase_interface import GPAW
from gpaw.nlopt.matrixel import make_nlodata
from gpaw.test.cachedfilehandler import CachedFilesHandler
from gpaw.test.gpwfile import GPWFiles, response_band_cutoff


class MMEFiles(CachedFilesHandler):
    """Create files that store momentum matrix elements."""

    def __init__(self, folder: Path, gpw_files: GPWFiles):
        super().__init__(folder, '.npz')
        self.gpw_files = gpw_files

    def _calculate_and_write(self, name, work_path):
        calc = GPAW(self.gpw_files[name], communicator=serial_comm)
        nb = response_band_cutoff[
            name if not name.endswith('_spinpol') else name[:-8]]
        nlodata = make_nlodata(calc, ni=0, nf=nb, comm=world)
        nlodata.write(work_path)
