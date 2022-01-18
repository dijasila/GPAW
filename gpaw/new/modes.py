from __future__ import annotations

from ase.units import Bohr
from gpaw.mpi import MPIComm, serial_comm
from gpaw.utilities.gpts import get_number_of_grid_points
from gpaw.core import UniformGrid


class Mode:
    name: str
    interpolation: str

    def __init__(self, force_complex_dtype=False):
        self.force_complex_dtype = force_complex_dtype

    def check_cell(self, cell):
        number_of_lattice_vectors = cell.rank
        if number_of_lattice_vectors < 3:
            raise ValueError(
                'GPAW requires 3 lattice vectors.  '
                f'Your system has {number_of_lattice_vectors}.')

    def create_uniform_grid(self,
                            h: float | None,
                            gpts,
                            cell,
                            pbc,
                            symmetry,
                            comm: MPIComm = serial_comm) -> UniformGrid:
        cell = cell / Bohr
        if h is not None:
            h /= Bohr

        realspace = (self.name != 'pw' and self.interpolation != 'fft')
        if not realspace:
            pbc = (True, True, True)

        if gpts is not None:
            if h is not None:
                raise ValueError("""You can't use both "gpts" and "h"!""")
            size = gpts
        else:
            size = get_number_of_grid_points(cell, h, self, realspace,
                                             symmetry.symmetry)
        return UniformGrid(cell=cell, pbc=pbc, size=size, comm=comm)
