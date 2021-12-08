import numpy as np
import ase.io.ulm as ulm
import gpaw
from ase.io.trajectory import write_atoms, read_atoms
from ase.units import Bohr, Ha
from gpaw.new.calculation import DFTCalculation
from gpaw.new.density import Density
from gpaw.utilities import pack


class OldStuff:
    def get_pseudo_wave_function(self, n):
        return self.calculation.ibzwfs[0].wave_functions.data[n]

    def write(self, filename, mode=''):
        """Write calculator object to a file.

        Parameters
        ----------
        filename
            File to be written
        mode
            Write mode. Use ``mode='all'``
            to include wave functions in the file.
        """
        self.log(f'Writing to {filename} (mode={mode!r})\n')

        write_gpw(filename, self.atoms, self.params,
                  self.calculation, skip_wfs=mode != 'all')


def write_gpw(filename: str,
              atoms,
              params,
              calculation: DFTCalculation,
              skip_wfs: bool = True) -> None:

    world = params.parallel['world']

    if world.rank == 0:
        writer = ulm.Writer(filename, tag='gpaw')
    else:
        writer = ulm.DummyWriter()

    with writer:
        writer.write(version=4,
                     gpaw_version=gpaw.__version__,
                     ha=Ha,
                     bohr=Bohr)

        write_atoms(writer.child('atoms'), atoms)
        writer.child('results').write(**calculation.results)
        writer.child('parameters').write(**params.params)

        density = calculation.state.density
        dms = density.density_matrices.collect()

        N = sum(i1 * (i1 + 1) // 2 for i1, i2 in dms.layout.shapes)
        D = np.zeros((density.ncomponents, N))

        n1 = 0
        for D_iis in dms.values():
            i1 = len(D_iis)
            n2 = n1 + i1 * (i1 + 1) // 2
            for s, D_ii in enumerate(D_iis.T):
                D[s, n1:n2] = pack(D_ii)
            n1 = n2

        writer.child('density').write(
            density=density.nt_s.collect().data * Bohr**-3,
            atomic_density_matrices=D)

        calculation.state.potential.write(writer.child('hamiltonian'))
        calculation.state.ibzwfs.write(writer.child('wave_functions'),
                                       skip_wfs)

    world.barrier()


def read_gpw(filename, log):
    log(f'Reading from {filename}')
    reader = ulm.Reader(filename)
    atoms = read_atoms(reader.atoms)

    builder = atoms, ...

    grid = builder, ...

    results = reader.results.asdict()
    if results:
        log('Read {}'.format(', '.join(sorted(results))))

    density = Density.read(reader, grid)
    potential = ...
    ibzwfs = ...

    calculation = DFTCalculation(ibzwfs, density, potential)
    # calculation.results = results
    return calculation
