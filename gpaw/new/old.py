import ase.io.ulm as ulm
import gpaw
import numpy as np
from ase.io.trajectory import read_atoms, write_atoms
from ase.units import Bohr, Ha
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.calculation import DFTCalculation, DFTState
from gpaw.new.density import Density
from gpaw.new.input_parameters import InputParameters
from gpaw.new.potential import Potential
# from gpaw.new.wave_functions import IBZWaveFunctions
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


def read_gpw(filename, log, parallel):
    log(f'Reading from {filename}')
    reader = ulm.Reader(filename)
    atoms = read_atoms(reader.atoms)

    kwargs = reader.parameters.asdict()
    kwargs['parallel'] = parallel
    params = InputParameters(kwargs)
    print(params)
    builder = DFTComponentsBuilder(atoms, params)

    array_sR = reader.density.density
    grid = builder.grid.new(comm=None)
    nt_sR = grid.empty(len(array_sR))
    nt_sR.data[:] = array_sR
    nt_sR = nt_sR.distribute(grid=builder.grid)

    array_sx = reader.density.atomic_density_matrices
    atom_array_layout = AtomArraysLayout([(setup.ni, setup.ni)
                                          for setup in setups],
                                         atomdist=atomdist)
    D_asii = atom_array_layout.empty(ndens + nmag)
    for a, D_sii in D_asii.items():
        D_sii[:] = unpack2(setups[a].initialize_density_matrix(f_asi[a]))

    density = Density.from_data_and_setups(nt_sR, D_asii, setups, charge)
    potential = Potential(...)
    ibzwfs = ...

    calculation = DFTCalculation(DFTState(ibzwfs, density, potential),
                                 builder.setups,
                                 builder.create_scf_loop(pot_calc),
                                 pot_calc)
    results = reader.results.asdict()
    if results:
        log(f'Read {", ".join(sorted(results))}')

    calculation.results = results
    return calculation, params
