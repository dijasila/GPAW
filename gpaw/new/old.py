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
from gpaw.utilities import pack, unpack2
from gpaw.core.atom_arrays import AtomArraysLayout


class OldStuff:
    def get_pseudo_wave_function(self, n):
        return self.calculation.ibzwfs[0].wave_functions.data[n]

    def get_atomic_electrostatic_potentials(self):
        _, _, Q_aL = self.calculation.pot_calc.calculate(
            self.calculation.state.density)
        Q_aL = Q_aL.gather()
        return Q_aL.data[::9] * (Ha / (4 * np.pi)**0.5)

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

    world = parallel['world']

    reader = ulm.Reader(filename)
    atoms = read_atoms(reader.atoms)

    kwargs = reader.parameters.asdict()
    kwargs['parallel'] = parallel
    params = InputParameters(kwargs)

    builder = DFTComponentsBuilder(atoms, params)
    kpt_band_comm = builder.communicators['D']

    if world.rank == 0:
        nt_sR_array = reader.density.density
        vt_sR_array = reader.hamiltonian.potential
        D_sap_array = reader.density.atomic_density_matrices
        D_saii_array = unpack_d(D_sap_array, builder.setups)
        dH_sap_array = reader.hamiltonian.atomic_hamiltonian_matrices
        dH_saii_array = unpack_d(dH_sap_array, builder.setups)
    else:
        nt_sR_array = None
        vt_sR_array = None
        D_saii_array = None
        dH_saii_array = None

    nt_sR = builder.grid.empty(builder.ncomponents)
    vt_sR = builder.grid.empty(builder.ncomponents)

    atom_array_layout = AtomArraysLayout([(setup.ni, setup.ni)
                                          for setup in builder.setups],
                                         atomdist=builder.atomdist)
    D_asii = atom_array_layout.empty(builder.ncomponents)
    dH_asii = atom_array_layout.empty(builder.ncomponents)

    if kpt_band_comm.rank == 0:
        nt_sR.scatter_from(nt_sR_array)
        vt_sR.scatter_from(vt_sR_array)
        D_asii.scatter_from(D_saii_array)
        dH_asii.scatter_from(dH_saii_array)

    kpt_band_comm.broadcast(nt_sR.data, 0)
    kpt_band_comm.broadcast(vt_sR.data, 0)
    kpt_band_comm.broadcast(D_asii.data, 0)
    kpt_band_comm.broadcast(dH_asii.data, 0)

    density = Density.from_data_and_setups(nt_sR, D_asii,
                                           builder.params.charge,
                                           builder.setups)
    potential = Potential(vt_sR, dH_asii, {})
    ibzwfs = ...

    calculation = DFTCalculation(
        DFTState(ibzwfs, density, potential),
        builder.setups,
        None,
        pot_calc=builder.create_potential_calculator())

    results = reader.results.asdict()
    if results:
        log(f'Read {", ".join(sorted(results))}')

    calculation.results = results
    return calculation, params


def unpack_d(D_sap_array, setups):
    ns = len(D_sap_array)
    naii = sum(setup.ni**2 for setup in setups)
    D_saii_array = np.empty((ns, naii))
    nap1 = 0
    naii1 = 0
    for setup in setups:
        nap2 = nap1 + (setup.ni * (setup.ni + 1)) // 2
        naii2 = naii1 + setup.ni**2
        D_saii_array[:, naii1:naii2] = unpack2(
            D_sap_array[:, nap1:nap2]).ravel()
        nap1 = nap2
        naii1 = naii2
    return D_saii_array


if __name__ == '__main__':
    import sys
    from gpaw.mpi import world
    read_gpw(sys.argv[1], print, {'world': world})
