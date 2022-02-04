import ase.io.ulm as ulm
import gpaw
import numpy as np
from ase.io.trajectory import read_atoms, write_atoms
from ase.units import Bohr, Ha
from gpaw.new.builder import builder as create_builder
from gpaw.new.calculation import DFTCalculation, DFTState
from gpaw.new.density import Density
from gpaw.new.input_parameters import InputParameters
from gpaw.new.potential import Potential
# from gpaw.new.wave_functions import IBZWaveFunctions
from gpaw.core.atom_arrays import AtomArraysLayout
from gpaw.typing import Array1D


class OldStuff:
    def get_pseudo_wave_function(self, n):
        return self.calculation.ibzwfs[0].wave_functions.data[n]

    def get_fermi_level(self) -> float:
        fl = self.calculation.state.ibzwfs.fermi_levels * Ha
        assert len(fl) == 1
        return fl[0]

    def get_homo_lumo(self, spin: int = None) -> Array1D:
        return self.calculation.state.ibzwfs.get_homo_lumo(spin) * Ha

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
        writer.child('parameters').write(**dict(params.items()))
        calculation.state.density.write(writer.child('density'))
        calculation.state.potential.write(writer.child('hamiltonian'))
        calculation.state.ibzwfs.write(writer.child('wave_functions'),
                                       skip_wfs)

    world.barrier()


def read_gpw(filename, log, parallel):
    log(f'Reading from {filename}')

    world = parallel['world']

    reader = ulm.Reader(filename)
    bohr = reader.bohr
    ha = reader.ha

    atoms = read_atoms(reader.atoms)

    kwargs = reader.parameters.asdict()
    kwargs['parallel'] = parallel
    params = InputParameters(kwargs)

    builder = create_builder(atoms, params)
    kpt_band_comm = builder.communicators['D']

    assert reader.version >= 4

    if world.rank == 0:
        nt_sR_array = reader.density.density * bohr**3
        vt_sR_array = reader.hamiltonian.potential / ha
        D_sap_array = reader.density.atomic_density_matrices
        dH_sap_array = reader.hamiltonian.atomic_hamiltonian_matrices / ha
    else:
        nt_sR_array = None
        vt_sR_array = None
        D_sap_array = None
        dH_sap_array = None

    nt_sR = builder.grid.empty(builder.ncomponents)
    vt_sR = builder.grid.empty(builder.ncomponents)

    atom_array_layout = AtomArraysLayout([(setup.ni * (setup.ni + 1) // 2)
                                          for setup in builder.setups],
                                         atomdist=builder.atomdist)
    D_asp = atom_array_layout.empty(builder.ncomponents)
    dH_asp = atom_array_layout.empty(builder.ncomponents)

    if kpt_band_comm.rank == 0:
        nt_sR.scatter_from(nt_sR_array)
        vt_sR.scatter_from(vt_sR_array)
        D_asp.scatter_from(D_sap_array)
        dH_asp.scatter_from(dH_sap_array)

    kpt_band_comm.broadcast(nt_sR.data, 0)
    kpt_band_comm.broadcast(vt_sR.data, 0)
    kpt_band_comm.broadcast(D_asp.data, 0)
    kpt_band_comm.broadcast(dH_asp.data, 0)

    density = Density.from_data_and_setups(nt_sR, D_asp.to_full(),
                                           builder.params.charge,
                                           builder.setups)
    potential = Potential(vt_sR, dH_asp.to_full(), {})
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


if __name__ == '__main__':
    import sys
    from gpaw.mpi import world
    read_gpw(sys.argv[1], print, {'world': world})
