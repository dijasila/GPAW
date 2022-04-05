from __future__ import annotations

import ase.io.ulm as ulm
import gpaw
import numpy as np
from ase.io.trajectory import read_atoms, write_atoms
from ase.units import Bohr, Ha
from gpaw.core.atom_arrays import AtomArraysLayout
from gpaw.new.builder import builder as create_builder
from gpaw.new.calculation import DFTCalculation, DFTState, units
from gpaw.new.density import Density
from gpaw.new.input_parameters import InputParameters
from gpaw.new.potential import Potential


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
        results = {key: value * units[key]
                   for key, value in calculation.results.items()}
        writer.child('results').write(**results)
        writer.child('parameters').write(
            **{k: v for k, v in params.items() if k != 'txt'})

        state = calculation.state
        state.density.write(writer.child('density'))
        state.potential.write(writer.child('hamiltonian'))
        wf_writer = writer.child('wave_functions')
        state.ibzwfs.write(wf_writer, skip_wfs)

        if not skip_wfs and params.mode['name'] == 'pw':
            write_wave_function_indices(wf_writer,
                                        state.ibzwfs,
                                        state.density.nt_sR.desc)

    world.barrier()


def write_wave_function_indices(writer, ibzwfs, grid):
    if ibzwfs.band_comm.rank != 0:
        return
    if ibzwfs.domain_comm.rank != 0:
        return

    kpt_comm = ibzwfs.kpt_comm
    ibz = ibzwfs.ibz
    (nG,) = ibzwfs.get_max_shape(global_shape=True)

    writer.add_array('indices', (len(ibz), nG), np.int32)

    index_G = np.zeros(nG, np.int32)
    size = tuple(grid.size)
    if ibzwfs.dtype == float:
        size = (size[0], size[1], size[2] // 2 + 1)

    for k, rank in enumerate(ibzwfs.rank_k):
        if rank == kpt_comm.rank:
            wfs = ibzwfs.wfs_qs[ibzwfs.q_k[k]][0]
            i_G = wfs.psit_nX.desc.indices(size)
            index_G[:len(i_G)] = i_G
            index_G[len(i_G):] = -1
            if rank == 0:
                writer.fill(index_G)
            else:
                kpt_comm.send(index_G, 0)
        elif kpt_comm.rank == 0:
            kpt_comm.receive(index_G, rank)
            writer.fill(index_G)


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

    (kpt_comm, band_comm, domain_comm, kpt_band_comm) = (
        builder.communicators[x] for x in 'kbdD')

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

    ibzwfs = builder.read_ibz_wave_functions(reader)

    calculation = DFTCalculation(
        DFTState(ibzwfs, density, potential),
        builder.setups,
        builder.create_scf_loop(),
        pot_calc=builder.create_potential_calculator())

    results = {key: value / units[key]
               for key, value in reader.results.asdict().items()}
    if results:
        log(f'Read {", ".join(sorted(results))}')

    calculation.results = results
    return atoms, calculation, params


if __name__ == '__main__':
    import sys

    from gpaw.mpi import world
    read_gpw(sys.argv[1], print, {'world': world})
