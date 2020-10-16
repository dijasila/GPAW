from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.units import Bohr

from .wavefunctions.pw import PWWaveFunctions, PWMapping, PWDescriptor
from .wavefunctions.arrays import PlaneWaveExpansionWaveFunctions
from gpaw.hints import Array2D

if TYPE_CHECKING:
    from . import GPAW


def interpolate_wave_functions(calc: 'GPAW',
                               atoms: Atoms):
    """Raises NotImplementedError if ..."""
    if calc.mode.name != 'pw':
        raise NotImplementedError

    wfs1 = calc.wfs
    dens1 = calc.density
    ham1 = calc.hamiltonian
    symm1 = calc.symmetry

    calc.density = None

    cell_cv = atoms.cell / Bohr

    magmom_av = np.zeros((len(atoms), 3))
    magmom_av[:, 2] = atoms.get_initial_magnetic_moments()
    calc.create_symmetry(magmom_av, cell_cv)
    if not equal_symms(symm1, calc.symmetry):
        calc.symmtries = symm1
        raise NotImplementedError

    kd2 = calc.create_kpoint_descriptor(wfs1.nspins)
    if (kd2.N_c != wfs1.kd.N_c).any():
        raise NotImplementedError

    N_c, h = calc.choose_number_of_grid_points(cell_cv, atoms.pbc,
                                               mode=calc.mode)
    gd = wfs1.gd.new_descriptor(N_c=N_c, cell_cv=cell_cv)

    wfs2 = PWWaveFunctions(
        wfs1.ecut,
        wfs1.gammacentered,
        wfs1.fftwflags,
        wfs1.dedepsilon,
        (calc.comms['K'],) + wfs1.scalapack_parameters[1:],
        wfs1.initksl,
        wfs1.wfs_mover,
        wfs1.collinear,
        gd,
        wfs1.nvalence,
        wfs1.setups,
        wfs1.bd,
        wfs1.dtype,
        wfs1.world,
        wfs1.kd,
        wfs1.kptband_comm,
        wfs1.timer)

    for kpt1, kpt2 in zip(wfs1.kpt_u, wfs2.kpt_u):
        psit2 = PlaneWaveExpansionWaveFunctions(
            wfs2.bd.nbands, wfs2.pd, wfs2.dtype,
            kpt=kpt2.q, dist=(wfs2.bd.comm, wfs2.bd.comm.size),
            spin=kpt2.s, collinear=wfs2.collinear)

        interpolate(wfs1.pd, wfs2.pd, kpt2.q,
                    kpt1.psit.array, psit2.array)

        kpt2.f_n = kpt1.f_n

        kpt1.psit = None
        kpt2.psit = psit2

    wfs2.occupations = wfs1.occupations

    calc.wfs = wfs2
    xc = ham1.xc
    calc.create_eigensolver(xc, wfs1.bd.nbands, calc.mode)

    totmom_v, magmom_av = dens1.estimate_magnetic_moments()
    calc.create_density(False, 'pw', dens1.background_charge, h)
    assert calc.density is not None
    calc.density.initialize(calc.setups, calc.timer,
                            magmom_av=magmom_av, hund=calc.parameters.hund)
    calc.density.set_mixer(calc.parameters.mixer)
    calc.density.log = calc.log

    calc.create_hamiltonian(realspace=False, mode=calc.mode, xc=xc)

    xc.initialize(calc.density, calc.hamiltonian, wfs2)


def equal_symms(symm1, symm2):
    """Compare two symmetry objects."""
    return (len(symm1.op_scc) == len(symm2.op_scc) and
            (symm1.op_scc == symm2.op_scc).all() and
            (symm1.ft_sc == symm2.ft_sc).all() and
            (symm1.a_sa == symm2.a_sa).all())


def interpolate(pd1: PWDescriptor,
                pd2: PWDescriptor,
                q: int,
                a1_nG: Array2D,
                a2_nG: Array2D) -> None:
    """Interpolate wave functions from one cell to another."""
    map12 = PWMapping(pd1, pd2, q)
    a2_nG[:] = 0.0
    for a1_G, a2_G in zip(a1_nG, a2_nG):
        map12.add_to2(a2_G, a1_G)
