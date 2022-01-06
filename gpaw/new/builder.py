from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Union

import numpy as np
from ase import Atoms
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.lfc import BasisFunctions
from gpaw.mixer import MixerWrapper, get_mixer_from_keywords
from gpaw.mpi import MPIComm, Parallelization, world
from gpaw.new import cached_property
from gpaw.new.brillouin import BZPoints, MonkhorstPackKPoints
from gpaw.new.davidson import Davidson
from gpaw.new.density import Density
from gpaw.new.input_parameters import InputParameters
from gpaw.new.modes import FDMode, PWMode
from gpaw.new.scf import SCFLoop
from gpaw.new.smearing import OccupationNumberCalculator
from gpaw.new.symmetry import Symmetries
from gpaw.new.wave_functions import IBZWaveFunctions
from gpaw.new.xc import XCFunctional
from gpaw.setup import Setups
from gpaw.symmetry import Symmetry as OldSymmetry
from gpaw.xc import XC


class DFTComponentsBuilder:
    def __init__(self,
                 atoms: Atoms,
                 params: dict[str, Any] | InputParameters):

        self.atoms = atoms.copy()

        number_of_lattice_vectors = atoms.cell.rank
        if number_of_lattice_vectors < 3:
            raise ValueError(
                'GPAW requires 3 lattice vectors.  '
                f'Your system has {number_of_lattice_vectors}.')

        if isinstance(params, dict):
            params = InputParameters(params)
        self.params = params

        parallel = params.parallel
        world = parallel['world']

        self.mode = create_mode(**params.mode)
        self.xc = XCFunctional(XC(params.xc))  # mode?
        self.setups = Setups(atoms.numbers,
                             params.setups,
                             params.basis,
                             self.xc.setup_name,
                             world)
        self.initial_magmoms = normalize_initial_magnetic_moments(
            params.magmoms, atoms, params.spinpol)

        symmetry = create_symmetry_object(atoms,
                                          self.setups.id_a,
                                          self.initial_magmoms,
                                          params.symmetry)
        bz = create_kpts(params.kpts, atoms)
        self.ibz = symmetry.reduce(bz)

        d = parallel.get('domain', None)
        k = parallel.get('kpt', None)
        b = parallel.get('band', None)
        self.communicators = create_communicators(world, len(self.ibz),
                                                  d, k, b)
        self.grid = self.mode.create_uniform_grid(
            params.h,
            params.gpts,
            atoms.cell,
            atoms.pbc,
            symmetry,
            comm=self.communicators['d'])

        self.wf_desc = self.mode.create_wf_description(self.grid, self.dtype)
        self.nct_aX = self.mode.create_pseudo_core_densities(
            self.setups, self.grid, self.fracpos_ac)

        if self.mode.name == 'fd':
            pass  # filter = create_fourier_filter(grid)
            # setups = stups.filter(filter)

        self.nelectrons = self.setups.nvalence - params.charge

        self.nbands = calculate_number_of_bands(params.nbands,
                                                self.setups,
                                                params.charge,
                                                self.initial_magmoms,
                                                self.mode.name == 'lcao')

        self.fine_grid = self.grid.new(size=self.grid.size_c * 2)
        # decomposition=[2 * d for d in grid.decomposition]

    @cached_property
    def ncomponents(self):
        if self.initial_magmoms is None:
            return 1
        if self.initial_magmoms.ndim == 1:
            return 2
        return 4

    @cached_property
    def dtype(self):
        if self.mode.force_complex_dtype:
            return complex
        if self.ibz.bz.gamma_only:
            return float
        return complex

    @cached_property
    def fracpos_ac(self):
        return self.atoms.get_scaled_positions()

    @cached_property
    def atomdist(self):
        return self.nct_aX.layout.atomdist

    @cached_property
    def nct_R(self):
        out = self.grid.empty()
        self.nct_aX.to_uniform_grid(out=out,
                                    scale=1.0 / (self.ncomponents % 3))
        return out

    def create_potential_calculator(self):
        return self.mode.create_potential_calculator(
            self.grid,
            self.fine_grid,
            self.setups,
            self.xc,
            self.params.poissonsolver,
            self.nct_aX, self.nct_R)

    def __repr__(self):
        return f'DFTComponentsBuilder({self.atoms}, {self.params})'

    def lcao_ibz_wave_functions(self, basis_set, potential):
        from gpaw.new.lcao import create_lcao_ibz_wave_functions
        sl_default = self.params.parallel['sl_default']
        sl_lcao = self.params.parallel['sl_lcao'] or sl_default
        return create_lcao_ibz_wave_functions(
            self.setups,
            self.communicators,
            self.nbands,
            self.ncomponents,
            self.nelectrons,
            self.fracpos_ac,
            self.dtype,
            self.grid,
            self.wf_desc,
            self.ibz,
            sl_lcao,
            basis_set,
            potential)

    def random_ibz_wave_functions(self):
        return IBZWaveFunctions.from_random_numbers(
            self.ibz,
            self.communicators['b'],
            self.communicators['k'],
            self.wf_desc,
            self.setups,
            self.fracpos_ac,
            self.nbands,
            self.nelectrons,
            self.dtype)

    def create_basis_set(self):
        kd = KPointDescriptor(self.ibz.bz.kpt_Kc, self.ncomponents % 3)
        kd.set_symmetry(SimpleNamespace(pbc=self.grid.pbc),
                        self.ibz.symmetries.symmetry,
                        comm=self.communicators['w'])
        kd.set_communicator(self.communicators['k'])

        basis_set = BasisFunctions(self.grid._gd,
                                   [setup.phit_j for setup in self.setups],
                                   kd,
                                   dtype=self.dtype,
                                   cut=True)
        basis_set.set_positions(self.fracpos_ac)
        return basis_set

    def density_from_superposition(self, basis_set):
        return Density.from_superposition(self.grid,
                                          self.nct_R,
                                          self.atomdist,
                                          self.setups,
                                          basis_set,
                                          self.initial_magmoms,
                                          self.params.charge,
                                          self.params.hund)

    def create_scf_loop(self, pot_calc):
        hamiltonian = self.mode.create_hamiltonian_operator(self.wf_desc)
        eigensolver = Davidson(self.nbands,
                               self.wf_desc,
                               self.communicators['b'],
                               hamiltonian.create_preconditioner,
                               **self.params.eigensolver)

        mixer = MixerWrapper(
            get_mixer_from_keywords(self.atoms.pbc.any(),
                                    self.ncomponents, **self.params.mixer),
            self.ncomponents, self.grid._gd)

        occ_calc = OccupationNumberCalculator(
            self.params.occupations,
            self.atoms.pbc,
            self.ibz,
            self.nbands,
            self.communicators,
            self.initial_magmoms,
            self.grid.icell)

        return SCFLoop(hamiltonian, pot_calc, occ_calc,
                       eigensolver, mixer, self.communicators['w'],
                       self.params.convergence,
                       self.params.maxiter)


def create_communicators(comm: MPIComm = None,
                         nibzkpts: int = 1,
                         domain: Union[int, tuple[int, int, int]] = None,
                         kpt: int = None,
                         band: int = None) -> dict[str, MPIComm]:
    parallelization = Parallelization(comm or world, nibzkpts)
    if domain is not None:
        domain = np.prod(domain)
    parallelization.set(kpt=kpt,
                        domain=domain,
                        band=band)
    comms = parallelization.build_communicators()
    comms['w'] = comm
    return comms


def create_fourier_filter(grid):
    gamma = 1.6

    h = ((grid.icell**2).sum(1)**-0.5 / grid.size).max()

    def filter(rgd, rcut, f_r, l=0):
        gcut = np.pi / h - 2 / rcut / gamma
        ftmp = rgd.filter(f_r, rcut * gamma, gcut, l)
        f_r[:] = ftmp[:len(f_r)]

    return filter


def normalize_initial_magnetic_moments(magmoms,
                                       atoms,
                                       force_spinpol_calculation=False):
    if magmoms is None:
        magmoms = atoms.get_initial_magnetic_moments()
    elif isinstance(magmoms, float):
        magmoms = np.zeros(len(atoms)) + magmoms
    else:
        magmoms = np.array(magmoms)

    collinear = magmoms.ndim == 1
    if collinear and not magmoms.any():
        magmoms = None

    if force_spinpol_calculation and magmoms is None:
        magmoms = np.zeros(len(atoms))

    return magmoms


def create_symmetry_object(atoms, ids=None, magmoms=None, parameters=None):
    ids = ids or [()] * len(atoms)
    if magmoms is None:
        pass
    elif magmoms.ndim == 1:
        ids = [id + (m,) for id, m in zip(ids, magmoms)]
    else:
        ids = [id + tuple(m) for id, m in zip(ids, magmoms)]
    symmetry = OldSymmetry(ids, atoms.cell, atoms.pbc, **(parameters or {}))
    symmetry.analyze(atoms.get_scaled_positions())
    return Symmetries(symmetry)


def create_mode(name, **kwargs):
    if name == 'pw':
        return PWMode(**kwargs)
    if name == 'fd':
        return FDMode(**kwargs)
    1 / 0


def create_kpts(kpts: dict[str, Any], atoms: Atoms) -> BZPoints:
    if 'points' in kpts:
        assert len(kpts) == 1, kpts
        return BZPoints(kpts['points'])
    assert len(kpts) == 1
    return MonkhorstPackKPoints(kpts['size'])


def calculate_number_of_bands(nbands, setups, charge, magmoms, is_lcao):
    nao = setups.nao
    nvalence = setups.nvalence - charge
    M = 0 if magmoms is None else np.linalg.norm(magmoms.sum(0))

    orbital_free = any(setup.orbital_free for setup in setups)
    if orbital_free:
        nbands = 1

    if isinstance(nbands, str):
        if nbands == 'nao':
            nbands = nao
        elif nbands[-1] == '%':
            cfgbands = (nvalence + M) / 2
            nbands = int(np.ceil(float(nbands[:-1]) / 100 * cfgbands))
        else:
            raise ValueError('Integer expected: Only use a string '
                             'if giving a percentage of occupied bands')

    if nbands is None:
        # Number of bound partial waves:
        nbandsmax = sum(setup.get_default_nbands()
                        for setup in setups)
        nbands = int(np.ceil((1.2 * (nvalence + M) / 2))) + 4
        if nbands > nbandsmax:
            nbands = nbandsmax
        if is_lcao and nbands > nao:
            nbands = nao
    elif nbands <= 0:
        nbands = max(1, int(nvalence + M + 0.5) // 2 + (-nbands))

    if nbands > nao and is_lcao:
        raise ValueError('Too many bands for LCAO calculation: '
                         '%d bands and only %d atomic orbitals!' %
                         (nbands, nao))

    if nvalence < 0:
        raise ValueError(
            'Charge %f is not possible - not enough valence electrons' %
            charge)

    if nvalence > 2 * nbands and not orbital_free:
        raise ValueError('Too few bands!  Electrons: %f, bands: %d'
                         % (nvalence, nbands))

    return nbands
