from __future__ import annotations

import warnings

import numpy as np
from ase.units import Bohr
from gpaw.hybrids import HybridXC
from gpaw.mpi import MPIComm, Parallelization, world
from gpaw.new.brillouin import BZ, MonkhorstPackKPoints
from gpaw.new.modes import FDMode
from gpaw.new.symmetry import Symmetry
from gpaw.new.xc import XCFunctional
from gpaw.scf import dict2criterion
from gpaw.setup import Setups
from gpaw.symmetry import Symmetry as OldSymmetry
from gpaw.xc import XC
from gpaw.new.potential import PotentialCalculator
from gpaw.new.wave_functions import IBZWaveFunctions
from gpaw.new.davidson import Davidson
from gpaw.new.scf import SCFLoop
from gpaw.new.density import Density
from gpaw.new.smearing import OccupationNumberCalculator


class DFTConfiguration:
    def __init__(self,
                 atoms,
                 params):
        self.params = params
        parallel = params.parallel
        world = parallel['world']

        self.mode = create_mode(params.mode)
        self.xc = XCFunctional(XC(params.xc))
        self.setups = Setups(atoms.numbers,
                             params.setups,
                             params.basis,
                             self.xc.setup_name,
                             world)
        self.initial_magmoms = normalize_initial_magnetic_moments(
            params.magmoms, atoms)

        symmetry = create_symmetry_object(atoms,
                                          self.setups.id_a,
                                          self.initial_magmoms,
                                          params.symmetry)
        bz = create_kpts(params.kpts, atoms)
        self.ibz = symmetry.reduce(bz)

        d = parallel.get('domain', None)
        k = parallel.get('kpt', None)
        b = parallel.get('band', None)
        if isinstance(self.xc, HybridXC):
            d = world.size
        self.communicators = create_communicators(world, len(self.ibz),
                                                  d, k, b)
        self.communicators['w'] = world

        self.grid = self.mode.create_uniform_grid(params.h,
                                                  params.gpts,
                                                  atoms.cell / Bohr,
                                                  atoms.pbc,
                                                  symmetry,
                                                  comm=self.communicators['d'])

        if self.mode.name == 'fd':
            pass  # filter = create_fourier_filter(grid)
            # setups = stups.filter(filter)

        self.fracpos = atoms.get_scaled_positions()

        self.grid2 = self.grid.new(size=self.grid.size * 2)
        # decomposition=[2 * d for d in grid.decomposition]

        self.nelectrons = self.setups.nvalence - params.charge

        if self.mode.name == 'pw':
            self.mode.create_plane_waves(self.grid)

        self.nbands = calculate_number_of_bands(params.nbands,
                                                self.setups,
                                                params.charge,
                                                self.initial_magmoms,
                                                self.mode.name == 'lcao')

        self._pot_calc = None

    def random_ibz_wave_functions(self):
        return IBZWaveFunctions.from_random_numbers(
            self.ibz,
            self.communicators['b'],
            self.communicators['k'],
            self.grid,
            self.setups,
            self.fracpos,
            self.nbands,
            self.nelectrons)

    def density_from_superposition(self):
        return Density.from_superposition(
            self.grid, self.setups, self.initial_magmoms,
            self.fracpos, self.params.charge, self.params.hund)

    @property
    def potential_calculator(self):
        if self._pot_calc is None:
            poisson_solver = self.mode.create_poisson_solver(
                self.grid2,
                self.params.poissonsolver)
            self._pot_calc = PotentialCalculator(self.grid, self.grid2,
                                                 self.setups, self.fracpos,
                                                 self.xc, poisson_solver)
        return self._pot_calc

    def scf_loop(self):
        hamiltonian = self.mode.create_hamiltonian_operator(self.grid)
        eigensolver = Davidson(self.nbands, self.grid, self.communicators['b'],
                               hamiltonian.create_preconditioner,
                               **self.params.eigensolver)

        mixer = ...
        cc = create_convergence_criteria(self.params.convergence)
        occ_calc = OccupationNumberCalculator(
            self.params.occupations,
            self.grid.pbc,
            self.ibz,
            self.nbands,
            self.communicators,
            self.initial_magmoms,
            self.grid.icell)

        return SCFLoop(hamiltonian, self.potential_calculator, occ_calc,
                       eigensolver, mixer, self.communicators['w'], cc)


def create_communicators(comm: MPIComm = None,
                         nibzkpts: int = 1,
                         domain: int | tuple[int, int, int] = None,
                         kpt: int = None,
                         band: int = None) -> dict[str, MPIComm]:
    parallelization = Parallelization(comm or world, nibzkpts)
    if domain is not None:
        domain = np.prod(domain)
    parallelization.set(kpt=kpt,
                        domain=domain,
                        band=band)
    comms = parallelization.build_communicators()
    return comms


def create_fourier_filter(grid):
    gamma = 1.6

    h = ((grid.icell**2).sum(1)**-0.5 / grid.size).max()

    def filter(rgd, rcut, f_r, l=0):
        gcut = np.pi / h - 2 / rcut / gamma
        ftmp = rgd.filter(f_r, rcut * gamma, gcut, l)
        f_r[:] = ftmp[:len(f_r)]

    return filter


def normalize_initial_magnetic_moments(magmoms, atoms):
    if magmoms is None:
        magmoms = atoms.get_initial_magnetic_moments()
    elif isinstance(magmoms, float):
        magmoms = np.zeros(len(atoms)) + magmoms
    else:
        magmoms = np.array(magmoms)

    collinear = magmoms.ndim == 1
    if collinear and not magmoms.any():
        magmoms = None

    return magmoms


def create_symmetry_object(atoms, ids=None, magmoms=None, parameters=None):
    ids = ids or [()] * len(atoms)
    if magmoms is None:
        pass
    elif magmoms.ndim == 1:
        ids = [id + (m,) for id, m in zip(ids, magmoms)]
    else:
        ids = [id + tuple(m) for id, m in zip(ids, magmoms)]
    print(ids, parameters)
    symmetry = OldSymmetry(ids, atoms.cell, atoms.pbc, **(parameters or {}))
    symmetry.analyze(atoms.get_scaled_positions())
    return Symmetry(symmetry)


def create_convergence_criteria(criteria):
    # Gather convergence criteria for SCF loop.
    custom = criteria.pop('custom', [])
    for name, criterion in criteria.items():
        if hasattr(criterion, 'todict'):
            # 'Copy' so no two calculators share an instance.
            criteria[name] = dict2criterion(criterion.todict())
        else:
            criteria[name] = dict2criterion({name: criterion})

    if not isinstance(custom, (list, tuple)):
        custom = [custom]
    for criterion in custom:
        if isinstance(criterion, dict):  # from .gpw file
            msg = ('Custom convergence criterion "{:s}" encountered, '
                   'which GPAW does not know how to load. This '
                   'criterion is NOT enabled; you may want to manually'
                   ' set it.'.format(criterion['name']))
            warnings.warn(msg)
            continue

        criteria[criterion.name] = criterion
        msg = ('Custom convergence criterion {:s} encountered. '
               'Please be sure that each calculator is fed a '
               'unique instance of this criterion. '
               'Note that if you save the calculator instance to '
               'a .gpw file you may not be able to re-open it. '
               .format(criterion.name))
        warnings.warn(msg)

    for criterion in criteria.values():
        criterion.reset()

    return criteria


def create_mode(mode):
    return FDMode()


def create_kpts(kpts, atoms):
    if 'points' in kpts:
        return BZ(kpts['points'])
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
