from __future__ import annotations

import warnings
from pathlib import Path
from typing import IO, Any, Union

from ase import Atoms
from ase.units import Bohr, Ha
from gpaw import __version__
from gpaw.new import Timer
from gpaw.new.calculation import DFTCalculation, DFTState, units
from gpaw.new.gpw import read_gpw, write_gpw
from gpaw.new.input_parameters import InputParameters
from gpaw.new.logger import Logger
from gpaw.new.pw.fulldiag import diagonalize
from gpaw.new.xc import XC
from gpaw.typing import Array1D, Array2D, Array3D
from gpaw.utilities import pack
from gpaw.utilities.memory import maxrss


def GPAW(filename: Union[str, Path, IO[str]] = None,
         **kwargs) -> ASECalculator:
    """Create ASE-compatible GPAW calculator."""
    params = InputParameters(kwargs)
    txt = params.txt
    if txt == '?':
        txt = '-' if filename is None else None
    world = params.parallel['world']
    log = Logger(txt, world)

    if filename is not None:
        kwargs.pop('txt', None)
        kwargs.pop('parallel', None)
        kwargs.pop('communicator', None)
        assert len(kwargs) == 0
        atoms, calculation, params, _ = read_gpw(filename, log,
                                                 params.parallel)
        return ASECalculator(params, log, calculation, atoms)

    write_header(log, world, params)
    return ASECalculator(params, log)


class ASECalculator:
    """This is the ASE-calculator frontend for doing a GPAW calculation."""
    def __init__(self,
                 params: InputParameters,
                 log: Logger,
                 calculation=None,
                 atoms=None):
        self.params = params
        self.log = log
        self.calculation = calculation

        self.atoms = atoms
        self.timer = Timer()

    def __repr__(self):
        params = []
        for key, value in self.params.items():
            val = repr(value)
            if len(val) > 40:
                val = '...'
            params.append((key, val))
        p = ', '.join(f'{key}: {val}' for key, val in params)
        return f'ASECalculator({p})'

    def calculate_property(self, atoms: Atoms, prop: str) -> Any:
        """Calculate (if not already calculated) a property.

        Must be one of

        * energy
        * forces
        * stress
        * magmom
        * magmoms
        * dipole
        """
        if self.calculation is not None:
            changes = compare_atoms(self.atoms, atoms)
            if changes & {'numbers', 'pbc', 'cell'}:
                # Start from scratch:
                if 'numbers' not in changes:
                    # Remember magmoms if there are any:
                    magmom_a = self.calculation.results.get('magmoms')
                    if magmom_a is not None:
                        atoms = atoms.copy()
                        atoms.set_initial_magnetic_moments(magmom_a)
                self.calculation = None

        if self.calculation is None:
            self.calculation = self.create_new_calculation(atoms)
            self.converge()
        elif changes:
            self.move_atoms(atoms)
            self.converge()

        if prop not in self.calculation.results:
            if prop == 'forces':
                with self.timer('Forces'):
                    self.calculation.forces()
            elif prop == 'stress':
                with self.timer('Stress'):
                    self.calculation.stress()
            elif prop == 'dipole':
                self.calculation.dipole()
            else:
                raise ValueError('Unknown property:', prop)

        return self.calculation.results[prop] * units[prop]

    def create_new_calculation(self, atoms: Atoms) -> DFTCalculation:
        with self.timer('Init'):
            calculation = DFTCalculation.from_parameters(atoms, self.params,
                                                         self.log)
        self.atoms = atoms.copy()
        return calculation

    def move_atoms(self, atoms):
        with self.timer('Move'):
            self.calculation = self.calculation.move_atoms(atoms)
        self.atoms = atoms.copy()

    def converge(self):
        """Iterate to self-consistent solution.

        Will also calculate "cheap" properties: energy, magnetic moments
        and dipole moment.
        """
        with self.timer('SCF'):
            self.calculation.converge()

        # Calculate all the cheap things:
        self.calculation.energies()
        self.calculation.dipole()
        self.calculation.magmoms()

        self.calculation.write_converged()

    def __del__(self):
        try:
            self.log('---')
            self.timer.write(self.log)
            mib = maxrss() / 1024**2
            self.log(f'\nMax RSS: {mib:.3f}  # MiB')
        except NameError:
            pass

    def get_potential_energy(self,
                             atoms: Atoms,
                             force_consistent: bool = False) -> float:
        return self.calculate_property(atoms,
                                       'free_energy' if force_consistent else
                                       'energy')

    def get_forces(self, atoms: Atoms) -> Array2D:
        return self.calculate_property(atoms, 'forces')

    def get_stress(self, atoms: Atoms) -> Array1D:
        return self.calculate_property(atoms, 'stress')

    def get_dipole_moment(self, atoms: Atoms) -> Array1D:
        return self.calculate_property(atoms, 'dipole')

    def get_magnetic_moment(self, atoms: Atoms) -> float:
        return self.calculate_property(atoms, 'magmom')

    def get_magnetic_moments(self, atoms: Atoms) -> Array1D:
        return self.calculate_property(atoms, 'magmoms')

    def write(self, filename, mode=''):
        """Write calculator object to a file.

        Parameters
        ----------
        filename:
            File to be written
        mode:
            Write mode. Use ``mode='all'``
            to include wave functions in the file.
        """
        self.log(f'# Writing to {filename} (mode={mode!r})\n')

        write_gpw(filename, self.atoms, self.params,
                  self.calculation, skip_wfs=mode != 'all')

    # Old API:

    def get_pseudo_wave_function(self, band, kpt=0, spin=0) -> Array3D:
        state = self.calculation.state
        wfs = state.ibzwfs.get_wfs(spin=spin, kpt=kpt, n1=band, n2=band + 1)
        basis = getattr(self.calculation.scf_loop.hamiltonian, 'basis', None)
        grid = state.density.nt_sR.desc
        wfs = wfs.to_uniform_grid_wave_functions(grid, basis)
        psit_R = wfs.psit_nX[0]
        if not psit_R.desc.pbc.all():
            psit_R = psit_R.to_pbc_grid()
        return psit_R.data * Bohr**-1.5

    def get_atoms(self):
        atoms = self.atoms.copy()
        atoms.calc = self
        return atoms

    def get_fermi_level(self) -> float:
        state = self.calculation.state
        fl = state.ibzwfs.fermi_levels * Ha
        assert len(fl) == 1
        return fl[0]

    def get_homo_lumo(self, spin: int = None) -> Array1D:
        state = self.calculation.state
        return state.ibzwfs.get_homo_lumo(spin) * Ha

    def get_number_of_electrons(self):
        state = self.calculation.state
        return state.ibzwfs.nelectrons

    def get_number_of_bands(self):
        state = self.calculation.state
        return state.ibzwfs.nbands

    def get_atomic_electrostatic_potentials(self):
        return self.calculation.electrostatic_potential().atomic_potentials()

    def get_pseudo_density(self, spin=None):
        return self.calculation.densities.pseudo_densities().data

    def get_eigenvalues(self, kpt=0, spin=0):
        state = self.calculation.state
        return state.ibzwfs.get_eigs_and_occs(k=kpt, s=spin)[0] * Ha

    def get_reference_energy(self):
        return self.calculation.setups.Eref * Ha

    def get_number_of_iterations(self):
        return self.calculation.scf_loop.niter

    def get_bz_k_points(self):
        state = self.calculation.state
        return state.ibzwfs.ibz.bz.kpt_Kc.copy()

    def get_ibz_k_points(self):
        state = self.calculation.state
        return state.ibzwfs.ibz.kpt_kc.copy()

    def calculate(self, atoms):
        self.get_potential_energy(atoms)

    @property
    def wfs(self):
        from gpaw.new.backwards_compatibility import FakeWFS
        return FakeWFS(self.calculation, self.atoms)

    @property
    def density(self):
        from gpaw.new.backwards_compatibility import FakeDensity
        return FakeDensity(self.calculation)

    @property
    def hamiltonian(self):
        from gpaw.new.backwards_compatibility import FakeHamiltonian
        return FakeHamiltonian(self.calculation)

    @property
    def spos_ac(self):
        return self.atoms.get_scaled_positions()

    def get_xc_difference(self, xcparams):
        """Calculate non-selfconsistent XC-energy difference."""
        state = self.calculation.state
        xc = XC(xcparams, state.density.ncomponents)
        exct = self.calculation.pot_calc.calculate_non_selfconsistent_exc(
            state.density.nt_sR, xc)
        dexc = 0.0
        for a, D_sii in state.density.D_asii.items():
            setup = self.setups[a]
            dexc += xc.calculate_paw_correction(setup, pack(D_sii))
        return (exct + dexc - state.potential.energies['xc']) * Ha

    def diagonalize_full_hamiltonian(self,
                                     nbands=None,
                                     scalapack=None,
                                     expert=None):
        if expert is not None:
            warnings.warn('Ignoring deprecated "expert" argument')
        state = self.calculation.state
        ibzwfs = diagonalize(state.potential,
                             state.ibzwfs,
                             self.calculation.scf_loop.occ_calc,
                             nbands)
        self.calculation.state = DFTState(ibzwfs,
                                          state.density,
                                          state.potential)
        nbands = ibzwfs.nbands
        self.params.nbands = nbands


def write_header(log, world, params):
    from gpaw.io.logger import write_header as header
    log(f'#  __  _  _\n# | _ |_)|_||  |\n# |__||  | ||/\\| - {__version__}\n')
    header(log, world)
    log('---')
    with log.indent('input parameters:'):
        log(**{k: v for k, v in params.items()})


def compare_atoms(a1: Atoms, a2: Atoms) -> set[str]:
    if len(a1.numbers) != len(a2.numbers) or (a1.numbers != a2.numbers).any():
        return {'numbers'}
    if (a1.pbc != a2.pbc).any():
        return {'pbc'}
    if abs(a1.cell - a2.cell).max() > 0.0:
        return {'cell'}
    if abs(a1.positions - a2.positions).max() > 0.0:
        return {'positions'}
    return set()
