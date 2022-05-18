from __future__ import annotations

from typing import Any, Union

import numpy as np
from ase import Atoms
from ase.geometry import cell_to_cellpar
from ase.units import Bohr, Ha

from gpaw.core.arrays import DistributedArrays
from gpaw.core.uniform_grid import UniformGridFunctions
from gpaw.new.builder import builder as create_builder
from gpaw.new.density import Density
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.input_parameters import InputParameters
from gpaw.new.logger import Logger
from gpaw.new.potential import Potential
from gpaw.output import plot
from gpaw.typing import Array1D, Array2D
from gpaw.utilities import (check_atoms_too_close,
                            check_atoms_too_close_to_boundary)

units = {'energy': Ha,
         'free_energy': Ha,
         'forces': Ha / Bohr,
         'stress': Ha / Bohr**3,
         'dipole': Bohr,
         'magmom': 1.0,
         'magmoms': 1.0}


class DFTState:
    def __init__(self,
                 ibzwfs: IBZWaveFunctions,
                 density: Density,
                 potential: Potential,
                 vHt_x: DistributedArrays = None,
                 nct_R: UniformGridFunctions = None):
        """State of a Kohn-Sham calculation."""
        self.ibzwfs = ibzwfs
        self.density = density
        self.potential = potential
        self.vHt_x = vHt_x  # initial guess for Hartree potential

    def __repr__(self):
        return (f'DFTState({self.ibzwfs!r}, '
                f'{self.density!r}, {self.potential!r})')

    def __str__(self):
        return f'{self.ibzwfs}\n{self.density}\n{self.potential}'

    def move(self, fracpos_ac, atomdist, delta_nct_R):
        self.ibzwfs.move(fracpos_ac, atomdist)
        self.potential.energies.clear()
        self.density.move(delta_nct_R)  # , atomdist) XXX


class DFTCalculation:
    def __init__(self,
                 state: DFTState,
                 setups,
                 scf_loop,
                 pot_calc,
                 log: Logger):
        self.state = state
        self.setups = setups
        self.scf_loop = scf_loop
        self.pot_calc = pot_calc
        self.log = log

        self.results: dict[str, Any] = {}
        self.fracpos_ac = self.pot_calc.fracpos_ac

    @classmethod
    def from_parameters(cls,
                        atoms: Atoms,
                        params: Union[dict, InputParameters],
                        log=None,
                        builder=None) -> DFTCalculation:
        """Create DFTCalculation object from parameters and atoms."""

        check_atoms_too_close(atoms)
        check_atoms_too_close_to_boundary(atoms)

        if params is None:
            params = {}
        if isinstance(params, dict):
            params = InputParameters(params)

        builder = builder or create_builder(atoms, params)

        if not isinstance(log, Logger):
            log = Logger(log, builder.world)

        basis_set = builder.create_basis_set()

        density = builder.density_from_superposition(basis_set)
        density.normalize()

        pot_calc = builder.create_potential_calculator()
        potential, vHt_x, _ = pot_calc.calculate(density)
        ibzwfs = builder.create_ibz_wave_functions(basis_set, potential)
        state = DFTState(ibzwfs, density, potential, vHt_x)
        scf_loop = builder.create_scf_loop()

        write_atoms(atoms, builder.grid, builder.initial_magmoms, log)
        log(state)
        log(builder.setups)
        log(scf_loop)
        log(pot_calc)

        return cls(state,
                   builder.setups,
                   scf_loop,
                   pot_calc,
                   log)

    def move_atoms(self, atoms) -> DFTCalculation:
        check_atoms_too_close(atoms)

        self.fracpos_ac = atoms.get_scaled_positions()

        atomdist = ...

        delta_nct_R = self.pot_calc.move(self.fracpos_ac,
                                         atomdist,
                                         self.state.density.ndensities)
        self.state.move(self.fracpos_ac, atomdist, delta_nct_R)

        magmoms = self.results.get('magmoms')
        write_atoms(atoms,
                    self.state.density.nt_sR.desc,
                    magmoms,
                    self.log)

        self.results = {}

        return self

    def iconverge(self, convergence=None, maxiter=None):
        self.state.ibzwfs.make_sure_wfs_are_read_from_gpw_file()
        for ctx in self.scf_loop.iterate(self.state,
                                         self.pot_calc,
                                         convergence,
                                         maxiter,
                                         log=self.log):
            yield ctx

    def converge(self,
                 convergence=None,
                 maxiter=None,
                 steps=99999999999999999):
        """Converge to self-consistent solution of Kohn-Sham equation."""
        for step, _ in enumerate(self.iconverge(convergence, maxiter),
                                 start=1):
            if step == steps:
                break
        else:  # no break
            self.log(scf_steps=step)

    def energies(self):
        energies1 = self.state.potential.energies.copy()
        energies2 = self.state.ibzwfs.energies
        energies1['kinetic'] += energies2['band']
        energies1['entropy'] = energies2['entropy']
        free_energy = sum(energies1.values())
        extrapolated_energy = free_energy + energies2['extrapolation']

        self.log('energies:  # [eV]')
        for name, e in energies1.items():
            self.log(f'  {name + ":":10}   {e * Ha:14.6f}')
        self.log(f'  total:       {free_energy * Ha:14.6f}')
        self.log(f'  extrapolated:{extrapolated_energy * Ha:14.6f}')

        self.results['free_energy'] = free_energy
        self.results['energy'] = extrapolated_energy

    def dipole(self):
        dipole_v = self.state.density.calculate_dipole_moment(self.fracpos_ac)
        x, y, z = dipole_v * Bohr
        self.log(f'dipole moment: [{x:.6f}, {y:.6f}, {z:.6f}]  # |e|*Ang\n')
        self.results['dipole'] = dipole_v

    def magmoms(self) -> tuple[Array1D, Array2D]:
        mm_v, mm_av = self.state.density.calculate_magnetic_moments()
        self.results['magmom'] = mm_v[2]
        self.results['magmoms'] = mm_av[:, 2].copy()

        if self.state.density.ncomponents > 1:
            x, y, z = mm_v
            self.log(f'total magnetic moment: [{x:.6f}, {y:.6f}, {z:.6f}]')
            self.log('local magnetic moments: [')
            for a, (setup, m_v) in enumerate(zip(self.setups, mm_av)):
                x, y, z = m_v
                c = ',' if a < len(mm_av) - 1 else ']'
                self.log(f'  [{x:9.6f}, {y:9.6f}, {z:9.6f}]{c}'
                         '  # {setup.symbol:2} {a}')
            self.log()
        return mm_v, mm_av

    def forces(self):
        """Return atomic force contributions."""
        xc = self.pot_calc.xc
        assert not xc.no_forces
        assert not hasattr(xc.xc, 'setup_force_corrections')

        # Force from projector functions (and basis set):
        F_av = self.state.ibzwfs.forces(self.state.potential.dH_asii)

        pot_calc = self.pot_calc
        Fcc_avL, Fnct_av, Fvbar_av = pot_calc.force_contributions(
            self.state)

        # Force from compensation charges:
        ccc_aL = \
            self.state.density.calculate_compensation_charge_coefficients()
        for a, dF_vL in Fcc_avL.items():
            F_av[a] += dF_vL @ ccc_aL[a]

        # Force from smooth core charge:
        for a, dF_v in Fnct_av.items():
            F_av[a] += dF_v[:, 0]

        # Force from zero potential:
        for a, dF_v in Fvbar_av.items():
            F_av[a] += dF_v[:, 0]

        domain_comm = ccc_aL.layout.atomdist.comm
        domain_comm.sum(F_av)

        F_av = self.state.ibzwfs.ibz.symmetries.symmetrize_forces(F_av)

        self.log('\nforces: [  # [eV/Ang]')
        s = Ha / Bohr
        for a, setup in enumerate(self.setups):
            x, y, z = F_av[a] * s
            c = ',' if a < len(F_av) - 1 else ']'
            self.log(f'  [{x:9.3f}, {y:9.3f}, {z:9.3f}]{c}'
                     f'  # {setup.symbol:2} {a}')

        self.results['forces'] = F_av

    def stress(self):
        stress_vv = self.pot_calc.stress_contribution(self.state)
        self.log('\nstress tensor: [  # [eV/Ang^3]')
        for (x, y, z), c in zip(stress_vv * (Ha / Bohr**3), ',,]'):
            self.log(f'  [{x:13.6f}, {y:13.6f}, {z:13.6f}]{c}')
        self.results['stress'] = stress_vv.flat[[0, 4, 8, 5, 2, 1]]

    def write_converged(self):
        self.state.ibzwfs.write_summary(self.log)


def write_atoms(atoms, grid, magmoms, log):
    if magmoms is None:
        magmoms = np.zeros((len(atoms), 3))
    elif magmoms.ndim == 1:
        m1 = magmoms
        magmoms = np.zeros((len(atoms), 3))
        magmoms[:, 2] = m1

    log()
    with log.comment():
        log(plot(atoms))

    log('\natoms: [  # symbols, positions [Ang] and initial magnetic moments')
    symbols = atoms.get_chemical_symbols()
    for a, ((x, y, z), (mx, my, mz)) in enumerate(zip(atoms.positions,
                                                      magmoms)):
        symbol = symbols[a]
        c = ']' if a == len(atoms) - 1 else ','
        log(f'  [{symbol:>3}, [{x:11.6f}, {y:11.6f}, {z:11.6f}],'
            f' [{mx:6.3f}, {my:6.3f}, {mz:6.3f}]]{c} # {a}')

    log('\ncell: [  # Ang')
    log('#     x            y            z')
    for (x, y, z), c in zip(atoms.cell, ',,]'):
        log(f'  [{x:11.6f}, {y:11.6f}, {z:11.6f}]{c}')

    log()
    log(f'periodic: [{", ".join(f"{str(p):10}" for p in atoms.pbc)}]')
    a, b, c, A, B, C = cell_to_cellpar(atoms.cell)
    log(f'lengths:  [{a:10.6f}, {b:10.6f}, {c:10.6f}]  # Ang')
    log(f'angles:   [{A:10.6f}, {B:10.6f}, {C:10.6f}]\n')
