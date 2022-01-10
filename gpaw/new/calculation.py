from __future__ import annotations

from typing import Any

import numpy as np
from ase.units import Bohr, Ha
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.input_parameters import InputParameters
from gpaw.new.wave_functions import IBZWaveFunctions
from gpaw.new.potential import Potential


class DFTState:
    def __init__(self,
                 ibzwfs: IBZWaveFunctions,
                 density,
                 potential: Potential,
                 vHt_x=None,
                 nct_R=None):
        """State of a Kohn-Sham calculation."""
        self.ibzwfs = ibzwfs
        self.density = density
        self.potential = potential
        self.vHt_x = vHt_x  # initial guess for Hartree potential

    def move(self, fracpos_ac, delta_nct_R):
        self.ibzwfs.move(fracpos_ac)
        self.potential.energies.clear()
        self.density.move(delta_nct_R)


class DFTCalculation:
    def __init__(self,
                 state: DFTState,
                 setups,
                 scf_loop,
                 pot_calc):
        self.state = state
        self.setups = setups
        self.scf_loop = scf_loop
        self.pot_calc = pot_calc

        self.results: dict[str, Any] = {}

    @classmethod
    def from_parameters(cls,
                        atoms,
                        params=None,
                        log=None,
                        builder=None) -> DFTCalculation:

        if isinstance(params, dict):
            params = InputParameters(params)

        builder = builder or DFTComponentsBuilder(atoms, params)

        basis_set = builder.create_basis_set()

        density = builder.density_from_superposition(basis_set)
        density.normalize()

        pot_calc = builder.create_potential_calculator()
        potential, vHt_x, _ = pot_calc.calculate(density)

        if params.random:
            log('Initializing wave functions with random numbers')
            ibzwfs = builder.random_ibz_wave_functions()
        else:
            ibzwfs = builder.lcao_ibz_wave_functions(basis_set, potential)

        write_atoms(atoms, builder.grid, builder.initial_magmoms, log)

        log(ibzwfs.ibz.symmetries)
        log(ibzwfs)

        return cls(DFTState(ibzwfs, density, potential, vHt_x),
                   builder.setups,
                   builder.create_scf_loop(pot_calc),
                   pot_calc)

    def move_atoms(self, atoms, log) -> DFTCalculation:
        self.fracpos_ac = atoms.get_scaled_positions()

        delta_nct_R = self.pot_calc.move(self.fracpos_ac,
                                         self.state.density.ndensities)
        self.state.move(self.fracpos_ac, delta_nct_R)

        magmoms = self.results.get('magmoms')
        write_atoms(atoms,
                    self.state.density.nt_sR.desc,
                    magmoms,
                    log)

        self.results = {}

        return self

    def iconverge(self, log, convergence=None, maxiter=None):
        log(self.scf_loop)
        for ctx in self.scf_loop.iterate(self.state,
                                         convergence,
                                         maxiter,
                                         log=log):
            yield ctx

    def converge(self,
                 log,
                 convergence=None,
                 maxiter=None,
                 steps=99999999999999999):
        """Converge to self-consistent solution of KS-equation."""
        for step, _ in enumerate(self.iconverge(log, convergence, maxiter),
                                 start=1):
            if step == steps:
                break
        else:  # no break
            log(f'Converged in {step} steps')

    def energies(self, log):
        energies1 = self.state.potential.energies.copy()
        energies2 = self.state.ibzwfs.energies
        energies1['kinetic'] += energies2['band']
        energies1['entropy'] = energies2['entropy']
        free_energy = sum(energies1.values())
        extrapolated_energy = free_energy + energies2['extrapolation']

        log('\nEnergies (eV):')
        for name, e in energies1.items():
            log(f'    {name + ":":10}   {e * Ha:14.6f}')
        log(f'    Total:       {free_energy * Ha:14.6f}')
        log(f'    Extrapolated:{extrapolated_energy * Ha:14.6f}')

        self.results['free_energy'] = free_energy
        self.results['energy'] = extrapolated_energy

    def dipole(self, log):
        dipole_v = self.density.calculate_dipole_moment() * Bohr
        self.log('Dipole moment: ({:.6f}, {:.6f}, {:.6f}) |e|*Ang\n'
                 .format(*dipole_v))
        self.results['dipole'] = dipole_v

    def magmoms(self, log):
        mm_v, mm_av = self.state.density.calculate_magnetic_moments()
        self.results['magmom'] = mm_v[2]
        self.results['magmoms'] = mm_av[:, 2].copy()

        if self.state.density.ncomponents > 1:
            x, y, z = mm_v
            log(f'Total magnetic moment: ({x:.6f}, {y:.6f}, {z:.6f})')
            log('Local magnetic moments:')
            for a, (setup, m_v) in enumerate(zip(self.setups, mm_av)):
                x, y, z = m_v
                log(f'{a:4} {setup.symbol:2} ({x:9.6f}, {y:9.6f}, {z:9.6f})')
            log()

    def forces(self, log):
        """Return atomic force contributions."""
        xc = self.pot_calc.xc
        assert not xc.no_forces
        assert not hasattr(xc.xc, 'setup_force_corrections')

        # Force from projector functions (and basis set):
        F_av = self.state.ibzwfs.forces(self.state.potential.dH_asii)

        pot_calc = self.pot_calc
        Fcc_aLv, Fnct_av, Fvbar_av = pot_calc.force_contributions(
            self.state)

        # Force from compensation charges:
        ccc_aL = \
            self.state.density.calculate_compensation_charge_coefficients()
        for a, dF_Lv in Fcc_aLv.items():
            F_av[a] += ccc_aL[a] @ dF_Lv

        # Force from smooth core charge:
        for a, dF_v in Fnct_av.items():
            F_av[a] += dF_v[0]

        # Force from zero potential:
        for a, dF_v in Fvbar_av.items():
            F_av[a] += dF_v[0]

        domain_comm = ccc_aL.layout.atomdist.comm
        domain_comm.sum(F_av)

        F_av = self.state.ibzwfs.ibz.symmetries.symmetrize_forces(F_av)

        log('\nForces [eV/Ang]:')
        c = Ha / Bohr
        for a, setup in enumerate(self.setups):
            x, y, z = F_av[a] * c
            log(f'{a:4} {setup.symbol:2} {x:10.3f} {y:10.3f} {z:10.3f}')

        self.results['forces'] = F_av

    def write_converged(self, log):
        self.state.ibzwfs.write_summary(log)


def write_atoms(atoms, grid, magmoms, log):
    from gpaw.output import print_cell, print_positions
    if magmoms is None:
        magmoms = np.zeros((len(atoms), 3))
    elif magmoms.ndim == 1:
        m1 = magmoms
        magmoms = np.zeros((len(atoms), 3))
        magmoms[:, 2] = m1
    print_positions(atoms, log, magmoms)
    print_cell(grid._gd, atoms.pbc, log)
