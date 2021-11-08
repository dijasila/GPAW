from __future__ import annotations

from ase.units import Bohr, Ha
from gpaw.new.configuration import DFTConfiguration
from gpaw.new.wave_functions import IBZWaveFunctions
from typing import Any


class DFTCalculation:
    def __init__(self,
                 cfg: DFTConfiguration,
                 ibzwfs: IBZWaveFunctions,
                 density,
                 potential):
        self.cfg = cfg
        self.ibzwfs = ibzwfs
        self.density = density
        self.potential = potential
        self._scf = None
        self.results: dict[str, Any] = {}

    @classmethod
    def from_parameters(cls, atoms, params, log) -> DFTCalculation:
        cfg = DFTConfiguration(atoms, params)

        basis_set = cfg.create_basis_set()

        density = cfg.density_from_superposition(basis_set)
        density.normalize()

        pot_calc = cfg.potential_calculator
        potential = pot_calc.calculate(density)

        if params.random:
            log('Initializing wave functions with random numbers')
            ibzwfs = cfg.random_ibz_wave_functions()
        else:
            ibzwfs = cfg.lcao_ibz_wave_functions(basis_set, potential)

        return cls(cfg, ibzwfs, density, potential)

    def move_atoms(self, atoms, log) -> DFTCalculation:
        cfg = DFTConfiguration(atoms, self.cfg.params)

        if self.cfg.ibz.symmetry != cfg.ibz.symmetry:
            raise ValueError

        self.density.move(cfg.fracpos)
        self.ibzwfs.move(cfg.fracpos)
        self.potential.energies.clear()

        return DFTCalculation(cfg, self.ibzwfs, self.density, self.potential)

    @property
    def scf(self):
        if self._scf is None:
            self._scf = self.cfg.scf_loop()
        return self._scf

    def converge(self, log, convergence=None):
        convergence = convergence or self.cfg.params.convergence
        log(self.scf)
        density, potential = self.scf.converge(self.ibzwfs,
                                               self.density,
                                               self.potential,
                                               convergence,
                                               self.cfg.params.maxiter,
                                               log)
        self.density = density
        self.potential = potential

    def energies(self, log):
        energies1 = self.potential.energies.copy()
        energies2 = self.ibzwfs.energies
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

    def forces(self, log):
        """Return atomic force contributions."""
        xc = self.cfg.xc
        assert not xc.no_forces
        assert not hasattr(xc.xc, 'setup_force_corrections')

        # Force from projector functions (and basis set):
        F_av = self.ibzwfs.forces(self.potential.dH_asii)

        pot_calc = self.cfg.potential_calculator
        Fcc_aLv, Fnct_av, Fvbar_av = pot_calc.forces(self.density.nct_acf)

        # Force from compensation charges:
        ccc_aL = self.density.calculate_compensation_charge_coefficients()
        dF_aLv = pot_calc.ghat_acf.derivative(pot_calc.vHt_x)
        for a, dF_Lv in dF_aLv.items():
            F_av[a] += ccc_aL[a] @ dF_Lv

        # Force from smooth core charge:
        dF_av = self.density.nct_acf.derivative(pot_calc.vt_X)
        for a, dF_v in dF_av.items():
            F_av[a] += dF_v[0]

        # Force from zero potential:
        dF_av = pot_calc.vbar_ax.derivative(pot_calc.nt_x)
        for a, dF_v in dF_av.items():
            F_av[a] += dF_v[0]

        self.cfg.communicators['d'].sum(F_av)

        F_av = self.ibzwfs.ibz.symmetry.symmetry.symmetrize_forces(F_av)

        log('\nForces in eV/Ang:')
        c = Ha / Bohr
        for a, setup in enumerate(self.cfg.setups):
            x, y, z = F_av[a] * c
            log(f'{a:4} {setup.symbol:2} {x:10.3f} {y:10.3f} {z:10.3f}')

        self.results['forces'] = F_av

    def write_converged(self, log):
        fl = self.ibzwfs.fermi_levels * Ha
        assert len(fl) == 1
        log(f'\nFermi level: {fl[0]:.3f}')

        ibz = self.ibzwfs.ibz
        for i, (x, y, z) in enumerate(ibz.points):
            log(f'\nkpt = [{x:.3f}, {y:.3f}, {z:.3f}], '
                f'weight = {ibz.weights[i]:.3f}:')
            log('  Band    eigenvalue   occupation')
            eigs, occs = self.ibzwfs.get_eigs_and_occs(i)
            eigs = eigs * Ha
            occs = occs * self.ibzwfs.spin_degeneracy
            for n, (e, f) in enumerate(zip(eigs, occs)):
                log(f'    {n:4} {e:10.3f}   {f:.3f}')
            if i == 3:
                break

    def ase_interface(self, log):
        from gpaw.new.ase_interface import ASECalculator
        return ASECalculator(self.cfg.params, log, self)
