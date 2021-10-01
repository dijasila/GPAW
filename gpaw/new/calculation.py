from __future__ import annotations
from gpaw.new.configuration import DFTConfiguration
from gpaw.new.wave_functions import IBZWaveFunctions
from ase.units import Ha, Bohr


class DFTCalculation:
    def __init__(self,
                 cfg: DFTConfiguration,
                 ibz_wfs: IBZWaveFunctions,
                 density,
                 potential):
        self.cfg = cfg
        self.ibz_wfs = ibz_wfs
        self.density = density
        self.potential = potential
        self._scf = None

    @classmethod
    def from_parameters(cls, atoms, params, log) -> DFTCalculation:
        cfg = DFTConfiguration(atoms, params)

        basis_set = cfg.create_basis_set()
        density = cfg.density_from_superposition(basis_set)
        density.normalize()
        pot_calc = cfg.potential_calculator
        potential = pot_calc.calculate(density)

        if params.random:
            ibz_wfs = cfg.random_ibz_wave_functions()
        else:
            ibz_wfs = cfg.lcao_ibz_wave_functions(basis_set, potential)

        return cls(cfg, ibz_wfs, density, potential)

    def energy(self, log):
        energies = self.potential.energies.copy()
        energies['kinetic'] += self.ibz_wfs.e_band
        energies['entropy'] = self.ibz_wfs.e_entropy
        log('\nEnergies (eV):')
        for name, e in energies.items():
            log(f'    {name + ":":10} {e * Ha:14.6f}')
        total_energy = sum(energies.values())
        log(f'    Total:     {total_energy * Ha:14.6f}')
        return total_energy

    def forces(self, log):
        """Return atomic force contributions."""
        xc = self.cfg.xc
        assert not xc.no_forces
        assert not hasattr(xc.xc, 'setup_force_corrections')

        pot_calc = self.cfg.potential_calculator

        # Force from projector functions (and basis set):
        forces = self.ibz_wfs.forces(self.potential.dv)

        # Force from compensation charges:
        ccc = self.density.calculate_compensation_charge_coefficients()
        F_aLv = pot_calc.compensation_charges.derivative(pot_calc.vext)
        for a, dF_Lv in F_aLv.items():
            forces[a] += ccc[a] @ dF_Lv

        # Force from smooth core charge:
        assert self.potential.vt.shape == (1,)
        F_av = self.density.core_acf.derivative(self.potential.vt[0])
        for a, dF_v in F_av.items():
            forces[a] += dF_v[0]

        # Force from zero potential:
        F_av = pot_calc.local_potentials.derivative(pot_calc.total_density2)
        for a, dF_v in F_av.items():
            forces[a] += dF_v[0]

        forces = self.ibz_wfs.ibz.symmetry.symmetry.symmetrize_forces(forces)

        log('\nForces in eV/Ang:')
        c = Ha / Bohr
        for a, setup in enumerate(self.cfg.setups):
            x, y, z = forces[a] * c
            log(f'{a:4} {setup.symbol:2} {x:10.3f} {y:10.3f} {z:10.3f}')
        log()

        return forces

    def move(self, fracpos):
        ...

    @property
    def scf(self):
        if self._scf is None:
            self._scf = self.cfg.scf_loop()
        return self._scf

    def converge(self, log, convergence=None):
        convergence = convergence or self.cfg.params.convergence
        log('\n', self.scf.description)
        density, potential = self.scf.converge(self.ibz_wfs,
                                               self.density,
                                               self.potential,
                                               convergence,
                                               self.cfg.params.maxiter,
                                               log)
        self.density = density
        self.potential = potential

    @staticmethod
    def read(filename, log, parallel):
        ...

    def write_converged(self, log):
        fl = self.ibz_wfs.fermi_levels * Ha
        assert len(fl) == 1
        log(f'\nFermi level: {fl[0]:.3f}\n')

        ibz = self.ibz_wfs.ibz
        for i, point in enumerate(ibz.points):
            log(f'kpt={point}, weight={ibz.weights[i]}')
            eigs, occs = self.ibz_wfs.get_eigs_and_occs(i)
            for n, (e, f) in enumerate(zip(eigs, occs)):
                log(f'{n} {e:10.3f} {f:.3f}')
