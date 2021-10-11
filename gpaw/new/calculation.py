from __future__ import annotations

import ase.io.ulm as ulm
from ase.io.trajectory import write_atoms, read_atoms
from ase.units import Bohr, Ha
from gpaw.new.configuration import DFTConfiguration
from gpaw.new.wave_functions import IBZWaveFunctions
import gpaw
from typing import Any


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
            ibz_wfs = cfg.random_ibz_wave_functions()
        else:
            ibz_wfs = cfg.lcao_ibz_wave_functions(basis_set, potential)

        return cls(cfg, ibz_wfs, density, potential)

    def move_atoms(self, atoms, log) -> DFTCalculation:
        cfg = DFTConfiguration(atoms, self.cfg.params)

        if self.cfg.ibz.symmetry != cfg.ibz.symmetry:
            raise ValueError

        self.density.move(cfg.fracpos)
        self.ibz_wfs.move(cfg.fracpos)
        self.potential.energies.clear()

        return DFTCalculation(cfg, self.ibz_wfs, self.density, self.potential)

    @property
    def scf(self):
        if self._scf is None:
            self._scf = self.cfg.scf_loop()
        return self._scf

    def converge(self, log, convergence=None):
        convergence = convergence or self.cfg.params.convergence
        log(self.scf)
        density, potential = self.scf.converge(self.ibz_wfs,
                                               self.density,
                                               self.potential,
                                               convergence,
                                               self.cfg.params.maxiter,
                                               log)
        self.density = density
        self.potential = potential

    def energies(self, log):
        energies1 = self.potential.energies.copy()
        energies2 = self.ibz_wfs.energies
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

        pot_calc = self.cfg.potential_calculator

        # Force from projector functions (and basis set):
        forces = self.ibz_wfs.forces(self.potential.dv)

        # Force from compensation charges:
        ccc = self.density.calculate_compensation_charge_coefficients()
        F_aLv = pot_calc.ghat_acf.derivative(pot_calc.vHt)
        for a, dF_Lv in F_aLv.items():
            forces[a] += ccc[a] @ dF_Lv

        # Force from smooth core charge:
        F_av = self.density.core_acf.derivative(pot_calc.vt)
        for a, dF_v in F_av.items():
            forces[a] += dF_v[0]

        # Force from zero potential:
        F_av = pot_calc.vbar_acf.derivative(pot_calc.nt)
        for a, dF_v in F_av.items():
            forces[a] += dF_v[0]

        forces = self.ibz_wfs.ibz.symmetry.symmetry.symmetrize_forces(forces)

        log('\nForces in eV/Ang:')
        c = Ha / Bohr
        for a, setup in enumerate(self.cfg.setups):
            x, y, z = forces[a] * c
            log(f'{a:4} {setup.symbol:2} {x:10.3f} {y:10.3f} {z:10.3f}')

        self.results['forces'] = forces

    def write_converged(self, log):
        fl = self.ibz_wfs.fermi_levels * Ha
        assert len(fl) == 1
        log(f'\nFermi level: {fl[0]:.3f}')

        ibz = self.ibz_wfs.ibz
        for i, (x, y, z) in enumerate(ibz.points):
            log(f'\nkpt = [{x:.3f}, {y:.3f}, {z:.3f}], '
                f'weight = {ibz.weights[i]:.3f}:')
            log('  Band    eigenvalue   occupation')
            eigs, occs = self.ibz_wfs.get_eigs_and_occs(i)
            eigs = eigs * Ha
            occs = occs * self.ibz_wfs.spin_degeneracy
            for n, (e, f) in enumerate(zip(eigs, occs)):
                log(f'    {n:4} {e:10.3f}   {f:.3f}')
            if i == 3:
                break

    def write(self,
              filename: str,
              skip_wfs: bool = True):
        world = self.cfg.communicators['w']
        if world.rank == 0:
            writer = ulm.Writer(filename, tag='gpaw')
        else:
            writer = ulm.DummyWriter()
        with writer:
            writer.write(version=4,
                         gpaw_version=gpaw.__version__,
                         ha=Ha,
                         bohr=Bohr)

            write_atoms(writer.child('atoms'), self.cfg.atoms)
            writer.child('results').write(**self.results)
            writer.child('parameters').write(**self.cfg.params.params)

            self.density.write(writer.child('density'))
            self.potential.write(writer.child('hamiltonian'))
            self.ibz_wfs.write(writer.child('wave_functions'), skip_wfs)

        world.barrier()

    @classmethod
    def read(cls, filename, log):
        log(f'Reading from {filename}')
        reader = ulm.Reader(filename)
        atoms = read_atoms(reader.atoms)

        print(reader.parameters)
        cfg = DFTConfiguration(atoms, dict(reader.parameters))
        res = reader.results
        print(res)
        results = dict((key, res.get(key)) for key in res.keys())
        if results:
            log('Read {}'.format(', '.join(sorted(results))))

        self.log('Reading input parameters:')
        self.density.read(reader)
        self.hamiltonian.read(reader)
        self.scf.read(reader)
        self.wfs.read(reader)

        return cls(cfg, ibz_wfs, density, potential)

    def ase_interface(self, log):
        from gpaw.new.ase_interface import ASECalculator
        return ASECalculator(self.cfg.params, log, self)
