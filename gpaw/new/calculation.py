from __future__ import annotations
from gpaw.new.configuration import DFTConfiguration
from gpaw.new.wave_functions import IBZWaveFunctions
from ase.units import Ha


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

    def move(self, fracpos):
        ...

    @property
    def scf(self):
        if self._scf is None:
            self._scf = self.cfg.scf_loop()
        return self._scf

    def converge(self, log, convergence=None):
        convergence = convergence or self.cfg.params.convergence
        log(self.scf.description)
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
        log(self.ibz_wfs.fermi_levels * Ha)
        for wfs in self.ibz_wfs:
            log(wfs.eigs * Ha)
            log(wfs.occs)
