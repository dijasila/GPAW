from gpaw.new.configuration import DFTConfiguration
from gpaw.new.scf import calculate_energy


class DFTCalculation:
    def __init__(self,
                 cfg,
                 ibz_wfs,
                 density,
                 potential):
        self.cfg = cfg
        self.ibz_wfs = ibz_wfs
        self.density = density
        self.potential = potential
        self._scf = None

    @classmethod
    def from_parameters(cls, atoms, params, log):
        cfg = DFTConfiguration(atoms, params)

        density = cfg.density_from_superposition()
        pot_calc = cfg.potential_calculator
        potential = pot_calc.calculate(density)

        if params.random:
            ibz_wfs = cfg.random_ibz_wave_functions()
        else:
            ...

        return cls(cfg, ibz_wfs, density, potential)

    def energy(self):
        return calculate_energy(self.ibz_wfs, self.potential)

    def move(self, fracpos):
        ...

    @property
    def scf(self):
        if self._scf is None:
            self._scf = self.cfg.scf_loop()
        return self._scf

    def converge(self, log, convergence=None):
        convergence = convergence or self.cfg.params.convergence

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
