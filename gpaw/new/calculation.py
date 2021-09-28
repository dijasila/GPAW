from gpaw.new.configuration import DFTConfiguration


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
        1 / 0

    def move(self, fracpos):
        ...

    def converge(self, log):
        if self._scf is None:
            self._scf = self.cfg.scf_loop()
        scf = self._scf

        for _ in scf.iconverge(self.ibz_wfs, self.density, self.potential,
                               log):
            pass

    @staticmethod
    def read(filename, log, parallel):
        ...
