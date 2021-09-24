from functools import partial


class SCFLoop:
    def __init__(self,
                 ibzwfs,
                 density,
                 potential,
                 hamiltonian,
                 pot_calc,
                 eigensolver,
                 mixer):
        self.ibzwfs = ibzwfs
        self.density = density
        self.potential = potential
        self.hamiltonian = hamiltonian
        self.eigensolver = eigensolver

    def converge(self, conv_criteria):
        for _ in self.iconverge(conv_criteria):
            pass

    def iconverge(self, conv_criteria):
        dS = self.density.setups.overlap_correction
        dH = self.potential.dH
        Ht = partial(self.hamiltonian.apply, self.potential.vt)
        while True:
            self.eigensolver.iterate(self.ibzwfs, Ht, dH, dS)
            yield
