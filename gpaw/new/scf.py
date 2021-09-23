class SCFLoop:
    def __init__(self,
                 ibzwfs,
                 density,
                 potential,
                 hamiltonian,
                 eigensolver,
                 mixer):
        self.ibzwfs = ibzwfs
        self.potential = potential
        self.eigensolver = eigensolver

    def converge(self, conv_criteria):
        yield from self.iconverge(conv_criteria)

    def iconverge(self, conv_criteria):
        while True:
            self.eigensolver.iterate(self.potential, self.ibzwfs)
            yield
