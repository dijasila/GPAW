class SCFLoop:
    def __init__(self,
                 ibzwfs,
                 density,
                 potential,
                 hamiltonian,
                 eigensolver,
                 mixer):
        ibzwfs.mykpts[0].orthonormalize()

    def converge(self, conv_criteria):
        ...
