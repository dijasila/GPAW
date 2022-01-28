class LCAOEigensolver:

    def iterate(self, state, hamiltonian) -> float:
        for wfs in state.ibzwfs:
            self.iterate1(wfs)
        return 0.0

    def iterate1(self, wfs):
        pass
