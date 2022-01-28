class LCAOEigensolver:

    def iterate(self, ibzwfs, Ht, dH, dS) -> float:

        for wfs in ibzwfs:
            self.iterate1(wfs, dH, dS)
        return 0.0

    def iterate1(self, wfs, dH, dS):
        pass
