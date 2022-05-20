class FakeWFS:
    def __init__(self, calculation):
        self.setups = calculation.setups
        self.state = calculation.state

    @property
    def kpt_u(self):
        return KPTU(self.state.ibzwfs.wfs_qs)


class KPTU:
    def __init__(self, wfs_qs):
        self.wfs_qs = wfs_qs

    def __getitem__(self, u):
        assert len(self.wfs_qs[0]) == 1
        return KPT(self.wfs_qs[u][0])


class KPT:
    def __init__(self, wfs):
        self.wfs = wfs
        self.projections = wfs.P_ani
