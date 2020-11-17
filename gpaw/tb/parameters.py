from gpaw.tb.repulsion import Repulsion


defaults = {
    ('H', 'H'): [2.75925029e+01, 2.05791019e-02, 4.58405757e-01]}


class DefaultParameters:
    def __getitem__(self, key):
        return Repulsion(*defaults[key])

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class ZeroRepulsion:
    def __getitem__(self, key):
        return Repulsion(0.0, 0.0, 0.01)

    def get(self, key, default=None):
        return Repulsion(0.0, 0.0, 0.01)
