from gpaw.tb.repulsion import ExpRepulsion


defaults = {
    ('H', 'H'): [2.75925029e+01, 2.05791019e-02, 4.58405757e-01]}


class DefaultParameters(dict):
    def __getitem__(self, key):
        return ExpRepulsion(*defaults[key])

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
