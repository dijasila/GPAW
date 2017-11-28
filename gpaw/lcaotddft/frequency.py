import numpy as np

from gpaw.tddft.units import eV_to_au

class Frequency(object):
    def __init__(self, frequency, folding, width, units='eV'):
        self.frequency = frequency

        if width is None:
            folding = None

        self.folding = folding
        if self.folding is None:
            self.width = None
        else:
            self.width = width

        if units == 'eV':
            for arg in ['frequency', 'width']:
                if getattr(self, arg) is not None:
                    setattr(self, arg, getattr(self, arg) * eV_to_au)
        elif units != 'au':
            raise RuntimeError('Unknown units: %s' % units)

        if not self.folding in [None, 'Gauss', 'Lorentz']:
            raise RuntimeError('Unknown folding: %s' % self.folding)

    def envelope(self, time):
        if self.folding is None:
            return 1.0
        elif self.folding == 'Gauss':
            return np.exp(- 0.5 * self.width**2 * time**2)
        elif self.folding == 'Lorentz':
            return np.exp(- self.width * time)

    def todict(self):
        d = dict(units='au')
        for arg in ['frequency', 'folding', 'width']:
            d[arg] = getattr(self, arg)
        return d
