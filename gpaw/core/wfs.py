class WaveFunctions:
    def __init__(self, projectors, wave_functions):
        self.projectors = projectors
        self.wave_functions = wave_functions
        self._projections = None

    @property
    def projections(self):
        if self._projections is None:
            self._projections = self.projectors.integrate(self.wave_functions)
        return self._projections
