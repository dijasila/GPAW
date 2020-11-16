from gpaw.wavefunctions.mode import Mode
from gpaw.tb.wavefunctions import TBWaveFunctions


class TB(Mode):
    name = 'tb'
    interpolation = 1
    force_complex_dtype = False

    def __init__(self) -> None:
        pass

    def __call__(self, ksl, **kwargs) -> TBWaveFunctions:
        return TBWaveFunctions(ksl, **kwargs)
