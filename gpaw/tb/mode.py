from typing import Dict, Tuple

from gpaw.wavefunctions.mode import Mode
from gpaw.tb.wavefunctions import TBWaveFunctions
from gpaw.tb.repulsion import Repulsion
from gpaw.tb.parameters import DefaultParameters


class TB(Mode):
    name = 'tb'
    interpolation = 1
    force_complex_dtype = False

    def __init__(self, parameters: Dict[Tuple[str, str], Repulsion] = None):
        if parameters is None:
            parameters = DefaultParameters()
        self.parameters = parameters

    def __call__(self, ksl, xc, **kwargs) -> TBWaveFunctions:
        return TBWaveFunctions(xc, ksl, **kwargs)
