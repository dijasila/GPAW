from typing import Dict, Tuple

from gpaw.wavefunctions.mode import Mode
from gpaw.tb.wavefunctions import TBWaveFunctions
from gpaw.tb.repulsion import Repulsion


class TB(Mode):
    name = 'tb'
    interpolation = 1
    force_complex_dtype = False

    def __init__(self, parameters: Dict[Tuple[str, str], Repulsion] = None):
        self.parameters = parameters

    def __call__(self, ksl, **kwargs) -> TBWaveFunctions:
        return TBWaveFunctions(ksl, **kwargs)
