from gpaw.hybrids.symmetry import Symmetry as PWSymmetry
from gpaw.kpt_descriptor import KPointDescriptor

class LCAOSymmetry(PWSymmetry):
    def __init__(self, kd: KPointDescriptor):
        PWSymmetry.__init__(self, kd)

        print(self.symmetry_map_ss)
