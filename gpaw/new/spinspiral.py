from gpaw.core.domain import Domain
from gpaw.new.spinors import SpinorWaveFunctionDescriptor
from gpaw.typing import Vector


class SpinSpiralWaveFunctionDescriptor(SpinorWaveFunctionDescriptor):
    def __init__(self,
                 desc: Domain,
                 qspiral: Vector):
        super().__init__(desc)
        self.qspiral_c = qspiral

    def new(self, *, kpt):
        return SpinSpiralWaveFunctionDescriptor(self.desc.new(kpt=kpt),
                                                self.qspiral_c)

    def atom_centered_functions(self,
                                pt_aj,
                                positions,
                                *,
                                atomdist=None):
        pt_aiX = self.desc.atom_centered_functions(pt_aj, positions,
                                                   atomdist=atomdist)
        return pt_aiX
