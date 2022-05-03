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

    def empty(self, nbands, band_comm):
        psit_nsG = self.desc.empty((nbands, 2), band_comm)
        psit_nsG.desc = self
        return psit_nsG

    def atom_centered_functions(self,
                                pt_aj,
                                positions,
                                *,
                                atomdist=None):
        pt_aiX = self.desc.atom_centered_functions(pt_aj, positions,
                                                   atomdist=atomdist)
        return SpiralPWAFC(pt_aiX, self.qspiral_c)


class QPWAFC:
    def __init__(self, pt_aiG, qspiral_c):
        self.pt_aiG = pt_aiG
        self.qspiral_c = qspiral_c


class SpiralPWAFC:
    def __init__(self, pt_aiG, qspiral_c):
        self.ptup_aiG = QPWAFC(pt_aiG, qspiral_c)
        self.ptdn_aiG = QPWAFC(pt_aiG, -qspiral_c)
