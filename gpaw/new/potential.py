from __future__ import annotations
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_arrays import AtomArrays


class Potential:
    def __init__(self,
                 vt_sR: DistributedArrays,
                 dH_asii: AtomArrays,
                 energies: dict[str, float]):
        self.vt_sR = vt_sR
        self.dH_asii = dH_asii
        self.energies = energies

    def dH(self, P_ain, out, spin):
        for a, I1, I2 in P_ain.layout.myindices:
            dH_ii = self.dH_asii[a][spin]
            out.data[I1:I2] = dH_ii @ P_ain.data[I1:I2]
        return out

    def write(self, writer):
        writer.write(vt=self.vt.collect().data)
