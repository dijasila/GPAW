from __future__ import annotations

from ase.units import Ha
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

    def dH(self, P_ani, out, spin):
        for a, I1, I2 in P_ani.layout.myindices:
            dH_ii = self.dH_asii[a][spin]
            out.data[:, I1:I2] = P_ani.data[:, I1:I2] @ dH_ii
        return out

    def write(self, writer):
        dH_asp = self.dH_asii.to_lower_triangle().gather()
        vt_sR = self.vt_sR.gather()
        if dH_asp is None:
            return
        writer.write(
            potential=vt_sR.data * Ha,
            atomic_hamiltonian_matrices=dH_asp.data,
            energies={name: val * Ha for name, val in self.energies.items()})
