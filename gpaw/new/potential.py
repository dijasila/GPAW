from __future__ import annotations

from ase.units import Ha
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_arrays import AtomArrays
from gpaw.new import zip


class Potential:
    def __init__(self,
                 vt_sR: DistributedArrays,
                 dH_asii: AtomArrays,
                 energies: dict[str, float]):
        self.vt_sR = vt_sR
        self.dH_asii = dH_asii
        self.energies = energies

    def __repr__(self):
        return f'Potential({self.vt_sR}, {self.dH_asii}, {self.energies})'

    def dH(self, P_ani, out_ani, spin):
        print('dH', spin)
        for (a, P_ni), out_ni in zip(P_ani.items(), out_ani.values()):
            dH_ii = self.dH_asii[a][spin]
            out_ni[:] = P_ni @ dH_ii
        return out_ani
        .......
        if len(P_ani.dims) == 2:  # (band, spinor)
            subscripts = 'nsi, ij -> nsj'
        else:
            subscripts = 'ni, ij -> nj'
        for (a, P_ni), out_ni in zip(P_ani.items(), out_ani.values()):
            dH_ii = self.dH_asii[a][spin]
            dS_ii = self[a].dO_ii
            np.einsum(subscripts, P_ni, dS_ii, out=out_ni)
        return out_ani

    def write(self, writer):
        dH_asp = self.dH_asii.to_lower_triangle().gather()
        vt_sR = self.vt_sR.gather()
        if dH_asp is None:
            return
        writer.write(
            potential=vt_sR.data * Ha,
            atomic_hamiltonian_matrices=dH_asp.data,
            energies={name: val * Ha for name, val in self.energies.items()})
