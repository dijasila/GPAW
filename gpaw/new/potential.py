from __future__ import annotations

import numpy as np
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

    def __str__(self) -> str:
        return (f'potential:\n'
                f'  grid points: {self.vt_sR.desc.size}\n')

    def dH(self, P_ani, out_ani, spin):
        if len(P_ani.dims) == 1:  # collinear wave functions
            for (a, P_ni), out_ni in zip(P_ani.items(), out_ani.values()):
                dH_ii = self.dH_asii[a][spin]
                np.einsum('ni, ij -> nj', P_ni, dH_ii, out=out_ni)
            return out_ani

        # Non-collinear wave functions:
        P_ansi = P_ani
        out_ansi = out_ani

        for (a, P_nsi), out_nsi in zip(P_ansi.items(), out_ansi.values()):
            v_ii, x_ii, y_ii, z_ii = (dh_ii.T for dh_ii in self.dH_asii[a])
            assert v_ii.dtype == complex
            out_nsi[:, 0] = (P_nsi[:, 0] @ (v_ii + z_ii) +
                             P_nsi[:, 1] @ (x_ii - 1j * y_ii))
            out_nsi[:, 1] = (P_nsi[:, 1] @ (v_ii - z_ii) +
                             P_nsi[:, 0] @ (x_ii + 1j * y_ii))
        return out_ansi

    def write(self, writer):
        dH_asp = self.dH_asii.to_lower_triangle().gather()
        vt_sR = self.vt_sR.gather()
        if dH_asp is None:
            return
        writer.write(
            potential=vt_sR.data * Ha,
            atomic_hamiltonian_matrices=dH_asp.data * Ha,
            energies={name: val * Ha for name, val in self.energies.items()})
