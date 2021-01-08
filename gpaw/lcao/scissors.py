import numpy as np
from ase.units import Ha

from .eigensolver import DirectLCAO


class Scissors(DirectLCAO):
    def __init__(self, components):
        DirectLCAO.__init__(self)
        self.components = []
        for homo, lumo, natoms in components:
            self.components.append((homo / Ha, lumo / Ha, natoms))
        self.first_time = True

    def write(self, writer):
        writer.write(name='lcao')

    def __repr__(self):
        txt = DirectLCAO.__repr__(self)
        txt += '\n    Scissors operators:\n'
        a1 = 0
        for homo, lumo, natoms in self.components:
            a2 = a1 + natoms
            txt += (f'      Atoms {a1}-{a2 - 1}: '
                    f'VB: {homo * Ha:+.3f} eV, '
                    f'CB: {lumo * Ha:+.3f} eV\n')
            a1 = a2
        return txt

    def calculate_hamiltonian_matrix(self, ham, wfs, kpt, Vt_xMM=None,
                                     root=-1, add_kinetic=True):
        H_MM = DirectLCAO.calculate_hamiltonian_matrix(
            self, ham, wfs, kpt, Vt_xMM, root, add_kinetic)
        if self.first_time:
            self.first_time = False
            return H_MM

        S_MM = wfs.S_qMM[kpt.q]
        assert abs(S_MM - S_MM.T.conj()).max() < 1e-10

        nocc = ham.setups.nvalence // 2
        C_nM = kpt.C_nM
        iC_Mn = np.linalg.inv(C_nM)
        Co_nM = C_nM.copy()
        Co_nM[nocc:] = 0.0
        Cu_nM = C_nM.copy()
        Cu_nM[:nocc] = 0.0

        print(C_nM)
        print(iC_Mn)
        M1 = 0
        a1 = 0
        for homo, lumo, natoms in self.components:
            a2 = a1 + natoms
            M2 = M1 + sum(setup.nao for setup in ham.setups[a1:a2])

            D_MM = np.zeros_like(S_MM)
            D_MM[M1:M2, M1:M2] = S_MM[M1:M2, M1:M2]

            H_MM += iC_Mn.conj() @ (
                Co_nM.conj() @ D_MM @ Co_nM.T * homo +
                Cu_nM.conj() @ D_MM @ Cu_nM.T * lumo) @ iC_Mn.T

            a1 = a2
            M1 = M2

        return H_MM
