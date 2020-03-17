from ase.units import Ha

from .eigensolver import DirectLCAO


class Scissors(DirectLCAO):
    def __init__(self, components):
        DirectLCAO.__init__(self)
        self.components = []
        for homo, lumo, calc in components:
            nocc = calc.setups.nvalence // 2
            C_qnM = [kpt.C_nM for kpt in calc.wfs.mykpts]
            print(homo, lumo, nocc)
            self.components.append((homo / Ha, lumo / Ha, nocc, C_qnM))

    def calculate_hamiltonian_matrix(self, hamiltonian, wfs, kpt, Vt_xMM=None,
                                     root=-1, add_kinetic=True):
        H_MM = DirectLCAO.calculate_hamiltonian_matrix(
            self, hamiltonian, wfs, kpt, Vt_xMM, root, add_kinetic)
        S_MM = wfs.S_qMM[kpt.q]

        M1 = 0
        for homo, lumo, nocc, C_qnM in self.components:
            C_nM = C_qnM[kpt.q]
            M2 = M1 + C_nM.shape[1]
            D_oM = C_nM[:nocc].dot(S_MM[M1:M2])
            H_MM += D_oM.T.dot(D_oM) * homo
            D_uM = C_nM[nocc:].dot(S_MM[M1:M2])
            H_MM += D_uM.T.dot(D_uM) * lumo
            M1 = M2

        return H_MM
