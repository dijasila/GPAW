from ase.units import Ha

from .eigensolver import DirectLCAO


class Scissors(DirectLCAO):
    def __init__(self, components):
        DirectLCAO.__init__(self)
        self.components = []
        for homo, lumo, calc in components:
            nocc = calc.setups.nvalence // 2
            C_qnM = [kpt.C_nM for kpt in calc.wfs.kpt_u]
            natoms = len(calc.atoms)
            self.components.append((homo / Ha, lumo / Ha, nocc, natoms, C_qnM))

    def write(self, writer):
        writer.write(name='lcao')

    def __repr__(self):
        txt = DirectLCAO.__repr__(self)
        txt += '\n    Scissors operators:\n'
        a1 = 0
        for homo, lumo, _, natoms, _ in self.components:
            a2 = a1 + natoms
            txt += (f'      Atoms {a1}-{a2 - 1}: '
                    f'VB: {homo * Ha:+.3f} eV, '
                    f'CB: {lumo * Ha:+.3f} eV\n')
            a1 = a2
        return txt

    def calculate_hamiltonian_matrix(self, hamiltonian, wfs, kpt, Vt_xMM=None,
                                     root=-1, add_kinetic=True):
        H_MM = DirectLCAO.calculate_hamiltonian_matrix(
            self, hamiltonian, wfs, kpt, Vt_xMM, root, add_kinetic)
        S_MM = wfs.S_qMM[kpt.q]
        assert abs(S_MM - S_MM.T.conj()).max() < 1e-10

        M1 = 0
        for homo, lumo, nocc, _, C_qnM in self.components:
            C_nM = C_qnM[kpt.q]
            M2 = M1 + C_nM.shape[1]
            D_oM = C_nM[:nocc].dot(S_MM[M1:M2])
            H_MM += D_oM.T.conj().dot(D_oM) * homo
            D_uM = C_nM[nocc:].dot(S_MM[M1:M2])
            H_MM += D_uM.T.conj().dot(D_uM) * lumo
            M1 = M2

        return H_MM
