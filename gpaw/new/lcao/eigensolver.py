import numpy as np


class LCAOEigensolver:
    def __init__(self, basis):
        self.basis = basis

    def iterate(self, state, hamiltonian) -> float:
        V_sxMM = [self.basis.calculate_potential_matrices(vt_R.data)
                  for vt_R in state.potential.vt_sR]
        dH_saii = [{a: dH_sii[s]
                    for a, dH_sii in state.potential.dH_asii.items()}
                   for s in range(len(V_sxMM))]

        for wfs in state.ibzwfs:
            self.iterate1(wfs, V_sxMM[wfs.spin], dH_saii[wfs.spin])
        return 0.0

    def iterate1(self, wfs, V_xMM, dH_aii):
        H_MM = V_xMM[0]
        if V_xMM.dtype == complex:
            phase_x = np.exp(2j * np.pi *
                             self.basis.sdisp_xc[1:] @ wfs.kpt_c)
            H_MM += np.einsum('x, xMN -> MN',
                              2 * phase_x, V_xMM[1:],
                              optimize=True)

        for a, dH_ii in dH_aii.items():
            P_Mi = wfs.P_aMi[a]
            H_MM += P_Mi.conj() @ dH_ii @ P_Mi.T

        if V_xMM.dtype == complex:
            H_MM *= 0.5
            H_MM += H_MM.conj().T

        H_MM += wfs.T_MM
