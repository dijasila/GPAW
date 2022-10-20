import numpy as np
from ase.dft.kpoints import monkhorst_pack


class GammaIntegrator:
    def __init__(self, truncation, kd, pd, chi0_wvv, chi0_wxvG):
        N = 4
        N_c = np.array([N, N, N])
        if truncation is not None:
            # Only average periodic directions if trunction is used
            N_c[kd.N_c == 1] = 1
        qf_qc = monkhorst_pack(N_c) / kd.N_c
        qf_qc *= 1.0e-6
        U_scc = kd.symmetry.op_scc
        qf_qc = kd.get_ibz_q_points(qf_qc, U_scc)[0]
        self.weight_q = kd.q_weights
        self.qf_qv = 2 * np.pi * (qf_qc @ pd.gd.icell_cv)
        self.a_wq = np.sum([chi0_vq * self.qf_qv.T
                            for chi0_vq in
                            np.dot(chi0_wvv, self.qf_qv.T)],
                           axis=1)
        self.a0_qwG = np.dot(self.qf_qv, chi0_wxvG[:, 0])
        self.a1_qwG = np.dot(self.qf_qv, chi0_wxvG[:, 1])
