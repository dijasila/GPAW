"""Zero-field splitting.

See::

    Spin decontamination for magnetic dipolar coupling calculations:
    Application to high-spin molecules and solid-state spin qubits

    Timur Biktagirov, Wolf Gero Schmidt, and Uwe Gerstmann

    Phys. Rev. Research 2, 022024(R) – Published 30 April 2020

"""
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC
from gpaw.hints import Array2D


def zfs(calc, n1, n2) -> Array2D:
    """"""
    wfs = calc.wfs

    kpt_s = wfs.kpt_qs[0]

    wf1, wf2 = (WaveFunctions.from_kpt(kpt, wfs.setups)
                for kpt in kpt_s)
    compensation_charge = PWLFC([data.ghat_l for data in wfs.setups], wfs.pd)
    compensation_charge.set_positions(wfs.spos_ac)

    zfs1(*wf_s, compensation_charge)


class WaveFuntions:
    def __init__(self, psit, projections, setups):
        pd = psit.pd
        N = len(psit)
        self.psit_nR = pd.gd.empty(N)
        for n, psit_G in enumerate(self.psit.array):
            self.psit_nR[n] = pd.ifft(psit_G)
        self.projections = projections
        self.setups = setups

    @staticmethod
    def from_kpt(kpt, setups):
        return WaveFunctions(kpt.psit, kpt.projections, setups)

    def __len__(self):
        return len(self.psit_nR)


def zfs1(wf1, wf2, compensation_charge) -> Array2D:
    pd = wf1.psit.pd
    G_Gv = pd.get_reciprocal_vectors()
    D_vv = np.zeros((3, 3))

    for n, psit1_R in enumerate(wf1.psit_nR):
        D_anii = {}
        for a, P1_ni in wf1.projections.items():
            D_nii = np.einsum('i, nj -> nij', P1_ni[n1], wf2.projections[a])
            D_anii[a] = D_nii

        n_nG = pd.empty(N)
        for n_G, psit2_R in zip(n_nG, wf2.psit_nR):
            n_G[:] = pf.fft(psit1_R * psit2_R)

        compensation_charge.add(n_nG, Q_anL)

    spin_density_G[0] = 0.0
    G2_G = pd.G2_qG[0].copy()
    G2_G[0] = 1.0
    spin_density_G /= G2_G

    # Anisotropic term:
    W_vva = np.empty((3, 3, len(spos_ac)))
    for v1 in range(3):
        for v2 in range(3):
            W_a = pd.integrate(G_Gv[:, v1] * G_Gv[:, v2] * spin_density_G,
                               eiGR_aG)
            W_vva[v1, v2] = -W_a / gd.dv

    W_a = np.trace(W_vva) / 3
    for v in range(3):
        W_vva[v, v] -= W_a
        W_vva[v, v] += W1_a

    return W_vva.transpose((2, 0, 1))
