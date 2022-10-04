import numpy as np
from gpaw.response.math_func import two_phi_planewave_integrals


class PAWCorrections:
    def __init__(self, Q_aGii):
        # Sometimes we loop over these in ways that are very dangerous.
        # It must be list, not dictionary.
        assert isinstance(Q_aGii, list)
        self.Q_aGii = Q_aGii

    def remap_somehow(self, setups, pos_av, M_vv, G_Gv, sym, sign):
        # This method is envious of setups and spos, which were used to create
        # PAWCorrections in the first place.  We can conclude that PAWCorrections
        # should likely store both on self.

        Q_aGii = []
        for a, Q_Gii in enumerate(self.Q_aGii):
            x_G = np.exp(1j * np.dot(G_Gv, (pos_av[a] -
                                            np.dot(M_vv, pos_av[a]))))
            U_ii = setups[a].R_sii[sym]

            Q_Gii = np.einsum('ij,kjl,ml->kim',
                              U_ii,
                              Q_Gii * x_G[:, None, None],
                              U_ii,
                              optimize='optimal')
            if sign == -1:
                Q_Gii = Q_Gii.conj()
            Q_aGii.append(Q_Gii)
        return PAWCorrections(Q_aGii)

    def multiply(self, P_ani, band):
        assert isinstance(P_ani, list)
        assert len(P_ani) == len(self.Q_aGii)

        C1_aGi = [np.dot(Qa_Gii, P1_ni[band].conj())
                  for Qa_Gii, P1_ni in zip(self.Q_aGii, P_ani)]
        return C1_aGi


def calculate_paw_corrections(setups, pd, spos_ac):
    q_v = pd.K_qv[0]
    optical_limit = np.allclose(q_v, 0)

    G_Gv = pd.get_reciprocal_vectors()
    if optical_limit:
        G_Gv[0] = 1

    pos_av = spos_ac @ pd.gd.cell_cv

    # Collect integrals for all species:
    Q_xGii = {}
    for id, atomdata in setups.setups.items():
        ni = atomdata.ni
        Q_Gii = two_phi_planewave_integrals(G_Gv, atomdata)
        Q_xGii[id] = Q_Gii.reshape(-1, ni, ni)

    Q_aGii = []
    for a, atomdata in enumerate(setups):
        id = setups.id_a[a]
        Q_Gii = Q_xGii[id]
        x_G = np.exp(-1j * np.dot(G_Gv, pos_av[a]))
        Q_aGii.append(x_G[:, np.newaxis, np.newaxis] * Q_Gii)
        if optical_limit:
            Q_aGii[a][0] = atomdata.dO_ii

    return Q_aGii
