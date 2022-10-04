import numpy as np
from gpaw.response.math_func import two_phi_planewave_integrals


class PAWCorrections:
    def __init__(self, Q_aGii, pd, setups, pos_av):
        # Sometimes we loop over these in ways that are very dangerous.
        # It must be list, not dictionary.
        assert isinstance(Q_aGii, list)
        self.Q_aGii = Q_aGii

        self.pd = pd
        self.setups = setups
        self.pos_av = pos_av

    def _new(self, Q_aGii):
        return PAWCorrections(Q_aGii, pd=self.pd, setups=self.setups,
                              pos_av=self.pos_av)

    def remap_somehow(self, M_vv, G_Gv, sym, sign):
        # This method is envious of setups and spos, which were used to create
        # PAWCorrections in the first place.  We can conclude that PAWCorrections
        # should likely store both on self.

        Q_aGii = []
        for a, Q_Gii in enumerate(self.Q_aGii):
            x_G = np.exp(1j * np.dot(G_Gv, (self.pos_av[a] -
                                            np.dot(M_vv, self.pos_av[a]))))
            U_ii = self.setups[a].R_sii[sym]

            Q_Gii = np.einsum('ij,kjl,ml->kim',
                              U_ii,
                              Q_Gii * x_G[:, None, None],
                              U_ii,
                              optimize='optimal')
            if sign == -1:
                Q_Gii = Q_Gii.conj()
            Q_aGii.append(Q_Gii)

        return self._new(Q_aGii)

    def remap_somehow_else(self, symop, G_Gv, M_vv):
        myQ_aGii = []
        for a, Q_Gii in enumerate(self.Q_aGii):
            x_G = np.exp(1j * np.dot(G_Gv, (self.pos_av[a] -
                                            np.dot(M_vv, self.pos_av[a]))))
            U_ii = self.setups[a].R_sii[symop.symno]
            Q_Gii = np.dot(np.dot(U_ii, Q_Gii * x_G[:, None, None]),
                           U_ii.T).transpose(1, 0, 2)
            if symop.sign == -1:
                Q_Gii = Q_Gii.conj()
            myQ_aGii.append(Q_Gii)
        return self._new(myQ_aGii)

    def multiply(self, P_ani, band):
        assert isinstance(P_ani, list)
        assert len(P_ani) == len(self.Q_aGii)

        C1_aGi = [np.dot(Qa_Gii, P1_ni[band].conj())
                  for Qa_Gii, P1_ni in zip(self.Q_aGii, P_ani)]
        return C1_aGi

    def reduce_ecut(self, G2G):
        Q_aGii = []
        if Q_aGii is not None:
            for a, Q_Gii in enumerate(self.Q_aGii):
                Q_aGii.append(Q_Gii.take(G2G, axis=0))

        # XXX actually we should return this with another PW descriptor.
        return self._new(Q_aGii)

    def almost_equal(self, otherpawcorr, G_G):
        for a, Q_Gii in enumerate(otherpawcorr.Q_aGii):
            e = abs(self.Q_aGii[a] - Q_Gii[G_G]).max()
            if e > 1e-12:
                return False
        return True


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

    return PAWCorrections(Q_aGii, pd=pd, setups=setups, pos_av=pos_av)
