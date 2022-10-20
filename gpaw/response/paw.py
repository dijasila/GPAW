import numpy as np
from gpaw.response.math_func import two_phi_planewave_integrals


class PWPAWCorrectionData:
    def __init__(self, Q_aGii, pd, setups, pos_av):
        # Sometimes we loop over these in ways that are very dangerous.
        # It must be list, not dictionary.
        assert isinstance(Q_aGii, list)
        assert len(Q_aGii) == len(pos_av) == len(setups)

        self.Q_aGii = Q_aGii

        self.pd = pd
        self.setups = setups
        self.pos_av = pos_av

    def _new(self, Q_aGii):
        return PWPAWCorrectionData(Q_aGii, pd=self.pd, setups=self.setups,
                                   pos_av=self.pos_av)

    def remap_somehow(self, M_vv, G_Gv, sym, sign):
        Q_aGii = []
        for a, Q_Gii in enumerate(self.Q_aGii):
            x_G = self._get_x_G(G_Gv, M_vv, self.pos_av[a])
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

    def _get_x_G(self, G_Gv, M_vv, pos_v):
        # This doesn't really belong here.  Or does it?  Maybe this formula
        # is only used with PAW corrections.
        return np.exp(1j * (G_Gv @ (pos_v - M_vv @ pos_v)))

    def remap_somehow_else(self, symop, G_Gv, M_vv):
        myQ_aGii = []
        for a, Q_Gii in enumerate(self.Q_aGii):
            x_G = self._get_x_G(G_Gv, M_vv, self.pos_av[a])
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

        C1_aGi = [Qa_Gii @ P1_ni[band].conj()
                  for Qa_Gii, P1_ni in zip(self.Q_aGii, P_ani)]
        return C1_aGi

    def reduce_ecut(self, G2G):
        # XXX actually we should return this with another PW descriptor.
        return self._new([Q_Gii.take(G2G, axis=0) for Q_Gii in self.Q_aGii])

    def almost_equal(self, otherpawcorr, G_G):
        for a, Q_Gii in enumerate(otherpawcorr.Q_aGii):
            e = abs(self.Q_aGii[a] - Q_Gii[G_G]).max()
            if e > 1e-12:
                return False
        return True


def get_pair_density_paw_corrections(setups, pd, spos_ac,
                                     alter_optical_limit=False):
    """Calculate and bundle paw corrections to the pair densities as a
    PWPAWCorrectionData object.

    NB: Due to the convolution of head, wings and body in Chi0, this method
    can alter the G=0 element in the optical limit. This is a bad behaviour
    and should be changed in the future when things are properly separated.
    Ultimately, this function will be redundant and
    calculate_pair_density_paw_corrections() could be used directly.
    """
    pawcorr = calculate_pair_density_paw_corrections(setups, pd, spos_ac)

    if alter_optical_limit:
        q_v = pd.K_qv[0]
        optical_limit = np.allclose(q_v, 0)
        if optical_limit:
            Q_aGii = pawcorr.Q_aGii.copy()

            for a, atomdata in enumerate(setups):
                Q_aGii[a][0] = atomdata.dO_ii

            pawcorr = PWPAWCorrectionData(Q_aGii, pd=pd, setups=setups,
                                          pos_av=pawcorr.pos_av)

    return pawcorr


def calculate_pair_density_paw_corrections(setups, pd, spos_ac):
    G_Gv = pd.get_reciprocal_vectors()
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
        x_G = np.exp(-1j * (G_Gv @ pos_av[a]))
        Q_aGii.append(x_G[:, np.newaxis, np.newaxis] * Q_Gii)

    return PWPAWCorrectionData(Q_aGii, pd=pd, setups=setups, pos_av=pos_av)
