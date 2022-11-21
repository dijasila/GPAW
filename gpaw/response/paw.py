import numpy as np

from gpaw.pw.lfc import ft
import numpy as np
from gpaw.gaunt import gaunt
from gpaw.spherical_harmonics import Y


def two_phi_planewave_integrals(k_Gv, setup=None,
                                rgd=None, phi_jg=None,
                                phit_jg=None, l_j=None):

    if setup is not None:
        rgd = setup.rgd
        l_j = setup.l_j
        # Obtain the phi_j and phit_j
        phi_jg = []
        phit_jg = []
        rcut2 = 2 * max(setup.rcut_j)
        gcut2 = rgd.ceil(rcut2)
        for phi_g, phit_g in zip(setup.data.phi_jg, setup.data.phit_jg):
            phi_g = phi_g.copy()
            phit_g = phit_g.copy()
            phi_g[gcut2:] = phit_g[gcut2:] = 0.
            phi_jg.append(phi_g)
            phit_jg.append(phit_g)
    else:
        assert rgd is not None
        assert phi_jg is not None
        assert l_j is not None

    # Construct L (l**2 + m) and j (nl) index
    L_i = []
    j_i = []
    for j, l in enumerate(l_j):
        for m in range(2 * l + 1):
            L_i.append(l**2 + m)
            j_i.append(j)
    ni = len(L_i)
    nj = len(l_j)

    if setup is not None:
        assert ni == setup.ni and nj == setup.nj

    if setup is not None:
        assert ni == setup.ni and nj == setup.nj

    # Initialize
    npw = k_Gv.shape[0]
    phi_Gii = np.zeros((npw, ni, ni), dtype=complex)

    G_LLL = gaunt(max(l_j))
    k_G = np.sum(k_Gv**2, axis=1)**0.5

    i1_start = 0

    for j1, l1 in enumerate(l_j):
        i2_start = 0
        for j2, l2 in enumerate(l_j):
            # Calculate the radial part of the product density
            rhot_g = phi_jg[j1] * phi_jg[j2] - phit_jg[j1] * phit_jg[j2]
            rhot_g[-1] = 0.0
            for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
                spline = rgd.spline(rhot_g, l=l, points=2**10)
                splineG = ft(spline, N=2**12)
                f_G = splineG.map(k_G) * (-1j)**l

                for m1 in range(2 * l1 + 1):
                    i1 = i1_start + m1
                    for m2 in range(2 * l2 + 1):
                        i2 = i2_start + m2
                        G_m = G_LLL[l1**2 + m1, l2**2 + m2, l**2:(l + 1)**2]
                        for m, G in enumerate(G_m):
                            if G == 0: #  If Gaunt coefficient is zero, no need to add
                                continue
                            x_G = Y(l**2 + m, *k_Gv.T) * f_G
                            phi_Gii[:, i1, i2] += G * x_G

            i2_start += 2 * l2 + 1
        i1_start += 2 * l1 + 1
    return phi_Gii.reshape(npw, ni * ni)


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


def get_pair_density_paw_corrections(setups, pd, spos_ac):
    """Calculate and bundle paw corrections to the pair densities as a
    PWPAWCorrectionData object."""
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
