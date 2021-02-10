import numpy as np
from gpaw.setup import create_setup
from gpaw.gaunt import gaunt


def coulomb(rgd, phi_jg, phit_jg, l_j, gt0_lg):
    gcut = gt0_lg.shape[1]
    phi_jg[:, gcut:] = 0.0
    phit_jg[:, gcut:] = 0.0
    n_jjg = np.einsum('jg, kg -> jkg', phi_jg, phi_jg) / (4 * np.pi)
    nt_jjg = np.einsum('jg, kg -> jkg', phit_jg, phit_jg) / (4 * np.pi)
    gt_lg = rgd.zeros(len(gt0_lg))
    gt_lg[:, :gcut] = gt0_lg / (4 * np.pi)

    for l, gt_g in enumerate(gt_lg):
        print(l, rgd.integrate(gt_g, l))

    lmax = max(l_j)
    G_LLL = gaunt(lmax)
    v_jjlg = {}
    vt_jjlg = {}
    nt_jjlg = {}
    for j1, dn_jg in enumerate(n_jjg - nt_jjg):
        for j2, dn_g in enumerate(dn_jg):
            l12 = l_j[j1] + l_j[j2]
            for l in range(l12 % 2, l12 + 1, 2):
                dN = rgd.integrate(dn_g, l)
                v_g = rgd.poisson(n_jjg[j1, j2], l)
                v_jjlg[j1, j2, l] = v_g
                nt_g = nt_jjg[j1, j2] + dN * gt_lg[l]
                nt_jjlg[j1, j2, l] = nt_g
                vt_g = rgd.poisson(nt_g, l)
                vt_jjlg[j1, j2, l] = vt_g
                dN2 = rgd.integrate(nt_g - n_jjg[j1, j2], l)
                print(j1, j2, l, dN, dN2)

    I = sum(2 * l + 1 for l in l_j)
    C_iiii = np.empty((I, I, I, I))
    for j1, i1, L1, j2, i2, L2 in indices2(l_j):
        for j3, i3, L3, j4, i4, L4 in indices2(l_j):
            C = 0.0
            for l in range(lmax * 2 + 1):
                LL = slice(l**2, (l + 1)**2)
                coef = G_LLL[L1, L2, LL] @ G_LLL[L3, L4, LL]
                if abs(coef) > 1e-14:
                    C += 2 * np.pi * coef * rgd.integrate(
                        (n_jjg[j1, j2] * v_jjlg[j3, j4, l] -
                         nt_jjlg[j1, j2, l] * vt_jjlg[j3, j4, l]), l - 1)
                    if i1 + i2 + i3 + i4 == 0:
                        print(2 * np.pi * rgd.integrate(
                            n_jjg[j1, j2] * v_jjlg[j3, j4, l], l - 1))
                        print(2 * np.pi * rgd.integrate(
                            nt_jjlg[j1, j2, l] * vt_jjlg[j3, j4, l], l - 1))
                        print(l, coef, C)
            C_iiii[i1, i2, i3, i4] = C
    return C_iiii


def indices(l_j):
    i = 0
    for j, l in enumerate(l_j):
        for L in range(l**2, (l + 1)**2):
            yield j, i, L
            i += 1


def indices2(l_j):
    for j1, i1, L1 in indices(l_j):
        for j2, i2, L2 in indices(l_j):
            yield j1, i1, L1, j2, i2, L2


s = create_setup('H', lmax=2)
C = coulomb(s.rgd,
            np.array(s.data.phi_jg),
            np.array(s.data.phit_jg),
            s.l_j,
            s.g_lg)
print(C[0,0,0])
print(s.M_pp[0])
print(s.dO_ii)



'''
    def calculate_coulomb_corrections(self, wn_lqg, wnt_lqg, wg_lg, wnc_g,
                                      wmct_g):
        """Calculate "Coulomb" energies."""
        # Can we reduce the excessive parameter passing?
        # Seems so ....
        # Added instance variables
        # T_Lqp = self.local_corr.T_Lqp
        # n_qg = self.local_corr.n_qg
        # Delta_lq = self.local_corr.Delta_lq
        # nt_qg = self.local_corr.nt_qg
        # Local variables derived from instance variables
        _np = self.ni * (self.ni + 1) // 2  # change to inst. att.?
        mct_g = self.local_corr.nct_g + self.Delta0 * self.g_lg[0]  # s.a.
        rdr_g = self.local_corr.rgd2.r_g * \
            self.local_corr.rgd2.dr_g  # change to inst. att.?

        A_q = 0.5 * (np.dot(wn_lqg[0], self.local_corr.nc_g) + np.dot(
            self.local_corr.n_qg, wnc_g))
        A_q -= sqrt(4 * pi) * self.Z * np.dot(self.local_corr.n_qg, rdr_g)
        A_q -= 0.5 * (np.dot(wnt_lqg[0], mct_g) +
                      np.dot(self.local_corr.nt_qg, wmct_g))
        A_q -= 0.5 * (np.dot(mct_g, wg_lg[0]) +
                      np.dot(self.g_lg[0], wmct_g)) * \
            self.local_corr.Delta_lq[0]
        M_p = np.dot(A_q, self.local_corr.T_Lqp[0])

        A_lqq = []
        for l in range(2 * self.local_corr.lcut + 1):
            A_qq = 0.5 * np.dot(self.local_corr.n_qg, np.transpose(wn_lqg[l]))
            A_qq -= 0.5 * np.dot(self.local_corr.nt_qg,
                                 np.transpose(wnt_lqg[l]))
            if l <= self.lmax:
                A_qq -= 0.5 * np.outer(self.local_corr.Delta_lq[l],
                                       np.dot(wnt_lqg[l], self.g_lg[l]))
                A_qq -= 0.5 * np.outer(np.dot(self.local_corr.nt_qg,
                                              wg_lg[l]),
                                       self.local_corr.Delta_lq[l])
                A_qq -= 0.5 * np.dot(self.g_lg[l], wg_lg[l]) * \
                    np.outer(self.local_corr.Delta_lq[l],
                             self.local_corr.Delta_lq[l])
            A_lqq.append(A_qq)

        M_pp = np.zeros((_np, _np))
        L = 0
        for l in range(2 * self.local_corr.lcut + 1):
            for m in range(2 * l + 1):  # m?
                M_pp += np.dot(np.transpose(self.local_corr.T_Lqp[L]),
                               np.dot(A_lqq[l], self.local_corr.T_Lqp[L]))
                L += 1

        return M_p, M_pp

    def calculate_integral_potentials(self, func):
        """Calculates a set of potentials using func."""
        wg_lg = [func(self, self.g_lg[l], l)
                 for l in range(self.lmax + 1)]
        wn_lqg = [np.array([func(self, self.local_corr.n_qg[q], l)
                            for q in range(self.local_corr.nq)])
                  for l in range(2 * self.local_corr.lcut + 1)]
        wnt_lqg = [np.array([func(self, self.local_corr.nt_qg[q], l)
                             for q in range(self.local_corr.nq)])
                   for l in range(2 * self.local_corr.lcut + 1)]
        wnc_g = func(self, self.local_corr.nc_g, l=0)
        wnct_g = func(self, self.local_corr.nct_g, l=0)
        wmct_g = wnct_g + self.Delta0 * wg_lg[0]
        return wg_lg, wn_lqg, wnt_lqg, wnc_g, wnct_g, wmct_g
'''
