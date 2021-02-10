import numpy as np
from gpaw.setup import create_setup
from gpaw.gaunt import gaunt
from gpaw.utilities import packed_index


def coulomb(rgd, phi_jg, phit_jg, l_j, gt0_lg):
    gcut = gt0_lg.shape[1]
    phi_jg[:, gcut:] = 0.0
    phit_jg[:, gcut:] = 0.0
    n_jjg = np.einsum('jg, kg -> jkg', phi_jg, phi_jg) / (4 * np.pi)
    nt_jjg = np.einsum('jg, kg -> jkg', phit_jg, phit_jg) / (4 * np.pi)
    gt_lg = rgd.zeros(len(gt0_lg))
    gt_lg[:, :gcut] = gt0_lg / (4 * np.pi)

    for l, gt_g in enumerate(gt_lg):
        assert abs(rgd.integrate(gt_g, l) - 1.0) < 1e-10

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
                dN = rgd.integrate(nt_g - n_jjg[j1, j2], l)
                assert abs(dN) < 1e-14

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
print(C[0, 0, 0])
print(s.M_pp[0])
ni = len(C)
for i1 in range(ni):
    for i2 in range(ni):
        p12 = packed_index(i1, i2, ni)
        for i3 in range(ni):
            for i4 in range(ni):
                p34 = packed_index(i3, i4, ni)
                print(i1, i2, i3, i4, p12, p34,
                      s.M_pp[p12, p34] - C[i1, i2, i3, i4])
