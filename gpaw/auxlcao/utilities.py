import numpy as np

""" The rgd.poisson returns the radial Poisson solution multiplied with r.

               /        n(r')
    v(r) r = r | dr' ----------
               /     | r - r' |


   This function divides the r above, thus returning the actual poisson solution.
   To avoid division by zero, the v_g[0] is approximated just by copying v_g[1] grid point.
"""

def Hartree(rgd, n_g, l):
    v_g = rgd.poisson(n_g, l)
    v_g[1:] /= rgd.r_g[1:]
    v_g[0] = v_g[1]
    return v_g

def spline_to_rgd(rgd, spline, spline2=None):
    f_g = rgd.zeros()
    for g, r in enumerate(rgd.r_g):
        f_g[g] = spline(r) * r**spline.l
    if spline2 is not None:
        return f_g * spline_to_rgd(rgd, spline2)
    return f_g

def get_compensation_charge_splines(setup, lmax, cutoff):
    rgd = setup.rgd
    wghat_l = []
    ghat_l = []
    for l in range(lmax+1):
        spline = setup.ghat_l[l]
        ghat_l.append(spline)
        g_g = spline_to_rgd(rgd, spline)
        v_g = Hartree(rgd, g_g, l)
        wghat_l.append(rgd.spline(v_g, cutoff, l, 500))
    return ghat_l, wghat_l

def get_auxiliary_splines(setup, lmax, cutoff):
    rgd = setup.rgd
    auxt_j = []
    wauxt_j = []
    sauxt_j = []
    wsauxt_j = []
    M_j = []

    for j1, spline1 in enumerate(setup.phit_j):
        for j2, spline2 in enumerate(setup.phit_j):
            if j1 > j2:
                continue
            l1 = spline1.get_angular_momentum_number()
            l2 = spline2.get_angular_momentum_number()
            for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
                if l > 2:
                    continue
                aux_g = spline_to_rgd(rgd, spline1, spline2)
                auxt_j.append(rgd.spline(aux_g, cutoff, l, 500))

                v_g = Hartree(rgd, aux_g, l)
                wauxt_j.append(rgd.spline(v_g, cutoff, l, 500))

                # Evaluate multipole moment
                M = rgd.integrate(aux_g * rgd.r_g**l) / (4*np.pi)
                M_j.append(M)

                g_g = spline_to_rgd(rgd, setup.ghat_l[l])
                Mref = rgd.integrate(g_g * rgd.r_g**l) / (4*np.pi)
                saux_g = aux_g - M / Mref * g_g
                sauxt_j.append(rgd.spline(saux_g, cutoff, l, 500))

                v_g = Hartree(rgd, saux_g, l)
                wsauxt_j.append(rgd.spline(v_g, cutoff, l, 500))
                assert(np.abs(v_g[-1])<1e-6)


    return auxt_j, wauxt_j, sauxt_j, wsauxt_j, M_j


def get_wgauxphit_product_splines(setup, wgaux_j, phit_j, cutoff):
    rgd = setup.rgd
    x = 0
    wgauxphit_x = []
    for wgaux in wgaux_j:
        lg = wgaux.l
        for j1, spline1 in enumerate(phit_j):
            l1 = spline1.l
            for l in range((l1 + lg) % 2, l1 + lg + 1, 2):
                wgauxphit_g = spline_to_rgd(rgd, wgaux, spline1)
                wgauxphit_x.append(rgd.spline(wgauxphit_g, cutoff, l))
    return wgauxphit_x


def safe_inv(W_AA):
    #eigs = np.linalg.eigvalsh(W_AA)
    #if np.any(eigs < 1e-8):
    #    print('Warning. Nearly singular matrix.')
    #    for x in eigs:
    #        print(x,end=' ')
    iW_AA = np.linalg.pinv(W_AA, hermitian=True, rcond=1e-6)
    #iW_AA = np.linalg.inv(W_AA)
    iW_AA = (iW_AA + iW_AA.T)/2
    return iW_AA

