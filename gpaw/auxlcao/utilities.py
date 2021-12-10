import numpy as np
from numpy.linalg import eigh
from collections import defaultdict

r""" The rgd.poisson returns the radial Poisson solution multiplied with r.

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

def get_compensation_charge_splines_screened(setup, lmax, cutoff):
    rgd = setup.rgd
    wghat_l = []
    ghat_l = []
    W_LL = np.zeros( ( (lmax+1)**2, (lmax+1)**2 ) )
    L=0
    for l in range(lmax+1):
        spline = setup.ghat_l[l]
        ghat_l.append(spline)
        g_g = spline_to_rgd(rgd, spline)
        v_g = setup.screened_coulomb.screened_coulomb(g_g, l)
        integral = rgd.integrate(g_g*v_g)
        for m in range(2*l+1):
            W_LL[L,L] = integral / (np.pi*4)
            L += 1
        wghat_l.append(rgd.spline(v_g, cutoff, l, 500))
    return ghat_l, wghat_l, W_LL

def _get_auxiliary_splines(setup, lmax, cutoff, poisson):
    rgd = setup.rgd

    auxt_lng = defaultdict(lambda: [])
    wauxt_lng = defaultdict(lambda: [])
    for j1, spline1 in enumerate(setup.phit_j):
        l1 = spline1.get_angular_momentum_number()
        for j2, spline2 in enumerate(setup.phit_j):
            if j1 > j2:
                continue
            l2 = spline2.get_angular_momentum_number()
            for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
                if l > 2:
                    continue
                aux_g = spline_to_rgd(rgd, spline1, spline2)
                auxt_lng[l].append(aux_g) #rgd.spline(aux_g, cutoff, l, 500))

                v_g = poisson(aux_g, l)
                wauxt_lng[l].append(v_g) #  #rgd.spline(v_g, cutoff, l, 500))

    # Splines
    auxt_j = []
    wauxt_j = []
    sauxt_j = []
    wsauxt_j = []
    M_j = []

    for l, auxt_ng in auxt_lng.items():
        auxt_ng = np.array(auxt_ng)
        wauxt_ng = np.array(wauxt_lng[l])
        N = len(auxt_ng)
        S_nn = np.zeros( (N, N) )
        #import matplotlib.pyplot as plt
        for n1, auxt_g in enumerate(auxt_ng):
            #plt.plot(rgd.r_g, auxt_g)
            #plt.plot(rgd.r_g, wauxt_ng[n1],'x')
            for n2, wauxt_g in enumerate(wauxt_ng):
                S_nn[n1, n2] = rgd.integrate(auxt_g * wauxt_g)
        S_nn = (S_nn + S_nn.T) / 2

        print('l=%d' % l, S_nn)
        eps_i, v_ni = eigh(S_nn)
        assert np.all(eps_i>0)
        nbasis = int((eps_i > 1e-2).sum())
        q_ni = np.dot(v_ni[:, -nbasis:],
                      np.diag(eps_i[-nbasis:]**-0.5))

        #plt.show()

        auxt_ig =  q_ni.T @ auxt_ng
        wauxt_ig = q_ni.T @ wauxt_ng

        # Evaluate reference multipole momement
        g_g = spline_to_rgd(rgd, setup.ghat_l[l])
        Mref = rgd.integrate(g_g * rgd.r_g**l) / (4*np.pi)

        for i in range(len(auxt_ig)):
            aux_g = auxt_ig[i]
            auxt_j.append(rgd.spline(aux_g, cutoff, l, 500))
            wauxt_j.append(rgd.spline(wauxt_ig[i], cutoff, l, 500))
            
            # Evaluate multipole moment
            M = rgd.integrate(auxt_g * rgd.r_g**l) / (4*np.pi)
            M_j.append(M)

            sauxt_g = auxt_g - M / Mref * g_g
            sauxt_j.append(rgd.spline(sauxt_g, cutoff, l, 500))

            v_g = poisson(sauxt_g, l)
            wsauxt_j.append(rgd.spline(v_g, cutoff, l, 500))
            print('Last potential element', v_g[-1])
            assert(np.abs(v_g[-1])<1e-6)


    return auxt_j, wauxt_j, sauxt_j, wsauxt_j, M_j

def get_auxiliary_splines(setup, lmax, cutoff):
    def poisson(n_g,l):
        return Hartree(setup.rgd, n_g, l)

    return _get_auxiliary_splines(setup, lmax, cutoff, poisson)

def get_auxiliary_splines_screened(setup, lmax, cutoff):
    return _get_auxiliary_splines(setup, lmax, cutoff, setup.screened_coulomb.screened_coulomb)


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
    eigs = np.linalg.eigvalsh(W_AA)
    #if np.any(eigs < 1e-8):
    #    print('Warning. Nearly singular matrix.')
    for x in eigs:
        print(x,end=' ')
    iW_AA = np.linalg.pinv(W_AA, hermitian=True, rcond=1e-10)
    #iW_AA = np.linalg.inv(W_AA)
    iW_AA = (iW_AA + iW_AA.T)/2
    return iW_AA

