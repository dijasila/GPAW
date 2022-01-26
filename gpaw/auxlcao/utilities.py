import numpy as np
from numpy.linalg import eigh
from collections import defaultdict
from gpaw.atom.shapefunc import shape_functions
import scipy

from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import LocalizedFunctionsCollection as LFC

from gpaw.gaunt import gaunt

G_LLL = gaunt(3) # XXX


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

    W_LL = np.zeros( ( (lmax+1)**2, (lmax+1)**2 ) )
    L=0

    for l in range(lmax+1):
        #spline = ghat_l[l]
        spline = setup.ghat_l[l]
        g_g = spline_to_rgd(rgd, spline) / (4*np.pi)**0.5
        spline = rgd.spline(g_g, cutoff, l, 2000)
        ghat_l.append(spline)
        v_g = Hartree(rgd, g_g, l)
        integral = rgd.integrate(g_g*v_g)
        for m in range(2*l+1):
            W_LL[L,L] = integral / (np.pi*4)
            L += 1
        wghat_l.append(rgd.spline(v_g, cutoff, l, 2000))
    return ghat_l, wghat_l, W_LL

def get_compensation_charge_splines_screened(setup, lmax, cutoff):
    rgd = setup.rgd # local_corr.rgd2
    wghat_l = []
    ghat_l = []
    W_LL = np.zeros( ( (lmax+1)**2, (lmax+1)**2 ) )
    L=0
    for l in range(lmax+1):
        g_g = spline_to_rgd(rgd, setup.ghat_l[l]) / (4*np.pi)**0.5
        spline = rgd.spline(g_g, cutoff, l, 2000)
        ghat_l.append(spline)

        v_g = setup.screened_coulomb.screened_coulomb(g_g, l)
        integral = rgd.integrate(g_g*v_g)
        for m in range(2*l+1):
            W_LL[L,L] = integral / (np.pi*4)
            L += 1
        wghat_l.append(rgd.spline(v_g, cutoff, l, 2000))
    return ghat_l, wghat_l, W_LL

def _get_auxiliary_splines(setup, lcomp, laux, cutoff, poisson, threshold=1e-2):
    print('   Auxiliary basis setup for ',setup.symbol)
    rgd = setup.rgd
    print('Threshold: %.10f' % threshold)
    print('Auxiliary splines cutoff', cutoff,'bohr')
    auxt_lng = defaultdict(lambda: [])
    wauxt_lng = defaultdict(lambda: [])

    def add(aux_g, l):
        auxt_lng[l].append(aux_g)
        v_g = poisson(aux_g, l)
        wauxt_lng[l].append(v_g)

    for j1, spline1 in enumerate(setup.phit_j):
        l1 = spline1.get_angular_momentum_number()
        for j2, spline2 in enumerate(setup.phit_j):
            if j1 > j2:
                continue
            l2 = spline2.get_angular_momentum_number()
            #for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
            for l in range(l1+l2+1):
                if l > laux:
                    continue
                #print(G_LLL[l1**2:(l1+1)**2,l2**2:(l2+1)**2, l**2:(l+1)**2])
                C = np.max(np.abs(G_LLL[l1**2:(l1+1)**2,l2**2:(l2+1)**2, l**2:(l+1)**2]))
                #print(np.max(G_LLL[l1**2:(l1+1)**2,l2**2:(l2+1)**2, l**2:(l+1)**2]))
                aux_g = spline_to_rgd(rgd, spline1, spline2) * C
                #print('abs norm', l1, l2, l, rgd.integrate(np.abs(aux_g)))
                #print(C)
                if C:
                    #print('adding',C,l1,l2,l)
                    add(aux_g, l)
                    
                #aux_g = spline_to_rgd(rgd, spline1, spline2) * np.max(G_LLL[l1**2,l2**2, l**2])
                #assert rgd.integrate(np.abs(aux_g))>1e-10

    if 1:
        for l in range(3):
            add(np.exp(-2*rgd.r_g**2), l)
            if l>=2:
                continue
            add(np.exp(-1.3*rgd.r_g**2), l) 
            if l==1:
                continue
            add(np.exp(-0.8*rgd.r_g**2), l) 
            add(np.exp(-0.6*rgd.r_g**2), l) 
            add(np.exp(-0.3*rgd.r_g**2), l) 
    else:
        print('Not adding extra aux')

    #if setup.Z == 1:
    #    add(np.exp(-1.2621205398*rgd.r_g**2),1)
    #    add(np.exp(-0.50199775874*rgd.r_g**2),1)
    #    add(np.exp(-0.71290724024*rgd.r_g**2),2)
    #    # add(np.exp(-1.6565726132*rgd.r_g**2),3)

    # Splines
    auxt_j = []
    wauxt_j = []
    sauxt_j = []
    wsauxt_j = []
    M_j = []

    ghat_l = [] 

    Atot = 0
    integrals_lAA = []

    C = [ 1.0/ (4*np.pi)**0.5 ]

    import matplotlib.pyplot as plt

    #for ghat, C in zip(setup.ghat_l, C):
    #    g_g = spline_to_rgd(rgd, ghat) * C 
    #    ghat_l.append(rgd.spline(g_g, cutoff, l, 2000))
    ghat_l = setup.gaux_l

    for l, auxt_ng in auxt_lng.items():
        auxt_ng = np.array(auxt_ng)
        wauxt_ng = np.array(wauxt_lng[l])

        N = len(auxt_ng)
        S_nn = np.zeros( (N, N) )
        for n1, auxt_g in enumerate(auxt_ng):
            #plt.plot(rgd.r_g, auxt_g)
            #plt.plot(rgd.r_g, wauxt_ng[n1],'x')
            #plt.show()
            for n2, wauxt_g in enumerate(wauxt_ng):
                S_nn[n1, n2] = rgd.integrate(auxt_g * wauxt_g)
        S_nn = (S_nn + S_nn.T) / 2
        #print('l=%d' % l, S_nn)
        eps_i, v_ni = eigh(S_nn)
        #print(eps_i)
        
        assert np.all(eps_i>-1e-10)
        nbasis = int((eps_i > threshold).sum())
        #q_ni = np.dot(v_ni[:, -nbasis:],
        #              np.diag(eps_i[-nbasis:]**-0.5))
        q_ni = v_ni[:, -nbasis:]

        #plt.show()

        if 0:
            #print('Skipping transformation with ', q_ni.T)
            auxt_ig =  auxt_ng.copy()
            wauxt_ig = wauxt_ng.copy()
            #auxt_ig[0] += auxt_ig[1]
            #wauxt_ig[0] += wauxt_ig[1]
        else:
            #print('Transform', q_ni)
            auxt_ig =  q_ni.T @ auxt_ng
            wauxt_ig = q_ni.T @ wauxt_ng
            #wauxtnew_ig = np.zeros_like(auxt_ig)
            #for i, auxt_g in enumerate(auxt_ig):
            #    v_g = poisson(auxt_g, l)
            #    wauxtnew_ig[i] = v_g
            #    #print(wauxt_ig[i, ::15],'vs',wauxtnew_ig[i,::15])
        if l <= lcomp:
            g_g = spline_to_rgd(rgd, ghat_l[l])
        else:
            g_g[:] = 0.0
        # Evaluate reference multipole momement
        if 0:
            g2_g = spline_to_rgd(rgd, rgd.spline(g_g, cutoff, l, 2000))
            print(g2_g)
            print(g_g)
        #print('Compensation charge integral', rgd.integrate(g_g) / (4*np.pi)**0.5) 

        Mref = rgd.integrate(G_LLL[0,l**2,l**2] * g_g * rgd.r_g**l * (4*np.pi)**0.5) / (4*np.pi)**0.5
        #print('Comp Multipole', Mref)
        L = 2*l+1
        I_AA = np.zeros( (L*len(auxt_ig), L*len(auxt_ig)))

        for n1, auxt_g in enumerate(auxt_ig):
            for n2, wauxt_g in enumerate(wauxt_ig):
                #print(n1, n2, auxt_g, wauxt_g, rgd.integrate(auxt_g*wauxt_g / 4*np.pi))
                #print(n1, n2, auxt_g, wauxt_g, rgd.integrate(auxt_g*wauxt_ig[n2] / 4*np.pi),'pass2')
                #plt.plot(rgd.r_g, auxt_g)
                #plt.plot(rgd.r_g, wauxt_g)
                I_AA[n1*L : (n1+1)*L, n2*L : (n2+1)*L ] = np.eye(L) * rgd.integrate(auxt_g*wauxt_g) / (4*np.pi)
        integrals_lAA.append(I_AA)

        #print(I_AA,'Integrals')
        for i in range(len(auxt_ig)):
            Atot += 2*l+1
            auxt_g = auxt_ig[i]
            auxt_j.append(rgd.spline(auxt_g, cutoff, l, 2000))
            wauxt_j.append(rgd.spline(wauxt_ig[i], cutoff, l, 2000))
            
            # Evaluate multipole moment
            if l <= lcomp: # XXX Not screening at all2:
                M = rgd.integrate(auxt_g * rgd.r_g**l) / (4*np.pi)**0.5
                M_j.append(M)
                #print('Compensation M', M)
                sauxt_g = auxt_g - M / Mref * g_g
                assert np.abs(rgd.integrate(sauxt_g * rgd.r_g**l))<1e-10
            else:
                M_j.append(0.0)
                sauxt_g = auxt_g

            sauxt_j.append(rgd.spline(sauxt_g, cutoff, l, 2000))
            v_g = poisson(sauxt_g, l)
            #print('sauxt_g * wsauxt radial integral', rgd.integrate(sauxt_g * v_g / (4*np.pi)))
            #print('sauxt_g * ghat radial integral', rgd.integrate(g_g * v_g / (4*np.pi)))
            #print('auxt_g before', auxt_g, wauxt_ig[i], rgd.integrate(auxt_g * wauxt_ig[i] / (4*np.pi)))
            #print('auxt_g * wauxt radial integral', rgd.integrate(auxt_g * wauxt_ig[i] / (4*np.pi)))
            #plt.plot(rgd.r_g, auxt_g+0.5,'--')
            #print('auxt_g after', auxt_g)
            #plt.plot(rgd.r_g, wauxt_ig[i]+0.5,'--')
            #plt.show()

            #print('auxt_g * wsauxt radial integral', rgd.integrate(auxt_g * v_g / (4*np.pi)))
            #print('sauxt_g * wauxt radial integral', rgd.integrate(sauxt_g * wauxt_ig[i] / (4*np.pi)))
            #print('sauxt_g * wsauxt radial integral', rgd.integrate(sauxt_g * v_g / (4*np.pi)))
            wsauxt_j.append(rgd.spline(v_g, cutoff, l, 2000))
            #print('Last potential element', v_g[-1])
            #assert(np.abs(v_g[-1])<1e-6)
        print('l=%d %d (radial) auxiliary functions to %d (radial), %d (full) functions' % (l, len(auxt_ng), len(auxt_ig), (2*l+1)*len(auxt_ig)))

    W_AA = scipy.linalg.block_diag(*integrals_lAA)

    #print(W_AA,'FULL W_AA')

    print()
    return auxt_j, wauxt_j, sauxt_j, wsauxt_j, M_j, W_AA
    
    n = 200
    a = 12.0
    gd = GridDescriptor((n, n, n), (a, a, a))
    wauxtlfc = LFC(gd, [wsauxt_j])
    auxtlfc = LFC(gd, [sauxt_j])
    ghatlfc = LFC(gd, [ghat_l])
    auxtlfc.set_positions([(0.5, 0.5, 0.5)])
    wauxtlfc.set_positions([(0.5, 0.5, 0.5)])
    ghatlfc.set_positions([(0.5, 0.5, 0.5)])
    comp = gd.zeros()
    rho = gd.zeros()
    V = gd.zeros()
    auxtlfc.add(rho)
    wauxtlfc.add(V)
    ghatlfc.add(comp)
    print(V[:,40,40])
    x = gd.integrate(rho)
    print('Real space 1s norm',x)
    print('Real space comp norm',gd.integrate(comp))
    print('Real space S_AA',gd.integrate(rho*V))
    print('Real space M_AL',gd.integrate(comp*V))
    print('Cutoff',cutoff)
    

def get_auxiliary_splines(setup, lcomp, laux, cutoff, threshold=1e-2):
    def poisson(n_g,l):
        return Hartree(setup.rgd, n_g, l)

    return _get_auxiliary_splines(setup, lcomp, laux, cutoff, poisson, threshold=threshold)

def get_auxiliary_splines_screened(setup, lcomp, laux, cutoff, threshold=1e-2):
    return _get_auxiliary_splines(setup, lcomp, laux, cutoff, setup.screened_coulomb.screened_coulomb, threshold=threshold)


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
                wgauxphit_x.append(rgd.spline(wgauxphit_g, cutoff, l, 2000))
    return wgauxphit_x

def safe_inv(W_AA):
    eigs = np.linalg.eigvalsh(W_AA)
    if np.any(np.abs(eigs) < 1e-9):
        print('Safeinv eigs')
        for x in eigs:
            print(x,end=' ')
        print('Warning. Nearly singular matrix.')
        print(W_AA)
   
    if 0:     
        iW_AA = np.linalg.pinv((W_AA+W_AA.T)/2, hermitian=True, rcond=1e-10)
        iW_AA = (iW_AA + iW_AA.T)/2
    else:
        return np.linalg.inv(W_AA)

