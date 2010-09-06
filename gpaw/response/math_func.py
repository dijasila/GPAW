from math import sqrt, pi

import numpy as np

from gpaw.utilities.blas import gemmdot
from gpaw.gaunt import gaunt as G_LLL
from gpaw.spherical_harmonics import Y


def delta_function(x0, dx, Nx, sigma):

    deltax = np.zeros(Nx)
    for i in range(Nx):
        deltax[i] = np.exp(-(i * dx - x0)**2/(4. * sigma))
    return deltax / (2. * sqrt(pi * sigma))


def hilbert_transform(specfunc_wGG, Nw, dw, eta):

    NwS = specfunc_wGG.shape[0]
    tmp_ww = np.zeros((Nw, NwS), dtype=complex)

    for iw in range(Nw):
        w = iw * dw
        for jw in range(NwS):
            ww = jw * dw 
            tmp_ww[iw, jw] = 1. / (w - ww + 1j*eta) - 1. / (w + ww + 1j*eta)

    chi0_wGG = gemmdot(tmp_ww, specfunc_wGG, beta = 0.)

    return chi0_wGG * dw

                
def two_phi_planewave_integrals(k_Gv, setup=None, rgd=None, phi_jg=None,
                                phit_jg=None,l_j=None):
    """Calculate PAW-correction matrix elements with planewaves.

    ::
    
      /  _       _   ik.r     _     ~   _   ik.r ~   _
      | dr [phi (r) e    phi (r) - phi (r) e    phi (r)]
      /        1            2         1            2

                        ll    -  /     2                      ~       ~
      = 4 * pi \sum_lm  i  Y (k) | dr r  [ phi (r) phi (r) - phi (r) phi (r) j (kr)
                            lm   /            1       2         1       2     ll

           /
        * | d\Omega Y     Y     Y
          /          l1m1  l2m2  lm
          
    """

    from scipy.special import sph_jn

    if setup is not None:
        ng = setup.ng
        g = np.arange(ng, dtype=float)
        r_g = setup.beta * g / (ng - g)
        dr_g = setup.beta * ng / (ng - g)**2
        phi_jg = setup.data.phi_jg # list object
        phit_jg = setup.data.phit_jg
        l_j = setup.l_j
    else:
        assert rgd is not None
        assert phi_jg is not None
        assert l_j is not None
        ng = rgd.ng
        r_g = rgd.r_g
        dr_g = rgd.dr_g

    
    # Construct L (l**2 + m) and j (nl) index
    L_i = []
    j_i = []
    for j, l in enumerate(l_j):
        for m in range(2 * l + 1):
            L_i.append(l**2 + m)
            j_i.append(j)
    ni = len(L_i)
    nj = len(l_j)
    lmax = max(l_j) * 2 + 1

    if setup is not None:
        assert ni == setup.ni and nj == setup.nj

    # Initialize
    npw = k_Gv.shape[0]
    R_jj = np.zeros((nj, nj))
    R_ii = np.zeros((ni, ni))
    phi_Gii = np.zeros((npw, ni, ni), dtype=complex)
    j_lg = np.zeros((lmax, ng))

    # Store (phi_j1 * phi_j2 - phit_j1 * phit_j2 ) for further use
    tmp_jjg = np.zeros((nj, nj, ng))
    for j1 in range(nj):
        for j2 in range(nj):
            tmp_jjg[j1, j2] = (phi_jg[j1] * phi_jg[j2] -
                               phit_jg[j1] * phit_jg[j2])

    # Loop over G vectors
    for iG in range(npw):
        kk = k_Gv[iG] 
        k = np.sqrt(np.dot(kk, kk)) # calculate length of q+G
        
        # Calculating spherical bessel function
        for ri in range(ng):
            j_lg[:,ri] = sph_jn(lmax - 1,  k*r_g[ri])[0]

        for li in range(lmax):
            # Radial part 
            for j1 in range(nj):
                for j2 in range(nj): 
                    R_jj[j1, j2] = np.dot(r_g**2*dr_g,
                                          tmp_jjg[j1, j2] * j_lg[li])

            for mi in range(2 * li + 1):
                # Angular part
                for i1 in range(ni):
                    L1 = L_i[i1]
                    j1 = j_i[i1]
                    for i2 in range(ni):
                        L2 = L_i[i2]
                        j2 = j_i[i2]
                        R_ii[i1, i2] = G_LLL[L1, L2, li**2+mi]  * R_jj[j1, j2]

                phi_Gii[iG] += R_ii * Y(li**2 + mi,
                                        kk[0]/k, kk[1]/k, kk[2]/k) * (-1j)**li

    
    phi_Gii *= 4 * pi

    return phi_Gii.reshape(npw, ni*ni)
