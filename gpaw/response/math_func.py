from gpaw.pw.lfc import ft
import numpy as np

try:
    from scipy.special import spherical_jn

    def sphj(n, z):
        return spherical_jn(range(n), z)

except ImportError:
    from scipy.special import sph_jn

    def sphj(n, z):
        return sph_jn(n - 1, z)[0]


from gpaw.gaunt import gaunt
from gpaw.spherical_harmonics import Y


def two_phi_planewave_integrals(k_Gv, setup=None, Gstart=0, Gend=None,
                                rgd=None, phi_jg=None,
                                phit_jg=None, l_j=None):

    if Gend is None:
        Gend = len(k_Gv)

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
            
            for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
                spline = rgd.spline(rhot_g, l=l, points=2**10)
                splineG = ft(spline, N=2**12)
                f_G = splineG.map(k_G)

                for m1 in range(2 * l1 + 1):
                    for m2 in range(2 * l2 + 1):
                        i1 = i1_start + m1
                        i2 = i2_start + m2
                        G_m = G_LLL[l1**2 + m1, l2**2 + m2, l**2:(l + 1)**2]
                        for m, G in enumerate(G_m):
                            if G == 0:
                                continue
                            x_G = Y(l**2 + m, *k_Gv.T) * f_G * (-1j)**l
                            phi_Gii[:, i1, i2] += G * x_G

            i2_start += 2 * l2 + 1
        i1_start += 2 * l1 + 1
    return phi_Gii.reshape(npw, ni * ni)
