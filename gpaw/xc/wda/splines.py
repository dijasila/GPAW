import numpy as np



# Write function that returns
# [spline(G_i)], [spline(G_i/r)]
# [spline(dG_i)], [spline(dG_i/r)]


def build_splines(nb_i, gd):
    from scipy.interpolate import interp1d
    from scipy.integrate import quad
    from ks import get_K_K
    from gpaw.atom.radialgd import fsbt
    na = np.newaxis
    C_i, dC_i = get_C(nb_i)
    # C_ikj = C_i[:, na, na]
    # dC_ikj = dC_i[:, na, na]

    lambd_i, dlambd_i = get_lambd(nb_i)
    # lambd_ikj = lambd_i[:, na, na]
    # dlambd_ikj = dlambd_i[:, na, na]
    Gsplines_i = []
    Grsplines_i = []

    for i, nb in enumerate(nb_i):
        C, dC = get_C(nb)
        lambd, dlambd = get_lambd(nb)
        K_K = get_K_K(gd)
        kmax = 1.2 * np.max(K_K)

        dr = 0.01
        rmax = lambd * 300
        r_j = np.arange(dr, rmax, dr)

        nks = 900
        k_k = np.exp(np.linspace(0, np.log(kmax), nks)) - 1
        # k_k = np.linspace(0, kmax, nks)

        assert np.allclose(np.min(k_k), 0)

        r_kj = r_j[na, :]
        k_kj = k_k[:, na]

        # def Gintegrand(r, k):
        #     return C * (1 - np.exp(- (lambd / r)**5)) * 4 * np.pi * np.sinc(k * r) * r**2

        # integral_k = np.zeros_like(k_k)
        # for ik, k in enumerate(k_k):
        #     inter = lambda r : Gintegrand(r, k)
        #     integral_k[ik] = quad(inter, 0, np.inf, limit=10)[0]
        
        # G_kj = C * (1 - np.exp(-(lambd / r_kj)**5))


        G_j = C * (1 - np.exp(- (lambd / r_j)**5))
        integral_k = 2 * np.pi * fsbt(0, G_j, r_j, k_k)
        
        # integrand_kj = (4 * np.pi * np.sinc(k_kj * r_kj) * r_kj**2 
        # * G_kj)
        # integral_k = np.sum(integrand_kj * dr, axis=-1)
        # integral_k = np.trapz(integrand_kj, r_kj)
        # Gsplines_i = interp1d(k_k, integral_ik, kind='cubic') Can also do this
        # print(f"integral_k[0] * nb: {integral_k[0] * nb}")
        interf = lambda k, G: interp1d(k, G,
                                       kind='linear')#, fill_value='extrapolate')
        Gspline = interf(k_k, integral_k)
        assert np.allclose(Gspline(k_k), integral_k)
        # integrand_kj = 4 * np.pi * r_kj \
        #                 * np.sinc(k_kj * r_kj) * G_kj
        # integral_k = np.sum(integrand_kj * dr, axis=-1)

        # def Grintegrand(r, k):
        #     return C * (1 - np.exp(- (lambd / r)**5)) * 4 * np.pi * np.sinc(k * r) * r

        # integral_k = np.zeros_like(k_k)
        # for ik, k in enumerate(k_k):
        #     inter = lambda r : Grintegrand(r, k)
        #     integral_k[ik] = quad(inter, 0, np.inf, limit=10)[0]

        Gr_j = C * (1 - np.exp(- (lambd / r_j)**5)) / r_j

        integral_k = 2 * np.pi * fsbt(0, Gr_j, r_j, k_k)

        Grspline = interf(k_k, integral_k)

        Gsplines_i.append(Gspline)
        Grsplines_i.append(Grspline)

    return Gsplines_i, Grsplines_i, None, None

    # dG_ikj = dC_ikj * (1 - np.exp(-(lambd_ikj / r_ikj)**5)) \
    #          + C_ikj * dlambd_ikj * (5 * lambd_ikj**4 / r_ikj**5) \
    #          * np.exp(-(lambd_ikj / r_ikj)**5)
    # integrand_ikj = 4 * np.pi * r_ikj**2 \
    #                 * np.sinc(k_ikj * r_ikj) * dG_ikj
    # integral_ik = np.sum(integrand_ikj * dr, axis=-1)
    # dGsplines_i = [interf(k_k, dG_k) for dG_k in integral_ik]

    # integrand_ikj = 4 * np.pi * r_ikj \
    #                 * np.sinc(k_ikj * r_ikj) * dG_ikj
    # integral_ik = np.sum(integrand_ikj * dr, axis=-1)
    # dGrsplines_i = [interf(k_k, dG_k) for dG_k in integral_ik]

    # return Gsplines_i, Grsplines_i, dGsplines_i, dGrsplines_i
    

def get_lambd(n):
    from scipy.special import gamma
    exc = get_lda_xc(n, 0)
    dn = 0.00000001
    dexc = (get_lda_xc(n + dn, 0) - exc) / dn
    
    lamb = - 3 * gamma(3 / 4) / (2 * gamma(2 / 5) * exc)

    dlambd = 3 * gamma(3 / 4) / (2 * gamma(2/5)) \
             * (1 / exc**2 * dexc)
    return lamb, dlambd


def get_C(n):
    from scipy.special import gamma
    lamb, dlambd = get_lambd(n)
    C = - 3 / (4 * np.pi * gamma(2 / 5) * n * lamb**3)

    dC = 3 / (4 * np.pi * gamma(2 / 5)) \
         * (1 / (n**2 * lamb**3) + 3 / (n * lamb**4) * dlambd)
    return C, dC

def get_lda_xc(n, spin):
    if np.allclose(n, 0):
        raise ValueError
        return 0

    from gpaw.xc.lda import lda_c, lda_x
    narr = np.array([n]) # np.array([n]).astype(np.float64)
    earr = np.zeros_like(narr) # (narr.shape[1:])
    varr = np.zeros_like(narr)

    lda_x(spin, earr, n, varr)
    zeta = 0
    lda_c(spin, earr, narr, varr, zeta)

    return earr / n

