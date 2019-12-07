import numpy as np



# Write function that returns
# [spline(G_i)], [spline(G_i/r)]
# [spline(dG_i)], [spline(dG_i/r)]


def build_splines(nb_i, gd):
    from scipy.interpolate import interp1d
    from gpaw.utilities.tools import construct_reciprocal
    K_K, _ = construct_reciprocal(gd)
    K_K = np.sqrt(K_K)
    K_K[0,0,0] = 0
    kmax = 1.2 * np.max(K_K)
    dr = 0.001
    rmax = 1
    r_j = np.arange(dr / 2, 10, dr)
    nks = 1000
    k_k = np.exp(np.linspace(0, np.log(kmax), nks)) - 1
    assert np.allclose(np.min(k_k), 0)
    na = np.newaxis
    r_ikj = r_j[na, na, :]
    k_ikj = k_k[na, :, na]
    C_i, dC_i = C(nb_i)
    C_ikj = C_i[:, na, na]
    dC_ikj = dC_i[:, na, na]
    lambd_i, dlambd_i = lambd(nb_i)
    lambd_ikj = lambd_i[:, na, na]
    dlambd_ikj = dlambd_i[:, na, na]
    G_ikj = C_ikj * (1 - np.exp(-(lambd_ikj / r_ikj)**5))

    integrand_ikj = 4 * np.pi * np.sinc(k_ikj * r_ikj) * r_ikj**2 \
                    * G_ikj
    integral_ik = np.sum(integrand_ikj * dr, axis=-1)
    # Gsplines_i = interp1d(k_k, integral_ik, kind='cubic') Can also do this
    interf = lambda k, G: interp1d(k, G,
                                   kind='cubic', fill_value='extrapolate')
    Gsplines_i = [interf(k_k, G_k) for G_k in integral_ik]

    

    integrand_ikj = 4 * np.pi * r_ikj \
                    * np.sinc(k_ikj * r_ikj) * G_ikj
    integral_ik = np.sum(integrand_ikj * dr, axis=-1)
    Grsplines_i = [interf(k_k, G_k) for G_k in integral_ik]

    dG_ikj = dC_ikj * (1 - np.exp(-(lambd_ikj / r_ikj)**5)) \
             + C_ikj * dlambd_ikj * (5 * lambd_ikj**4 / r_ikj**5) \
             * np.exp(-(lambd_ikj / r_ikj)**5)
    integrand_ikj = 4 * np.pi * r_ikj**2 \
                    * np.sinc(k_ikj * r_ikj) * dG_ikj
    integral_ik = np.sum(integrand_ikj * dr, axis=-1)
    dGsplines_i = [interf(k_k, dG_k) for dG_k in integral_ik]

    integrand_ikj = 4 * np.pi * r_ikj \
                    * np.sinc(k_ikj * r_ikj) * dG_ikj
    integral_ik = np.sum(integrand_ikj * dr, axis=-1)
    dGrsplines_i = [interf(k_k, dG_k) for dG_k in integral_ik]

    return Gsplines_i, Grsplines_i, dGsplines_i, dGrsplines_i
    

def lambd(n):
    from scipy.special import gamma
    exc = get_lda_xc(n, 0)
    dn = 0.000001
    dexc = (get_lda_xc(n + dn, 0) - exc) / dn
    
    lambd = - 3 * gamma(3 / 4) / (2 * gamma(2 / 5) * exc)

    dlambd = 3 * gamma(3 / 4) / (2 * gamma(2/5)) \
             * (1 / exc**2 * dexc)
    return lambd, dlambd


def C(n):
    from scipy.special import gamma
    lamb, dlambd = lambd(n)
    C = - 3 / (4 * np.pi * gamma(2 / 5) * n * lamb**3)

    dC = 3 / (4 * np.pi * gamma(2 / 5)) \
         * (1 / (n**2 * lamb**3) + 3 / (n * lamb**4) * dlambd)
    return C, dC

def get_lda_xc(n, spin):
    if np.allclose(n, 0):
        return 0

    from gpaw.xc.lda import lda_c, lda_x
    narr = n # np.array([n]).astype(np.float64)
    earr = np.zeros_like(narr) # (narr.shape[1:])
    varr = np.zeros_like(narr)

    lda_x(spin, earr, n, varr)
    zeta = 0
    lda_c(spin, earr, narr, varr, zeta)

    return earr[0] / n

