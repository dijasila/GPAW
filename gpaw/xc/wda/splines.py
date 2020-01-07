import numpy as np

def build_splines(nb_i, gd):
    from scipy.interpolate import interp1d
    from scipy.integrate import quad
    from ks import get_K_K
    from gpaw.atom.radialgd import fsbt
    na = np.newaxis
    C_i, dC_i = get_C(nb_i)

    lambd_i, dlambd_i = get_lambd(nb_i)
    Gsplines_i = []
    Grsplines_i = []

    for i, nb in enumerate(nb_i):
        C, dC = get_C(nb)
        lambd, dlambd = get_lambd(nb)
        K_K = get_K_K(gd)
        kmax = 1.2 * np.max(K_K)

        dr = 0.01
        rmax = lambd * 500
        r_j = np.arange(dr, rmax, dr)

        nks = 900
        k_k = np.exp(np.linspace(0, np.log(kmax), nks)) - 1

        assert np.allclose(np.min(k_k), 0)

        G_j = C * (1 - np.exp(- (lambd / r_j)**5))
        integral_k = 2 * np.pi * fsbt(0, G_j, r_j, k_k)
        
        interf = lambda k, G: interp1d(k, G,
                                       kind='cubic', fill_value='extrapolate')
        Gspline = interf(k_k, integral_k)
        assert np.allclose(Gspline(k_k), integral_k)

        Gr_j = C * (1 - np.exp(- (lambd / r_j)**5)) / r_j

        integral_k = 2 * np.pi * fsbt(0, Gr_j, r_j, k_k)

        Grspline = interf(k_k, integral_k)

        Gsplines_i.append(Gspline)
        Grsplines_i.append(Grspline)

    return Gsplines_i, Grsplines_i    

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

