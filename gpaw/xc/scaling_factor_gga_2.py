
import numpy as np


class PurePythonSFG2Kernel:

    def __init__(self):
        self.name = 'SFG2'
        self.type = 'GGA'

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg, dedsigma_xg,
                  tau_sg=None, dedtau_sg=None,
                  n_stot=None, v_scom=None, spin=None):

        e_g[:] = 0.
        dedsigma_xg[:] = 0.
        nspins = n_stot.shape[0]

        if spin == 0:
            s = 0
        else:
            s = 2

        e, v, v_t, v_gr = scaling_factor(
            n_sg[spin], sigma_xg[s], n_stot[spin], nspins, 0.25)

        e_g += e
        dedsigma_xg[s] += v_gr
        dedn_sg[spin] += v
        v_scom[spin] += v_t


REGULARIZATION = 1.0e-16


# a2 = |grad n|^2
def scaling_factor(n, a2, n_tot, nspins, c0):

    spin_sc = 3.0 - nspins

    eps = REGULARIZATION
    n[n < eps] = 1.0e-40
    n_tot[n_tot < eps] = 1.0e-40 * spin_sc

    const1 = 4.0 * (3. * np.pi**2.)**(2./3.)
    const2 = c0 * 8.0/3.0

    # (2 * k_F * n) ** 2.0
    tkfn2 = const1 * n**(8.0/3.0)
    u = n / (n_tot / spin_sc)
    s2 = a2 / tkfn2

    u[u > 1.0] = 1.0
    u[u < 0.0] = 0.0

    pok = 0.2
    g1 = g1_sf(u, pok)
    dg1 = dg1_sf(u, pok)
    g2 = g2_sf(u, pok)
    dg2 = dg2_sf(u, pok)
    h = h_sf(s2, c0)
    f = 1. - (1. - g2) * h

    eps = n * g1 * f
    dedn = g1 * f + u * (dg2 * h * g1 + dg1 * f) - \
        g1 * (1. - g2) * h**2. * s2 * const2
    dedn_t = - u**2.0 * (dg1 * f + g1 * dg2 * h )
    deda2 = c0 * g1 * (1. - g2) * h**2 * n / tkfn2

    return eps, dedn,  dedn_t, deda2


def g1_sf(u, eps):
    return u**eps


def dg1_sf(u, eps):
    return eps * (u ** (eps - 1.))


def g2_sf(u, eps):
    return u**(1.0 - eps)


def dg2_sf(u, eps):
    return (1.0 - eps) * (u ** (- eps))


def h_sf(s2, a):
    return 1. / (1. + a * s2)
