from math import pi

import numpy as np

from gpaw.xc.scaling_factor import calculate_paw_correction
from gpaw.utilities.blas import axpy
from gpaw.fd_operators import Gradient
from gpaw.sphere.lebedev import Y_nL, weight_n
from gpaw.xc.pawcorrection import rnablaY_nLv
from gpaw.xc.functional import XCFunctional


class SFGRadialExpansion:
    def __init__(self, rcalc, *args):
        self.rcalc = rcalc
        self.args = args

    def __call__(self, rgd, D_sLq, n_qg, nc0_sg, D_sLq_total, spin):
        n_sLg = np.dot(D_sLq, n_qg)
        n_sLg_total = np.dot(D_sLq_total, n_qg)

        n_sLg_total[spin, 0] += nc0_sg[spin]

        dndr_sLg = np.empty_like(n_sLg)
        for n_Lg, dndr_Lg in zip(n_sLg, dndr_sLg):
            for n_g, dndr_g in zip(n_Lg, dndr_Lg):
                rgd.derivative(n_g, dndr_g)

        nspins, Lmax, nq = D_sLq.shape
        dEdD_sqL = np.zeros((nspins, nq, Lmax))
        dEdD_sqL_com = np.zeros((nspins, nq, Lmax))

        E = 0.0
        for n, Y_L in enumerate(Y_nL[:, :Lmax]):
            w = weight_n[n]
            rnablaY_Lv = rnablaY_nLv[n, :Lmax]
            e_g, dedn_sg, b_vsg, dedsigma_xg, v_scom = \
                self.rcalc(rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv, n,
                           n_sLg_total, spin,
                           *self.args)

            dEdD_sqL += np.dot(rgd.dv_g * dedn_sg,
                               n_qg.T)[:, :, np.newaxis] * (w * Y_L)

            dEdD_sqL_com += np.dot(rgd.dv_g * v_scom,
                                   n_qg.T)[:, :, np.newaxis] * (w * Y_L)

            dedsigma_xg *= rgd.dr_g
            B_vsg = dedsigma_xg[::2] * b_vsg
            if nspins == 2:
                B_vsg += 0.5 * dedsigma_xg[1] * b_vsg[:, ::-1]
            B_vsq = np.dot(B_vsg, n_qg.T)
            dEdD_sqL += 8 * pi * w * np.inner(rnablaY_Lv, B_vsq.T).T
            E += w * rgd.integrate(e_g)

        return E, dEdD_sqL, dEdD_sqL_com


# First part of gga_calculate_radial - initializes some quantities.
def radial_gga_vars(rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv, n_sLg_tot):

    nspins = len(n_sLg)

    n_sg = np.dot(Y_L, n_sLg)
    n_sg_tot = np.dot(Y_L, n_sLg_tot)


    a_sg = np.dot(Y_L, dndr_sLg)
    b_vsg = np.dot(rnablaY_Lv.T, n_sLg)

    sigma_xg = rgd.empty(2 * nspins - 1)
    sigma_xg[::2] = (b_vsg ** 2).sum(0)
    if nspins == 2:
        sigma_xg[1] = (b_vsg[:, 0] * b_vsg[:, 1]).sum(0)
    sigma_xg[:, 1:] /= rgd.r_g[1:] ** 2
    sigma_xg[:, 0] = sigma_xg[:, 1]
    sigma_xg[::2] += a_sg ** 2
    if nspins == 2:
        sigma_xg[1] += a_sg[0] * a_sg[1]

    e_g = rgd.empty()
    dedn_sg = rgd.zeros(nspins)
    dedsigma_xg = rgd.zeros(2 * nspins - 1)

    v_comm = rgd.zeros(nspins)

    return e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, a_sg, b_vsg, \
           n_sg_tot, v_comm


def add_radial_gradient_correction(rgd, sigma_xg, dedsigma_xg, a_sg):
    nspins = len(a_sg)
    vv_sg = sigma_xg[:nspins]  # reuse array
    for s in range(nspins):
        rgd.derivative2(-2 * rgd.dv_g * dedsigma_xg[2 * s] * a_sg[s],
                        vv_sg[s])
    if nspins == 2:
        v_g = sigma_xg[2]
        rgd.derivative2(rgd.dv_g * dedsigma_xg[1] * a_sg[1], v_g)
        vv_sg[0] -= v_g
        rgd.derivative2(rgd.dv_g * dedsigma_xg[1] * a_sg[0], v_g)
        vv_sg[1] -= v_g

    vv_sg[:, 1:] /= rgd.dv_g[1:]
    vv_sg[:, 0] = vv_sg[:, 1]
    return vv_sg


class SFGRadialCalculator:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv, n,
                 n_sLg_tot, spin):
        (e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, a_sg,
         b_vsg, n_stot, v_scom) = \
            radial_gga_vars(rgd, n_sLg, Y_L, dndr_sLg,
                            rnablaY_Lv, n_sLg_tot)

        self.kernel.calculate(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg,
                              n_stot=n_stot,
                              v_scom=v_scom,
                              spin=spin)

        vv_sg = add_radial_gradient_correction(rgd, sigma_xg,
                                               dedsigma_xg, a_sg)

        return e_g, dedn_sg + vv_sg, b_vsg, dedsigma_xg, v_scom


def calculate_sigma(gd, grad_v, n_sg):
    """Calculate sigma(r) and grad n(r).
                  _     __   _  2     __    _
    Returns sigma(r) = |\/ n(r)|  and \/ n (r).

    With multiple spins, sigma has the three elements

                   _     __     _  2
            sigma (r) = |\/ n  (r)|  ,
                 0           up

                   _     __     _      __     _
            sigma (r) =  \/ n  (r)  .  \/ n  (r) ,
                 1           up            dn

                   _     __     _  2
            sigma (r) = |\/ n  (r)|  .
                 2           dn
    """
    nspins = len(n_sg)
    gradn_svg = gd.empty((nspins, 3))
    sigma_xg = gd.zeros(nspins * 2 - 1)
    for v in range(3):
        for s in range(nspins):
            grad_v[v](n_sg[s], gradn_svg[s, v])
            axpy(1.0, gradn_svg[s, v]**2, sigma_xg[2 * s])
        if nspins == 2:
            axpy(1.0, gradn_svg[0, v] * gradn_svg[1, v], sigma_xg[1])
    return sigma_xg, gradn_svg


def add_gradient_correction(grad_v, gradn_svg, sigma_xg, dedsigma_xg, v_sg):
    """Add gradient correction to potential.

    ::

                      __   /    de(r)    __      \
        v  (r) += -2  \/ . |  ---------  \/ n(r) |
         xc                \  dsigma(r)          /

    Adds arbitrary data to sigma_xg.  Be sure to pass a copy if you need
    sigma_xg after calling this function.
    """
    nspins = len(v_sg)
    # vv_g is a calculation buffer.  Its contents will be corrupted.
    vv_g = sigma_xg[0]
    for v in range(3):
        for s in range(nspins):
            grad_v[v](dedsigma_xg[2 * s] * gradn_svg[s, v], vv_g)
            axpy(-2.0, vv_g, v_sg[s])
            if nspins == 2:
                grad_v[v](dedsigma_xg[1] * gradn_svg[s, v], vv_g)
                axpy(-1.0, vv_g, v_sg[1 - s])
                # TODO: can the number of gradient evaluations be reduced?


def gga_vars(gd, grad_v, n_sg):
    nspins = len(n_sg)
    sigma_xg, gradn_svg = calculate_sigma(gd, grad_v, n_sg)
    dedsigma_xg = gd.empty(nspins * 2 - 1)
    return sigma_xg, dedsigma_xg, gradn_svg


def get_gradient_ops(gd, nn):
    return [Gradient(gd, v, n=nn).apply for v in range(3)]


class SFG(XCFunctional):
    def __init__(self, kernel, stencil=2):
        XCFunctional.__init__(self, kernel.name, kernel.type)
        self.kernel = kernel
        self.stencil_range = stencil

    def calculate(self, gd, n_sg, v_sg=None, e_g=None,
                  n_stot=None, v_scom=None, spin=None):

        if gd is not self.gd:
            self.set_grid_descriptor(gd)
        if e_g is None:
            e_g = gd.empty()
        if v_sg is None:
            v_sg = np.zeros_like(n_sg)

        self.calculate_impl(gd, n_sg, v_sg, e_g,
                            n_stot, v_scom, spin)

        return gd.integrate(e_g)


    def set_grid_descriptor(self, gd):
        XCFunctional.set_grid_descriptor(self, gd)
        self.grad_v = get_gradient_ops(gd, self.stencil_range)

    def todict(self):
        d = super(SFG, self).todict()
        d['stencil'] = self.stencil_range
        return d

    def get_description(self):
        return ('{} with {} nearest neighbor stencil'
                .format(self.name, self.stencil_range))

    def calculate_impl(self, gd, n_sg, v_sg, e_g,
                       n_stot=None, v_scom=None, spin=None):

        sigma_xg, dedsigma_xg, gradn_svg = gga_vars(gd, self.grad_v, n_sg)
        self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg,
                              n_stot=n_stot, v_scom=v_scom,
                              spin=spin
                              )
        add_gradient_correction(self.grad_v, gradn_svg, sigma_xg,
                                dedsigma_xg, v_sg)

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None,
                                 D_sp_tot=None, dEdD_sp_tot=None,
                                 spin=None
                                 ):

        rcalc = SFGRadialCalculator(self.kernel)
        expansion = SFGRadialExpansion(rcalc)
        return calculate_paw_correction(expansion,
                                        setup, D_sp, dEdD_sp,
                                        addcoredensity, a,
                                        D_sp_total=D_sp_tot,
                                        dEdD_sp_tot=dEdD_sp_tot,
                                        spin=spin
                                        )

    def stress_tensor_contribution(self, n_sg):

        raise NotImplementedError

        sigma_xg, gradn_svg = calculate_sigma(
            self.gd, self.grad_v, n_sg)
        nspins = len(n_sg)
        dedsigma_xg = self.gd.empty(nspins * 2 - 1)
        v_sg = self.gd.zeros(nspins)
        e_g = self.gd.empty()
        self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)

        def integrate(a1_g, a2_g=None):
            return self.gd.integrate(a1_g, a2_g, global_integral=False)

        P = integrate(e_g)
        for v_g, n_g in zip(v_sg, n_sg):
            P -= integrate(v_g, n_g)
        for sigma_g, dedsigma_g in zip(sigma_xg, dedsigma_xg):
            P -= 2 * integrate(sigma_g, dedsigma_g)

        stress_vv = P * np.eye(3)
        for v1 in range(3):
            for v2 in range(3):
                stress_vv[v1, v2] -= integrate(gradn_svg[0, v1] *
                                               gradn_svg[0, v2],
                                               dedsigma_xg[0]) * 2
                if nspins == 2:
                    stress_vv[v1, v2] -= integrate(gradn_svg[0, v1] *
                                                   gradn_svg[1, v2],
                                                   dedsigma_xg[1]) * 2
                    stress_vv[v1, v2] -= integrate(gradn_svg[1, v1] *
                                                   gradn_svg[1, v2],
                                                   dedsigma_xg[2]) * 2
        self.gd.comm.sum(stress_vv)
        return stress_vv

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None,
                            n_sg_total=None, spin=None):

        raise NotImplementedError

        dndr_sg = np.empty_like(n_sg)
        for n_g, dndr_g in zip(n_sg, dndr_sg):
            rgd.derivative(n_g, dndr_g)

        dndr_sg_tot = np.empty_like(n_sg_total)
        for n_g, dndr_g in zip(n_sg_total, dndr_sg_tot):
            rgd.derivative(n_g, dndr_g)


        if e_g is None:
            e_g = rgd.empty()

        rcalc = SFGRadialCalculator(self.kernel)

        e_g[:], dedn_sg = rcalc(
            rgd, n_sg[:, np.newaxis], [1.0], dndr_sg[:, np.newaxis],
            np.zeros((1, 3)), n=None,
            n_sg_total=n_sg_total[:, np.newaxis],
            dndr_sg_tot = dndr_sg_tot[:, np.newaxis], spin=spin)[:2]
        v_sg[:] = dedn_sg
        return rgd.integrate(e_g)


class PurePythonSFGKernel:

    def __init__(self):
        self.name = 'SFG'
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

    g = g_sf(u)
    dg = dg_sf(u)
    h = h_sf(s2, c0)
    f = 1. - (1. - g) * h

    eps = n * f
    dedn = f + u * dg * h - (1. - g) * h**2. * s2 * const2
    dedn_t = -dg * u**2.0 * h
    deda2 = c0 * (1. - g) * h**2 * n / tkfn2

    return eps, dedn,  dedn_t, deda2


def g_sf(u):
    return u


def dg_sf(u):
    return 1.


def h_sf(s2, a):
    return 1. / (1. + a * s2)
