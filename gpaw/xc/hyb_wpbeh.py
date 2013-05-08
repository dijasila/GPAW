import numpy as np

from math import pi
from math import exp
from math import atan

from scipy.special import erf
from scipy.special import erfc
from scipy.special import expn as expint

from gpaw.xc.lda import LDA
from gpaw.xc.gga import GGA
from gpaw.utilities.blas import axpy
from gpaw.fd_operators import Gradient
from gpaw.sphere.lebedev import Y_nL, weight_n
from gpaw.xc.pawcorrection import rnablaY_nLv


class wpbehkernel:
    def __init__(self, name):
        self.name = name
        self.type = 'GGA'
        self.name_pbe = 'PBE'

    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg,
                  tau_sg=None, dedtau_sg=None):

        e_g[:] = 0.
        dedsigma_xg[:] = 0.

        self.kappa, self.mu, self.beta, self.hybrid = pbe_constants(self.name)

        # spin-paired:
        if len(n_sg) == 1:
            n = n_sg[0]
            n[n < 1e-20] = 1e-40

            # exchange
            res = gga_x(self.name, 0, n, sigma_xg[0], self.kappa, self.mu)
            res_pbe = gga_x(self.name_pbe, 0, n, sigma_xg[0], \
            self.kappa, self.mu)

            ex, rs, dexdrs, dexda2 = res
            ex_pbe, rs_pbe, dexdrs_pbe, dexda2_pbe = res_pbe
            # correlation
            res = gga_c(0, n, sigma_xg[0], 0, self.beta)
            ec, rs_, decdrs, decda2, decdzeta = res

            e_g[:] += n * (-ex * self.hybrid + ex_pbe + ec)
            v_sg[:] += ex + ec - rs * (dexdrs + decdrs) / 3.
            dedsigma_xg[:] += n * (dexda2 + decda2)

        # spin-polarized:
        else:
            na = 2. * n_sg[0]
            na[na < 1e-20] = 1e-40

            nb = 2. * n_sg[1]
            nb[nb < 1e-20] = 1e-40

            n = 0.5 * (na + nb)
            zeta = 0.5 * (na - nb) / n

            # exchange
            exa, rsa, dexadrs, dexada2 = gga_x(
                   self.name, 1, na, 4.0 * sigma_xg[0], self.kappa, self.mu)
            exb, rsb, dexbdrs, dexbda2 = gga_x(
                   self.name, 1, nb, 4.0 * sigma_xg[2], self.kappa, self.mu)
            exa_pbe, rsa_pbe, dexadrs_pbe, dexada2_pbe = gga_x(
                   self.name_pbe, 1, na, 4.0 * sigma_xg[0], \
                   self.kappa, self.mu)
            exb_pbe, rsb_pbe, dexbdrs_pbe, dexbda2_pbe = gga_x(
                   self.name_pbe, 1, nb, 4.0 * sigma_xg[2], \
                   self.kappa, self.mu)
            a2 = sigma_xg[0] + 2.0 * sigma_xg[1] + sigma_xg[2]
            # correlation
            ec, rs, decdrs, decda2, decdzeta = gga_c(1, n, a2, zeta, self.beta)

            e_g[:] += 0.5 * (-na * exa * self.hybrid - nb * exb * self.hybrid + \
                      na * exa_pbe + na * exa_pbe) + n * ec
            v_sg[0][:] += (exa + ec - (rsa * dexadrs + rs * decdrs) / 3.0
                            - (zeta - 1.0) * decdzeta)
            v_sg[1][:] += (exb + ec - (rsb * dexbdrs + rs * decdrs) / 3.0
                            - (zeta + 1.0) * decdzeta)
            dedsigma_xg[0][:] += 2.0 * na * dexada2 + n * decda2
            dedsigma_xg[1][:] += 2.0 * n * decda2
            dedsigma_xg[2][:] += 2.0 * nb * dexbda2 + n * decda2


#These constant will be used by HSE
def pbe_constants(name):
    if name in ['HSE06', 'HSE12', 'HSE12s']:
        kappa = 0.804
        mu = 0.2195149727645171
        beta = 0.06672455060314922
        if name == 'HSE06':
            hybrid = 0.25
        if name == 'HSE12':
            hybrid = 0.313
        if name == 'HSE12s':
            hybrid = 0.425
    else:
        raise NotImplementedError(name)

    return kappa, mu, beta, hybrid


def gga_x(name, spin, n, a2, kappa, mu):
    assert spin in [0, 1]

    C0I, C1, C2, C3, CC1, CC2, IF2, GAMMA = gga_constants()
    rs = (C0I / n) ** (1 / 3.)

    # lda part
    ex = C1 / rs
    dexdrs = -ex / rs

    # gga part
    c = (C2 * rs / n) ** 2.
    s2 = a2 * c
    s = np.sqrt(s2)

    if name in ['HSE06', 'HSE12', 'HSE12s', 'UnScHole']:
#0.250 fraction of Exact Exchange
        if name == 'HSE06':
            omega = 0.11

#0.313 fraction of Exact Exchange
        elif name == 'HSE12':
            omega = 0.0978

#0.425 fraction of Exact Exchange
        elif name == 'HSE12s':
            omega = 0.2159

#Unscreened GGA calculation
        elif name == 'UnScHole':
            omega = 1e-7

        Fx, dFxrs, dFxds2 = wpbe_analy_erfc_approx_grad(spin, n, s, omega)

    elif name in ['PBE', 'PBEsol', 'zvPBEsol']:
        x = 1.0 + mu * s2 / kappa
        Fx = 1.0 + kappa - kappa / x
        dFxds2 = mu / (x**2.)
    elif name == 'RPBE':
        arg = np.maximum(-mu * s2 / kappa, -5.e2)
        x = np.exp(arg)
        Fx = 1.0 + kappa * (1.0 - x)
        dFxds2 = mu * x
    else:
        raise NotImplementedError(name)

    if name in ['HSE06', 'HSE12', 'HSE12s', 'UnScHole']:
        dexdrs = dexdrs * Fx + ex * dFxrs
        dexda2 = ex * dFxds2 * c
        ex *= Fx
    else:
        ds2drs = 8.0 * c * a2 / rs
        dexdrs = dexdrs * Fx + ex * dFxds2 * ds2drs
        dexda2 = ex * dFxds2 * c
        ex *= Fx

    return ex, rs, dexdrs, dexda2


def gga_c(spin, n, a2, zeta, BETA):
    assert spin in [0, 1]
    from gpaw.xc.lda import G

    C0I, C1, C2, C3, CC1, CC2, IF2, GAMMA = gga_constants()
    rs = (C0I / n) ** (1 / 3.)

    # lda part
    ec, decdrs_0 = G(rs ** 0.5, 0.031091, 0.21370, 7.5957,
                     3.5876, 1.6382, 0.49294)

    if spin == 0:
        decdrs = decdrs_0
        decdzeta = 0.  # dummy
    else:
        e1, decdrs_1 = G(rs ** 0.5, 0.015545, 0.20548, 14.1189,
                         6.1977, 3.3662, 0.62517)
        alpha, dalphadrs = G(rs ** 0.5, 0.016887, 0.11125, 10.357,
                         3.6231, 0.88026, 0.49671)
        alpha *= -1.
        dalphadrs *= -1.
        zp = 1.0 + zeta
        zm = 1.0 - zeta
        xp = zp ** (1 / 3.)
        xm = zm ** (1 / 3.)
        f = CC1 * (zp * xp + zm * xm - 2.0)
        f1 = CC2 * (xp - xm)
        zeta3 = zeta * zeta * zeta
        zeta4 = zeta * zeta * zeta * zeta
        x = 1.0 - zeta4
        decdrs = (decdrs_0 * (1.0 - f * zeta4) +
                  decdrs_1 * f * zeta4 +
                  dalphadrs * f * x * IF2)
        decdzeta = (4.0 * zeta3 * f * (e1 - ec - alpha * IF2) +
                   f1 * (zeta4 * e1 - zeta4 * ec + x * alpha * IF2))
        ec += alpha * IF2 * f * x + (e1 - ec) * f * zeta4

    # gga part
    n2 = n * n
    if spin == 1:
        phi = 0.5 * (xp * xp + xm * xm)
        phi2 = phi * phi
        phi3 = phi * phi2
        t2 = C3 * a2 * rs / (n2 * phi2)
        y = -ec / (GAMMA * phi3)
    else:
        t2 = C3 * a2 * rs / n2
        y = -ec / GAMMA

    x = np.exp(y)

    A = np.zeros_like(x)
    indices = np.nonzero(y)
    A[indices] = (BETA / (GAMMA * (x[indices] - 1.0)))

    At2 = A * t2
    nom = 1.0 + At2
    denom = nom + At2 * At2
    H = GAMMA * np.log(1.0 + BETA * t2 * nom / (denom * GAMMA))
    tmp = (GAMMA * BETA / (denom * (BETA * t2 * nom + GAMMA * denom)))
    tmp2 = A * A * x / BETA
    dAdrs = tmp2 * decdrs
    if spin == 1:
        H *= phi3
        tmp *= phi3
        dAdrs /= phi3
    dHdt2 = (1.0 + 2.0 * At2) * tmp
    dHdA = -At2 * t2 * t2 * (2.0 + At2) * tmp
    decdrs += dHdt2 * 7.0 * t2 / rs + dHdA * dAdrs
    decda2 = dHdt2 * C3 * rs / n2
    if spin == 1:
        dphidzeta = np.zeros_like(x)
        ind1 = np.nonzero(xp)
        ind2 = np.nonzero(xm)
        dphidzeta[ind1] += 1.0 / (3.0 * xp[ind1])
        dphidzeta[ind2] -= 1.0 / (3.0 * xm[ind2])
        dAdzeta = tmp2 * (decdzeta - 3.0 * ec * dphidzeta / phi) / phi3
        decdzeta += ((3.0 * H / phi - dHdt2 * 2.0 * t2 / phi) * dphidzeta
                      + dHdA * dAdzeta)
        decda2 /= phi2
    ec += H

    return ec, rs, decdrs, decda2, decdzeta


def gga_constants():
    from gpaw.xc.lda import lda_constants
    C0I, C1, CC1, CC2, IF2 = lda_constants()
    C2 = 0.26053088059892404
    C3 = 0.10231023756535741
    GAMMA = 0.0310906908697

    return C0I, C1, C2, C3, CC1, CC2, IF2, GAMMA


#This part of of code has been taken from JDFTx; R. Sundararaman, K. Letchworth-Weaver and T.A. Arias, JDFTx, available from http://jdftx.sourceforge.net (2012)

#!Evaluate \f$ \int_0^\infty dy y^n e^{-Ay^2} \textrm{erfc}{By} \f$ and its derivatives
# integralErfcGaussian(double A, double B, double& result_A, double& result_B);

# Slater exchange as a function of rs (PER PARTICLE):
#1
def integralErfcGaussian_1(A, B):
    invApBsq = 1. / (A + B * B)
    #(A+B^2) ^ (-1/2)
    invsqrtApBsq = np.sqrt(invApBsq)
    #(A+B^2) ^ (-3/2)
    inv3sqrtApBsq = invsqrtApBsq * invApBsq
    invA = 1. / A
    result_A = (B * (1.5 * A + B * B) * inv3sqrtApBsq - 1.) * \
               0.5 * invA * invA
    result_B = -0.5 * inv3sqrtApBsq
    return ((1. - B * invsqrtApBsq) * 0.5 * invA, result_A, result_B)


#2
def integralErfcGaussian_2(A, B):
    invsqrtPi = 1. / np.sqrt(pi)
    invApBsq = 1. / (A + B * B)
    invA = 1. / A
    atanTerm = np.arctan(np.sqrt(A) / B) * invA / np.sqrt(A)
    result_A = invsqrtPi * (B * (1.25 + 0.75 * B * B * invA) * \
               invApBsq * invApBsq * invA - 0.75 * atanTerm / A)
    result_B = -invsqrtPi * invApBsq * invApBsq
    return (0.5 * invsqrtPi * (-B * invApBsq / A + atanTerm), result_A, result_B)


#3
def integralErfcGaussian_3(A, B):
    invApBsq = 1. / (A + B * B)
    # (A+B^2) ^ (-1/2)
    invsqrtApBsq = np.sqrt(invApBsq)
    # (A+B^2) ^ (-3/2)
    inv3sqrtApBsq = invsqrtApBsq * invApBsq
    # (A+B^2) ^ (-5/2)
    inv5sqrtApBsq = inv3sqrtApBsq * invApBsq
    invA = 1. / A
    result_A = (-invA * invA + B * (0.375 * inv5sqrtApBsq + invA * \
               (0.5 * inv3sqrtApBsq + invA * invsqrtApBsq))) * invA
    result_B = -0.75 * inv5sqrtApBsq
    return ((1. - B * (invsqrtApBsq + 0.5 * A * inv3sqrtApBsq)) * 0.5 * invA * invA, result_A, result_B)


#5
def integralErfcGaussian_5(A, B):
    invApBsq = 1. / (A + B * B)
    #(A+B^2) ^ (-1/2)
    invsqrtApBsq = np.sqrt(invApBsq)
    #(A+B^2) ^ (-3/2)
    inv3sqrtApBsq = invsqrtApBsq * invApBsq
    #(A+B^2) ^ (-5/2)
    inv5sqrtApBsq = inv3sqrtApBsq * invApBsq
    #(A+B^2) ^ (-7/2)
    inv7sqrtApBsq = inv5sqrtApBsq * invApBsq
    invA = 1. / A
    result_A = -3. * invA * (invA * invA * invA - \
                B * (0.3125 * inv7sqrtApBsq + invA * (0.375 * inv5sqrtApBsq + \
                invA * (0.5 * inv3sqrtApBsq + invA * invsqrtApBsq))))
    result_B = -1.875 * inv7sqrtApBsq
    return (invA * (invA * invA - B * (0.375 * inv5sqrtApBsq + \
           invA * (0.5 * inv3sqrtApBsq + invA * invsqrtApBsq))), result_A, result_B)


#Short-ranged omega-PBE GGA exchange - used in the HSE06 hybrid functional
#[J Heyd, G E Scuseria, and M Ernzerhof, J. Chem. Phys. 118, 3865 (2003)]
def wpbe_analy_erfc_approx_grad(spin, rs, s2, omega):
    assert spin in [0, 1]
#omega-PBE enhancement factor:
#Parametrization of PBE hole [J. Perdew and M. Ernzerhof, J. Chem. Phys. 109, 3313 (1998)]
    A = 1.0161144
    B = -0.37170836
    C = -0.077215461
    D = 0.57786348
    #-- Function h := s2 H using the Pade approximant eqn (A5) for H(s)
    H_a1 = 0.00979681
    H_a2 = 0.0410834
    H_a3 = 0.187440
    H_a4 = 0.00120824
    H_a5 = 0.0347188
    s = np.sqrt(s2)
    hNum = s2 * s2 * (H_a1 + s2 * H_a2)
    hNum_s2 = s2 * (2. * H_a1 + s2 * (3. * H_a2))
    hDenInv = 1. / (1. + s2 * s2 * (H_a3 + s * H_a4 + s2 * H_a5))
    hDen_s2 = s2 * (2. * H_a3 + s * (2.5 * H_a4) + s2 * (3. * H_a5))
    h = hNum * hDenInv
    h_s2 = (hNum_s2 - hNum * hDenInv * hDen_s2) * hDenInv
    #Function f := C (1 + s2 F) in terms of h using eqn (25)
    #and noting that eqn (14) => (4./9)*A*A + B - A*D = -1/2
    f = C - 0.5 * h - (1. / 27) * s2
    f_s2 = -0.5 * h_s2 - (1. / 27)
    #Function g := E (1 + s2 G) in terms of f and h using eqns (A1-A3)
    Dph = D + h
    sqrtDph = np.sqrt(Dph)
    gExpArg_h = 2.25 / A
    gExpArg = gExpArg_h * h
    gErfcArg = np.sqrt(gExpArg)
    gExpErfcTerm = np.exp(gExpArg) * erfc(gErfcArg)
    gExpErfcTerm_h = (gExpErfcTerm - (1. / np.sqrt(pi)) / gErfcArg) * gExpArg_h
    g = -Dph * (0.2 * f + Dph * ((4. / 15) * B + Dph * ((8. / 15) * A + \
         sqrtDph * (0.8 * np.sqrt(pi)) * (np.sqrt(A) * gExpErfcTerm - 1.))))
    g_h = -(0.2 * f + Dph * ((8. / 15) * B + Dph * (1.6 * A + \
           sqrtDph * (0.8 * np.sqrt(pi)) * (np.sqrt(A) * \
           (3.5 * gExpErfcTerm + Dph * gExpErfcTerm_h) - 3.5))))
    g_f = -Dph * 0.2
    g_s2 = g_h * h_s2 + g_f * f_s2

    #Accumulate contributions from each gaussian-erfc-polynomial integral:
    #(The prefactor of -8/9 in the enhancement factor is included at the end)
    # prefactor to rs in the argument to erfc
    erfcArgPrefac = omega * (4. / (9 * pi) ** 1. / 3)
    erfcArg = erfcArgPrefac * rs
    I = 0.0
    I_s2 = 0.0
    I_erfcArg = 0.0
    #5 fit gaussians for the 1/y part of the hole from the HSE paper:
    a1 = -0.000205484
    b1 = 0.006601306
    a2 = -0.109465240
    b2 = 0.259931140
    a3 = -0.064078780
    b3 = 0.520352224
    a4 = -0.008181735
    b4 = 0.118551043
    a5 = -0.000110666
    b5 = 0.046003777

    fit1, fit1_h, fit1_erfcArg = integralErfcGaussian_1(b1 + h, erfcArg)
    fit2, fit2_h, fit2_erfcArg = integralErfcGaussian_1(b2 + h, erfcArg)
    fit3, fit3_h, fit3_erfcArg = integralErfcGaussian_2(b3 + h, erfcArg)
    fit4, fit4_h, fit4_erfcArg = integralErfcGaussian_2(b4 + h, erfcArg)
    fit5, fit5_h, fit5_erfcArg = integralErfcGaussian_3(b5 + h, erfcArg)

    I += a1 * fit1 + a2 * fit2 + a3 * fit3 + a4 * fit4 + a5 * fit5
    I_s2 += (a1 * fit1_h + a2 * fit2_h + a3 * fit3_h + a4 * fit4_h + a5 * fit5_h) * h_s2
    I_erfcArg += a1 * fit1_erfcArg + a2 * fit2_erfcArg + \
                 a3 * fit3_erfcArg + a4 * fit4_erfcArg + a5 * fit5_erfcArg
    #Analytical gaussian terms present in the PBE hole:
    term1, term1_h, term1_erfcArg = integralErfcGaussian_1(D + h, erfcArg)
    term2, term2_h, term2_erfcArg = integralErfcGaussian_3(D + h, erfcArg)
    term3, term3_h, term3_erfcArg = integralErfcGaussian_5(D + h, erfcArg)

    I += B * term1 + f * term2 + g * term3
    I_s2 += f_s2 * term2 + g_s2 * term3 + (B * term1_h + f * term2_h + g * term3_h) * h_s2
    I_erfcArg += B * term1_erfcArg + f * term2_erfcArg + g * term3_erfcArg
    #GGA result:
    fx = (-8. / 9) * I
    dfxdrs = (-8. / 9) * (I + I_erfcArg * erfcArgPrefac)
    dfxds2 = (-8. / 9) * I_s2
    return fx, dfxdrs, dfxds2
