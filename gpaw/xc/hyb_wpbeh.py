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

    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):

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

        Fx, dFxrs, dFxds = wpbe_analy_erfc_approx_grad(spin, n, s, omega)

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
        dexda2 = ex * dFxds * 2 * s * c
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

#This part of the code has been taken from Quantum
#Espresso (http://www.quantum-espresso.org/)
#distribution and has been accordingly modified. More
#details can be found in Heyd's thesis
#in which the FORTRAN code is also provided for the calculation
#of enhancement factor
###############################################################
# HSE (wPBE) stabbing starts HERE
# Note, that you can get PBEhole functional,
# M. Ernzerhof, J. Chem. Phys. 109, 3313 (1998),
# from this by just setting omega = 0
################################################################


def wpbe_constants():
    Zero = 0.0
    One = 1.0
    Two = 2.0
    Three = 3.0
    Four = 4.0
    Five = 5.0
    Six = 6.0
    Seven = 7.0
    Eight = 8.0
    Nine = 9.0
    Ten = 10.0
    Fifteen = 15.0
    Sixteen = 16.0
    r36 = 36.0
    r64 = 64.0
    r81 = 81.0
    r256 = 256.0
    r384 = 384.0
    r864 = 864.0
    r1944 = 1944.0
    r4374 = 4374.0
    r27 = 27.0
    r48 = 48.0
    r120 = 120.0
    r128 = 128.0
    r144 = 144.0
    r288 = 288.0
    r324 = 324.0
    r512 = 512.0
    r729 = 729.0
    r20 = 20.0
    r32 = 32.0
    r243 = 243.0
    r2187 = 2187.0
    r6561 = 6561.0
    r40 = 40.0
    r12 = 12.0
    r25 = 25.0
    r30 = 30.0
    r54 = 54.0
    r75 = 75.0
    r105 = 105.0
    r135 = 135.0
    r1215 = 1215.0
    r15309 = 15309.0
    return Zero, One, Two, Three, Four, \
    Five, Six, Seven, Eight, Nine, Ten, \
    Fifteen, Sixteen, r36, r64, r81, \
    r256, r384, r864, r1944, r4374, \
    r27, r48, r120, r128, r144, r288, \
    r324, r512, r729, r20, r32, r243, \
    r2187, r6561, r40, r12, r25, r30, \
    r54, r75, r105, r135, r1215, r15309


def wpbe_analy_erfc_approx_grad(spin, dens, s, omega):
#############################################################################
#
#     wPBE Enhancement Factor (erfc approx.,analytical, gradients)
#     It also calculates the terms needed to calculate the potential
#     May be I will comment them since I don't need potential for now
##############################################################################
#General constants
    assert spin in [0, 1]

    Zero, One, Two, Three, Four, \
    Five, Six, Seven, Eight, Nine, Ten, \
    Fifteen, Sixteen, r36, r64, r81, \
    r256, r384, r864, r1944, r4374, \
    r27, r48, r120, r128, r144, r288, \
    r324, r512, r729, r20, r32, r243, \
    r2187, r6561, r40, r12, r25, r30, \
    r54, r75, r105, r135, r1215, r15309 = wpbe_constants()

    f12 = One / Two
    f13 = One / Three
    f14 = One / Four
    f18 = One / Eight

    f23 = Two * f13
    f43 = Two * f23

    f32 = Three / Two
    f72 = Seven / Two
    f34 = Three / Four
    f94 = Nine / Four
    f98 = Nine / Eight
    f1516 = Fifteen / Sixteen
    
    pi = np.pi
    pi2 = pi * pi
    pi_23 = pi2 ** f13
    srpi = np.sqrt(pi)

    Three_13 = Three ** f13

#Constants from fit

    ea1 = -1.128223946706117
    ea2 = 1.452736265762971
    ea3 = -1.243162299390327
    ea4 = 0.971824836115601
    ea5 = -0.568861079687373
    ea6 = 0.246880514820192
    ea7 = -0.065032363850763
    ea8 = 0.008401793031216

    eb1 = 1.455915450052607

#Constants for PBE hole

    A = 1.0161144
    B = -3.7170836e-1
    C = -7.7215461e-2
    D = 5.7786348e-1
    E = -5.1955731e-2
    X = - Eight / Nine

#Constants for fit of H(s) (PBE)

    Ha1 = 9.79681e-3
    Ha2 = 4.10834e-2
    Ha3 = 1.87440e-1
    Ha4 = 1.20824e-3
    Ha5 = 3.47188e-2

#     Constants for F(H) (PBE)

    Fc1 = 6.4753871e0
    Fc2 = 4.7965830e-1

#Constants for polynomial expansion for EG for small s

    EGa1 = -2.628417880e-2
    EGa2 = -7.117647788e-2
    EGa3 = 8.534541323e-2

#Constants for large x expansion of exp(x)*ei(-x)

    expei1 = 4.03640e0
    expei2 = 1.15198e0
    expei3 = 5.03627e0
    expei4 = 4.19160e0

#Cutoff criterion below which to use polynomial expansion

    EGscut = 8.0e-2
    wcutoff = 1.4e1
    expfcutoff = 7.0e2

#Calculate prelim variables

    xkf = (Three * pi2 * dens) ** f13
    xkfrho = xkf * dens

    A2 = A * A
    A3 = A2 * A
    A4 = A3 * A
    A12 = np.sqrt(A)
    A32 = A12 * A
    A52 = A32 * A
    A72 = A52 * A

    w = omega / xkf
    w2 = w * w
    w3 = w2 * w
    w4 = w2 * w2
    w5 = w3 * w2
    w6 = w5 * w
    w7 = w6 * w
    w8 = w7 * w

    d1rw = -(One / (Three * dens)) * w

    X = - Eight / Nine

    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    s5 = s4 * s
    s6 = s5 * s

#Calculate wPBE enhancement factor

    Hnum = Ha1 * s2 + Ha2 * s4
    Hden = One + Ha3 * s4 + Ha4 * s5 + Ha5 * s6

    H = Hnum / Hden

    d1sHnum = Two * Ha1 * s + Four * Ha2 * s3
    d1sHden = Four * Ha3 * s3 + Five * Ha4 * s4 + Six * Ha5 * s5

    d1sH = (Hden * d1sHnum - Hnum * d1sHden) / (Hden * Hden)

    F = Fc1 * H + Fc2
    d1sF = Fc1 * d1sH

#Change exponent of Gaussian if we're using the simple approx.

    if(w.all() > wcutoff):
        eb1 = 2.0e0

#Calculate helper variables (should be moved later on...)

    Hsbw = s2 * H + eb1 * w2
    Hsbw2 = Hsbw * Hsbw
    Hsbw3 = Hsbw2 * Hsbw
    Hsbw4 = Hsbw3 * Hsbw
    Hsbw12 = np.sqrt(Hsbw)
    Hsbw32 = Hsbw12 * Hsbw
    Hsbw52 = Hsbw32 * Hsbw
    Hsbw72 = Hsbw52 * Hsbw

    d1sHsbw = d1sH * s2 + Two * s * H
    d1rHsbw = Two * eb1 * d1rw * w

    DHsbw = D + s2 * H + eb1 * w2
    DHsbw2 = DHsbw * DHsbw
    DHsbw3 = DHsbw2 * DHsbw
    DHsbw4 = DHsbw3 * DHsbw
    DHsbw5 = DHsbw4 * DHsbw
    DHsbw12 = np.sqrt(DHsbw)
    DHsbw32 = DHsbw12 * DHsbw
    DHsbw52 = DHsbw32 * DHsbw
    DHsbw72 = DHsbw52 * DHsbw
    DHsbw92 = DHsbw72 * DHsbw

    HsbwA94 = f94 * Hsbw / A
    HsbwA942 = HsbwA94 * HsbwA94
    HsbwA943 = HsbwA942 * HsbwA94
    HsbwA945 = HsbwA943 * HsbwA942
    HsbwA9412 = np.sqrt(HsbwA94)

    DHs = D + s2 * H
    DHs2 = DHs * DHs
    DHs3 = DHs2 * DHs
    DHs4 = DHs3 * DHs
    DHs72 = DHs3 * np.sqrt(DHs)
    DHs92 = DHs72 * DHs

    d1sDHs = Two * s * H + s2 * d1sH

    DHsw = DHs + w2
    DHsw2 = DHsw * DHsw
    DHsw52 = np.sqrt(DHsw) * DHsw2
    DHsw72 = DHsw52 * DHsw

    d1rDHsw = Two * d1rw * w

    if(s.all() > EGscut):
        G_a = srpi * (Fifteen * E + Six * C * (One + F * s2) * DHs + \
              Four * B * (DHs2) + Eight * A * (DHs3)) * \
              (One / (Sixteen * DHs72)) - \
              f34 * pi * np.sqrt(A) * np.exp(f94 * H * s2 / A) * \
              (One - erf(f32 * s * np.sqrt(H / A)))

        d1sG_a = (One / r32) * srpi * \
                 ((r36 * (Two * H + d1sH * s) / (A12 * np.sqrt(H / A))) + \
                 (One / DHs92) * \
                 (-Eight * A * d1sDHs * DHs3 - r105 * d1sDHs * E - \
                 r30 * C * d1sDHs * DHs * (One + s2 * F) + \
                 r12 * DHs2 * (-B * d1sDHs + C * s * (d1sF * s + Two * F))) - \
                 ((r54 * np.exp(f94 * H * s2 / A) * srpi * s * \
                 (Two * H + d1sH * s) * \
                 erfc(f32 * np.sqrt(H / A) * s)) / A12))

        G_b = (f1516 * srpi * s2) / DHs72

        d1sG_b = (Fifteen * srpi * s * (Four * DHs - Seven * d1sDHs * s)) / \
                 (r32 * DHs92)

        EG = - (f34 * pi + G_a) / G_b

        d1sEG = (-Four * d1sG_a * G_b + d1sG_b * (Four * G_a + Three * pi)) / \
                 (Four * G_b * G_b)

    else:
        EG = EGa1 + EGa2 * s2 + EGa3 * s4
        d1sEG = Two * EGa2 * s + Four * EGa3 * s3

#Calculate the terms needed in any case

    term2 = (DHs2 * B + DHs * C + Two * E + DHs * s2 * C * F + \
            Two * s2 * EG) / (Two * DHs3)

    d1sterm2 = (-Six * d1sDHs * (EG * s2 + E) + \
               DHs2 * (-d1sDHs * B + s * C * (d1sF * s + Two * F)) + \
               Two * DHs * (Two * EG * s - d1sDHs * C + \
               s2 * (d1sEG - d1sDHs * C * F))) / \
               (Two * DHs4)

    term3 = - w * (Four * DHsw2 * B + Six * DHsw * C + Fifteen * E + \
              Six * DHsw * s2 * C * F + Fifteen * s2 * EG) / \
             (Eight * DHs * DHsw52)

    d1sterm3 = w * (Two * d1sDHs * DHsw * (Four * DHsw2 * B + \
               Six * DHsw * C + Fifteen * E + \
               Three * s2 * (Five * EG + Two * DHsw * C * F)) + \
               DHs * (r75 * d1sDHs * (EG * s2 + E) + \
               Four * DHsw2 * (d1sDHs * B - \
               Three * s * C * (d1sF * s + Two * F)) - \
               Six * DHsw * (-Three * d1sDHs * C + \
               s * (Ten * EG + Five * d1sEG * s - \
               Three * d1sDHs * s * C * F)))) / \
               (Sixteen * DHs2 * DHsw72)

    d1rterm3 = (-Two * d1rw * DHsw * (Four * DHsw2 * B + \
               Six * DHsw * C + Fifteen * E + \
               Three * s2 * (Five * EG + Two * DHsw * C * F)) + \
               w * d1rDHsw * (r75 * (EG * s2 + E) + \
               Two * DHsw * (Two * DHsw * B + Nine * C + \
               Nine * s2 * C * F))) / \
               (Sixteen * DHs * DHsw72)

    term4 = - w3 * (DHsw * C + Five * E + DHsw * s2 * C * F + \
               Five * s2 * EG) / (Two * DHs2 * DHsw52)

    d1sterm4 = (w3 * (Four * d1sDHs * DHsw * (DHsw * C + Five * E + \
               s2 * (Five * EG + DHsw * C * F)) + \
               DHs * (r25 * d1sDHs * (EG * s2 + E) - \
               Two * DHsw2 * s * C * (d1sF * s + Two * F) + \
               DHsw * (Three * d1sDHs * C + s * (-r20 * EG - \
               Ten * d1sEG * s + \
               Three * d1sDHs * s * C * F))))) / \
               (Four * DHs3 * DHsw72)

    d1rterm4 = (w2 * (-Six * d1rw * DHsw * (DHsw * C + Five * E + \
               s2 * (Five * EG + DHsw * C * F)) + \
               w * d1rDHsw * (r25 * (EG * s2 + E) + \
               Three * DHsw * C * (One + s2 * F)))) / \
               (Four * DHs2 * DHsw72)

    term5 = - w5 * (E + s2 * EG) / \
              (DHs3 * DHsw52)

    d1sterm5 = (w5 * (Six * d1sDHs * DHsw * (EG * s2 + E) + \
               DHs * (-Two * DHsw * s * (Two * EG + d1sEG * s) + \
               Five * d1sDHs * (EG * s2 + E)))) / \
               (Two * DHs4 * DHsw72)

    d1rterm5 = (w4 * Five * (EG * s2 + E) * (-Two * d1rw * DHsw + \
               d1rDHsw * w)) / \
               (Two * DHs3 * DHsw72)

    if((s.all() > 0.0) or (w.all() > 0.0)):
        t10 = (f12) * A * np.log(Hsbw / DHsbw)
        t10d1 = f12 * A * (One / Hsbw - One / DHsbw)
        d1st10 = d1sHsbw * t10d1
        d1rt10 = d1rHsbw * t10d1

#Calculate exp(x)*f(x) depending on size of x

    if(HsbwA94.all() < expfcutoff):
        piexperf = pi * np.exp(HsbwA94) * erfc(HsbwA9412)
#	expei    = Exp(HsbwA94)*Ei(-HsbwA94)
        expei = np.exp(HsbwA94) * (-expint(1, HsbwA94))

    else:
#       print *,dens,s," LARGE HsbwA94"
        piexperf = pi * (One / (srpi * HsbwA9412) - \
                   One / (Two * np.sqrt(pi * HsbwA943)) + \
                   Three / (Four * np.sqrt(pi * HsbwA945)))

        expei = - (One / HsbwA94) * \
                   (HsbwA942 + expei1 * HsbwA94 + expei2) / \
                   (HsbwA942 + expei3 * HsbwA94 + expei4)

#Calculate the derivatives (based on the orig. expression)
#--> Is this ok? ==> seems to be ok...

    piexperfd1 = - (Three * srpi * np.sqrt(Hsbw / A)) / (Two * Hsbw) + \
                    (Nine * piexperf) / (Four * A)
    d1spiexperf = d1sHsbw * piexperfd1
    d1rpiexperf = d1rHsbw * piexperfd1

    expeid1 = f14 * (Four / Hsbw + (Nine * expei) / A)
    d1sexpei = d1sHsbw * expeid1
    d1rexpei = d1rHsbw * expeid1

    if (w.all() == Zero):
#       Fall back to original expression for the PBE hole

        t1 = -f12 * A * expei
        d1st1 = -f12 * A * d1sexpei
        d1rt1 = -f12 * A * d1rexpei

        if (s > 0.0):

            term1 = t1 + t10
            d1sterm1 = d1st1 + d1st10
            d1rterm1 = d1rt1 + d1rt10

            Fx_wpbe = X * (term1 + term2)

            d1sfx = X * (d1sterm1 + d1sterm2)
            d1rfx = X * d1rterm1

        else:

            Fx_wpbe = 1.0

#TODO This is checked to be true for term1
#How about the other terms???

            d1sfx = 0.0
            d1rfx = 0.0

    elif (w.all() > wcutoff):
#Use simple Gaussian approximation for large w
#print *,dens,s," LARGE w"

        term1 = -f12 * A * (expei + np.log(DHsbw) - np.log(Hsbw))
        term1d1 = - A / (Two * DHsbw) - f98 * expei
        d1sterm1 = d1sHsbw * term1d1
        d1rterm1 = d1rHsbw * term1d1

        Fx_wpbe = X * (term1 + term2 + term3 + term4 + term5)

        d1sfx = X * (d1sterm1 + d1sterm2 + d1sterm3 + \
        d1sterm4 + d1sterm5)

        d1rfx = X * (d1rterm1 + d1rterm3 + d1rterm4 + d1rterm5)

    else:
#For everything else, use the full blown expression
#First, we calculate the polynomials for the first term

        np1 = -f32 * ea1 * A12 * w + r27 * ea3 * w3 / (Eight * A12) - \
               r243 * ea5 * w5 / (r32 * A32) + r2187 * \
               ea7 * w7 / (r128 * A52)

        d1rnp1 = - f32 * ea1 * d1rw * A12 + (r81 * ea3 * d1rw * w2) / \
                   (Eight * A12) - (r1215 * ea5 * d1rw * w4) / (r32 * A32) + \
                   (r15309 * ea7 * d1rw * w6) / (r128 * A52)

        np2 = -A + f94 * ea2 * w2 - r81 * ea4 * w4 / (Sixteen * A) + \
               r729 * ea6 * w6 / (r64 * A2) - r6561 * ea8 * w8 / (r256 * A3)

        d1rnp2 = f12 * (Nine * ea2 * d1rw * w) - \
                 (r81 * ea4 * d1rw * w3) / (Four * A) + \
                 (r2187 * ea6 * d1rw * w5) / (r32 * A2) - \
                 (r6561 * ea8 * d1rw * w7) / (r32 * A3)

#The first term is

        t1 = f12 * (np1 * piexperf + np2 * expei)
        d1st1 = f12 * (d1spiexperf * np1 + d1sexpei * np2)
        d1rt1 = f12 * (d1rnp2 * expei + d1rpiexperf * np1 + \
        d1rexpei * np2 + d1rnp1 * piexperf)

#The factors for the main polynomoal in w and their derivatives

        f2 = (f12) * ea1 * srpi * A / DHsbw12
        f2d1 = - ea1 * srpi * A / (Four * DHsbw32)
        d1sf2 = d1sHsbw * f2d1
        d1rf2 = d1rHsbw * f2d1

        f3 = (f12) * ea2 * A / DHsbw
        f3d1 = - ea2 * A / (Two * DHsbw2)
        d1sf3 = d1sHsbw * f3d1
        d1rf3 = d1rHsbw * f3d1

        f4 = ea3 * srpi * (-f98 / Hsbw12 + f14 * A / DHsbw32)
                 
        f4d1 = ea3 * srpi * ((Nine / (Sixteen * Hsbw32)) - \
                (Three * A / (Eight * DHsbw52)))
        d1sf4 = d1sHsbw * f4d1
        d1rf4 = d1rHsbw * f4d1

        f5 = ea4 * (One / r128) * (-r144 * (One / Hsbw) + \
             r64 * (One / DHsbw2) * A)
        f5d1 = ea4 * ((f98 / Hsbw2) - (A / DHsbw3))
        d1sf5 = d1sHsbw * f5d1
        d1rf5 = d1rHsbw * f5d1

        f6 = ea5 * (Three * srpi * (Three * DHsbw52 * \
                (Nine * Hsbw - Two * A) + Four * Hsbw32 * A2)) / \
                (r32 * DHsbw52 * Hsbw32 * A)
        f6d1 = ea5 * srpi * ((r27 / (r32 * Hsbw52)) - \
                (r81 / (r64 * Hsbw32 * A)) - \
                ((Fifteen * A) / (Sixteen * DHsbw72)))
        d1sf6 = d1sHsbw * f6d1
        d1rf6 = d1rHsbw * f6d1

        f7 = ea6 * (((r32 * A) / DHsbw3 + \
             (-r36 + (r81 * s2 * H) / A) / Hsbw2)) / r32

        d1sf7 = ea6 * (Three * (r27 * d1sH * DHsbw4 * Hsbw * s2 + \
                Eight * d1sHsbw * A * (Three * DHsbw4 - Four * Hsbw3 * A) + \
                r54 * DHsbw4 * s * (Hsbw - d1sHsbw * s) * H)) / \
                (r32 * DHsbw4 * Hsbw3 * A)
        d1rf7 = ea6 * d1rHsbw * ((f94 / Hsbw3) - ((Three * A) / DHsbw4) - \
                ((r81 * s2 * H) / (Sixteen * Hsbw3 * A)))

        f8 = ea7 * (-Three * srpi * (-r40 * Hsbw52 * A3 + \
             Nine * DHsbw72 * (r27 * Hsbw2 - Six * Hsbw * A + \
             Four * A2))) / (r128 * DHsbw72 * Hsbw52 * A2)
                 
        f8d1 = ea7 * srpi * ((r135 / (r64 * Hsbw72)) + \
               (r729 / (r256 * Hsbw32 * A2)) - \
               (r243 / (r128 * Hsbw52 * A)) - \
               ((r105 * A) / (r32 * DHsbw92)))
        d1sf8 = d1sHsbw * f8d1
        d1rf8 = d1rHsbw * f8d1

        f9 = (r324 * ea6 * eb1 * DHsbw4 * Hsbw * A + \
             ea8 * (r384 * Hsbw3 * A3 + DHsbw4 * (-r729 * Hsbw2 + \
             r324 * Hsbw * A - r288 * A2))) / \
             (r128 * DHsbw4 * Hsbw3 * A2)
        f9d1 = -((r81 * ea6 * eb1) / (Sixteen * Hsbw3 * A)) + \
                ea8 * ((r27 / (Four * Hsbw4)) + \
                (r729 / (r128 * Hsbw2 * A2)) - \
                (r81 / (Sixteen * Hsbw3 * A)) - ((r12 * A / DHsbw5)))
                      
        d1sf9 = d1sHsbw * f9d1
        d1rf9 = d1rHsbw * f9d1

        t2t9 = f2 * w + f3 * w2 + f4 * w3 + f5 * w4 + f6 * w5 + \
               f7 * w6 + f8 * w7 + f9 * w8
        d1st2t9 = d1sf2 * w + d1sf3 * w2 + d1sf4 * w3 + d1sf5 * w4 + \
                  d1sf6 * w5 + d1sf7 * w6 + d1sf8 * w7 + d1sf9 * w8
                         
        d1rt2t9 = d1rw * f2 + d1rf2 * w + Two * d1rw * f3 * w + \
                  d1rf3 * w2 + Three * d1rw * f4 * w2 + \
                  d1rf4 * w3 + Four * d1rw * f5 * w3 + \
                  d1rf5 * w4 + Five * d1rw * f6 * w4 + \
                  d1rf6 * w5 + Six * d1rw * f7 * w5 + \
                  d1rf7 * w6 + Seven * d1rw * f8 * w6 + \
                  d1rf8 * w7 + Eight * d1rw * f9 * w7 + d1rf9 * w8

#The final value of term1 for 0 < omega < wcutoff is:

        term1 = t1 + t2t9 + t10

        d1sterm1 = d1st1 + d1st2t9 + d1st10
        d1rterm1 = d1rt1 + d1rt2t9 + d1rt10

#The final value for the enhancement factor and its
#derivatives is:

        Fx_wpbe = X * (term1 + term2 + term3 + term4 + term5)

        d1sfx = X * (d1sterm1 + d1sterm2 + d1sterm3 + \
                d1sterm4 + d1sterm5)

        d1rfx = X * (d1rterm1 + d1rterm3 + d1rterm4 + d1rterm5)
        return Fx_wpbe, d1rfx, d1sfx
