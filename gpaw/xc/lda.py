from math import sqrt, pi

import numpy as np

from gpaw.xc.functional import XCFunctional
from gpaw.sphere.lebedev import Y_nL, weight_n


class LDARadialExpansion:
    def __init__(self, expander, *, n_qg, nc_g):
        self.n_qg = n_qg
        n_sLg = np.dot(expander.D_sLq, n_qg)
        if nc_g is not None:
            n_sLg[:, 0] += nc_g / expander.nspins * (4 * np.pi)**0.5
        self.expander = expander
        self.nspins = len(n_sLg)
        self.rgd = expander.rgd
        
        self.n_sLg = n_sLg
        self.n_sng = np.empty((self.nspins, len(weight_n), self.rgd.N))
        np.einsum('nL,sLg->sng', expander.Y_nL, n_sLg, optimize=True, out=self.n_sng)

    def integrate(self, potential, sign=1.0, dEdD_sp=None):
        E = np.einsum('ng,g,n', potential.e_ng, self.expander.rgd.dv_g, weight_n, optimize=True)
        if dEdD_sp is not None:
            dEdD_sqL = np.einsum('g,n,sng,qg,nL->sqL', 
                                 self.expander.rgd.dv_g,
                                 weight_n,
                                 potential.dedn_sng,
                                 self.n_qg,
                                 self.expander.Y_nL, optimize=True)
            dE = np.einsum('sqL,pqL->sp', dEdD_sqL, self.expander.xcc.B_pqL, optimize=True)
            dEdD_sp += sign * dE
        return sign * E

class LDAPotentialExpansion:
    def __init__(self, dedn_sng, e_ng):
        self.dedn_sng = dedn_sng
        self.e_ng = e_ng

    def empty_like(radial_expansion):
        return LDAPotentialExpansion(np.zeros_like(radial_expansion.n_sng),
                                     np.empty_like(radial_expansion.n_sng[0]))


class LDARadialExpander:
    def __init__(self, setup, D_sp=np.zeros(0)): #rcalc, collinear=True, addcoredensity=True):
        xcc = setup.xc_correction
        
        # Packed density matrix p=(i, j), where i and j are partial wave indices
        self.D_sp = D_sp
        
        # Expansion with respect to independent radial parts times spherical harmonic
        self.D_sLq = np.inner(D_sp, xcc.B_pqL.T)
        self.nspins = len(self.D_sLq)
        self.xcc = xcc
        self.Lmax = xcc.B_pqL.shape[2]
        self.Y_nL = Y_nL[:, :self.Lmax].copy()
        self.rgd = self.xcc.rgd

    def expansion_cls(self):
        return LDARadialExpansion

    def expansion_vars(self, ae=True, addcoredensity=True):
        # Valence radial parts
        n_qg = self.xcc.n_qg if ae else self.xcc.nt_qg
        # Core radial density
        nc_g = self.xcc.nc_g if ae else self.xcc.nct_g
        if not addcoredensity:
            nc_g = None
        return dict(n_qg=n_qg,
                    nc_g=nc_g)

    def expand(self, ae=True, addcoredensity=True):
        dct = self.expansion_vars(ae=ae, addcoredensity=addcoredensity)
        return self.expansion_cls()(self, **dct)


class LDARadialCalculator:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, expansion):
        assert isinstance(expansion, LDARadialExpansion)
        potential = LDAPotentialExpansion.empty_like(expansion)
        self.kernel.calculate(potential.e_ng, expansion.n_sng, potential.dedn_sng)
        return potential


class LDA(XCFunctional):
    def __init__(self, kernel):
        self.kernel = kernel
        XCFunctional.__init__(self, kernel.name, kernel.type)

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        self.kernel.calculate(e_g, n_sg, v_sg)

    def get_radial_expander(self, setup, D_sp):
        return LDARadialExpander(setup, D_sp)

    def get_radial_calculator(self):
        return LDARadialCalculator(self.kernel)

    def calculate_radial(self, rgd, n_sLg, Y_nL):
        rcalc = LDARadialCalculator(self.kernel)
        return rcalc(rgd, n_sLg, Y_nL)

    def stress_tensor_contribution(self, n_sg, skip_sum=False):
        nspins = len(n_sg)
        v_sg = self.gd.zeros(nspins)
        e_g = self.gd.empty()
        self.calculate_impl(self.gd, n_sg, v_sg, e_g)
        stress = self.gd.integrate(e_g, global_integral=False)
        for v_g, n_g in zip(v_sg, n_sg):
            stress -= self.gd.integrate(v_g, n_g, global_integral=False)
        if not skip_sum:
            stress = self.gd.comm.sum_scalar(stress)
        return np.eye(3) * stress


class PurePythonLDAKernel:
    def __init__(self):
        self.name = 'LDA'
        self.type = 'LDA'

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):

        e_g[:] = 0.
        if len(n_sg) == 1:
            n = n_sg[0]
            n[n < 1e-20] = 1e-40

            # exchange
            lda_x(0, e_g, n, dedn_sg[0])
            # correlation
            lda_c(0, e_g, n, dedn_sg[0], 0)

        else:
            na = 2. * n_sg[0]
            na[na < 1e-20] = 1e-40
            nb = 2. * n_sg[1]
            nb[nb < 1e-20] = 1e-40
            n = 0.5 * (na + nb)
            zeta = 0.5 * (na - nb) / n

            # exchange
            lda_x(1, e_g, na, dedn_sg[0])
            lda_x(1, e_g, nb, dedn_sg[1])
            # correlation
            lda_c(1, e_g, n, dedn_sg, zeta)


def lda_x(spin, e, n, v):
    assert spin in [0, 1]
    C0I, C1, CC1, CC2, IF2 = lda_constants()

    rs = (C0I / n) ** (1 / 3.)
    ex = C1 / rs
    dexdrs = -ex / rs
    if spin == 0:
        e[:] += n * ex
    else:
        e[:] += 0.5 * n * ex
    v += ex - rs * dexdrs / 3.


def lda_c(spin, e, n, v, zeta):
    assert spin in [0, 1]
    C0I, C1, CC1, CC2, IF2 = lda_constants()

    rs = (C0I / n) ** (1 / 3.)
    ec, decdrs_0 = G(rs ** 0.5,
                     0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)

    if spin == 0:
        e[:] += n * ec
        v += ec - rs * decdrs_0 / 3.
    else:
        e1, decdrs_1 = G(rs ** 0.5,
                         0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
        alpha, dalphadrs = G(rs ** 0.5,
                             0.016887, 0.11125, 10.357, 3.6231, 0.88026,
                             0.49671)
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
        e[:] += n * ec
        v[0] += ec - rs * decdrs / 3.0 - (zeta - 1.0) * decdzeta
        v[1] += ec - rs * decdrs / 3.0 - (zeta + 1.0) * decdzeta


def G(rtrs, gamma, alpha1, beta1, beta2, beta3, beta4):
    Q0 = -2.0 * gamma * (1.0 + alpha1 * rtrs * rtrs)
    Q1 = 2.0 * gamma * rtrs * (beta1 +
                               rtrs * (beta2 +
                                       rtrs * (beta3 +
                                               rtrs * beta4)))
    G1 = Q0 * np.log(1.0 + 1.0 / Q1)
    dQ1drs = gamma * (beta1 / rtrs + 2.0 * beta2 +
                      rtrs * (3.0 * beta3 + 4.0 * beta4 * rtrs))
    dGdrs = -2.0 * gamma * alpha1 * G1 / Q0 - Q0 * dQ1drs / (Q1 * (Q1 + 1.0))
    return G1, dGdrs


def lda_constants():
    C0I = 0.238732414637843
    C1 = -0.45816529328314287
    CC1 = 1.9236610509315362
    CC2 = 2.5648814012420482
    IF2 = 0.58482236226346462
    return C0I, C1, CC1, CC2, IF2
