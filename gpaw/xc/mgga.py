from math import sqrt, pi

import numpy as np

from gpaw.xc.gga import GGA
from gpaw.utilities.blas import axpy
from gpaw.fd_operators import Gradient
from gpaw.lfc import LFC
from gpaw.sphere.lebedev import weight_n


class MGGA(GGA):
    orbital_dependent = True

    def __init__(self, kernel, nn=1):
        """Meta GGA functional.

        nn: int
            Number of neighbor grid points to use for FD stencil for
            wave function gradient.
        """
        self.nn = nn
        GGA.__init__(self, kernel)

    def set_grid_descriptor(self, gd):
        GGA.set_grid_descriptor(self, gd)

    def get_setup_name(self):
        return 'PBE'

    def initialize(self, density, hamiltonian, wfs, occupations):
        self.wfs = wfs
        self.tauct = LFC(wfs.gd,
                         [[setup.tauct] for setup in wfs.setups],
                         forces=True, cut=True)
        self.tauct_G = None
        self.dedtaut_sG = None
        self.restrict = hamiltonian.restrictor.apply
        self.interpolate = density.interpolator.apply
        self.taugrad_v = [Gradient(wfs.gd, v, n=self.nn, dtype=wfs.dtype).apply
                          for v in range(3)]

    def set_positions(self, spos_ac):
        self.tauct.set_positions(spos_ac)
        if self.tauct_G is None:
            self.tauct_G = self.wfs.gd.empty()
        self.tauct_G[:] = 0.0
        self.tauct.add(self.tauct_G)

    def calculate_gga(self, e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg):
        taut_sG = self.wfs.calculate_kinetic_energy_density(self.taugrad_v)
        taut_sg = np.empty_like(nt_sg)
        for taut_G, taut_g in zip(taut_sG, taut_sg):
            taut_G += 1.0 / self.wfs.nspins * self.tauct_G
            self.interpolate(taut_G, taut_g)
        dedtaut_sg = np.empty_like(nt_sg)
        self.kernel.calculate(e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg,
                              taut_sg, dedtaut_sg)
        self.dedtaut_sG = self.wfs.gd.empty(self.wfs.nspins)
        self.ekin = 0.0
        for s in range(self.wfs.nspins):
            self.restrict(dedtaut_sg[s], self.dedtaut_sG[s])
            self.ekin -= self.wfs.gd.integrate(
                self.dedtaut_sG[s] * (taut_sG[s] -
                                      self.tauct_G / self.wfs.nspins))

    def apply_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                            Htpsit_xG, dH_asp):
        a_G = self.wfs.gd.empty(dtype=psit_xG.dtype)
        for psit_G, Htpsit_G in zip(psit_xG, Htpsit_xG):
            for v in range(3):
                self.taugrad_v[v](psit_G, a_G, kpt.phase_cd)
                self.taugrad_v[v](self.dedtaut_sG[kpt.s] * a_G, a_G,
                                  kpt.phase_cd)
                axpy(-0.5, a_G, Htpsit_G)

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None):
        assert not hasattr(self, 'D_sp')
        self.D_sp = D_sp
        self.n = 0
        self.ae = True
        self.c = setup.xc_correction
        self.dEdD_sp = dEdD_sp

        if self.c.tau_npg is None:
            self.c.tau_npg, self.c.taut_npg = self.initialize_kinetic(self.c)
            print('TODO: tau_ypg is HUGE!  There must be a better way.')

        E = GGA.calculate_paw_correction(self, setup, D_sp, dEdD_sp,
                                         addcoredensity, a)
        del self.D_sp, self.n, self.ae, self.c, self.dEdD_sp
        return E

    def calculate_gga_radial(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        nspins = len(n_sg)
        if self.ae:
            tau_pg = self.c.tau_npg[self.n]
            tauc_g = self.c.tauc_g / (sqrt(4 * pi) * nspins)
            sign = 1.0
        else:
            tau_pg = self.c.taut_npg[self.n]
            tauc_g = self.c.tauct_g / (sqrt(4 * pi) * nspins)
            sign = -1.0
        tau_sg = np.dot(self.D_sp, tau_pg) + tauc_g
        dedtau_sg = np.empty_like(tau_sg)
        self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg,
                              tau_sg, dedtau_sg)
        if self.dEdD_sp is not None:
            self.dEdD_sp += (sign * weight_n[self.n] *
                             np.inner(dedtau_sg * self.c.rgd.dv_g, tau_pg))
        self.n += 1
        if self.n == len(weight_n):
            self.n = 0
            self.ae = False

    def calculate_spherical(self, rgd, n_sg, v_sg):
        raise NotImplementedError

    def add_forces(self, F_av):
        dF_av = self.tauct.dict(derivative=True)
        self.tauct.derivative(self.dedtaut_sG.sum(0), dF_av)
        for a, dF_v in dF_av.items():
            F_av[a] += dF_v[0]

    def estimate_memory(self, mem):
        bytecount = self.wfs.gd.bytecount()
        mem.subnode('MGGA arrays', (1 + self.wfs.nspins) * bytecount)

    def initialize_kinetic(self, xccorr):
        nii = xccorr.nii
        nn = len(xccorr.rnablaY_nLv)
        ng = len(xccorr.phi_jg[0])

        tau_npg = np.zeros((nn, nii, ng))
        taut_npg = np.zeros((nn, nii, ng))
        self.create_kinetic(xccorr, nn, xccorr.phi_jg, tau_npg)
        self.create_kinetic(xccorr, nn, xccorr.phit_jg, taut_npg)
        return tau_npg, taut_npg

    def create_kinetic(self, x, ny, phi_jg, tau_ypg):
        """Short title here.

        kinetic expression is::

                                             __         __
          tau_s = 1/2 Sum_{i1,i2} D(s,i1,i2) \/phi_i1 . \/phi_i2 +tauc_s

        here the orbital dependent part is calculated::

          __         __
          \/phi_i1 . \/phi_i2 =
                      __    __
                      \/YL1.\/YL2 phi_j1 phi_j2 +YL1 YL2 dphi_j1 dphi_j2
                                                         ------  ------
                                                           dr     dr
          __    __
          \/YL1.\/YL2 [y] = Sum_c A[L1,c,y] A[L2,c,y] / r**2

        """
        nj = len(phi_jg)
        ni = len(x.jlL)
        nii = ni * (ni + 1) // 2
        dphidr_jg = np.zeros(np.shape(phi_jg))
        for j in range(nj):
            phi_g = phi_jg[j]
            x.rgd.derivative(phi_g, dphidr_jg[j])

        # Second term:
        for y in range(ny):
            i1 = 0
            p = 0
            Y_L = x.Y_nL[y]
            for j1, l1, L1 in x.jlL:
                for j2, l2, L2 in x.jlL[i1:]:
                    c = Y_L[L1] * Y_L[L2]
                    temp = c * dphidr_jg[j1] * dphidr_jg[j2]
                    tau_ypg[y, p, :] += temp
                    p += 1
                i1 += 1
        ##first term
        for y in range(ny):
            i1 = 0
            p = 0
            rnablaY_Lv = x.rnablaY_nLv[y, :x.Lmax]
            Ax_L = rnablaY_Lv[:, 0]
            Ay_L = rnablaY_Lv[:, 1]
            Az_L = rnablaY_Lv[:, 2]
            for j1, l1, L1 in x.jlL:
                for j2, l2, L2 in x.jlL[i1:]:
                    temp = (Ax_L[L1] * Ax_L[L2] + Ay_L[L1] * Ay_L[L2]
                            + Az_L[L1] * Az_L[L2])
                    temp *= phi_jg[j1] * phi_jg[j2]
                    temp[1:] /= x.rgd.r_g[1:] ** 2
                    temp[0] = temp[1]
                    tau_ypg[y, p, :] += temp
                    p += 1
                i1 += 1
        tau_ypg *= 0.5

        return


class PurePythonMGGAKernel:
    def __init__(self, name='pyTPSSx'):
        assert name in ['pyTPSSx', 'pyrevTPSSx']
        self.name = name
        self.type = 'MGGA'

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg, dedsigma_xg,
                  tau_sg, dedtau_sg):

        e_g[:] = 0.
        dedsigma_xg[:] = 0.
        dedtau_sg[:] = 0.

        # spin-paired:
        if len(n_sg) == 1:
            n = n_sg[0]
            n[n < 1e-20] = 1e-40
            sigma = sigma_xg[0]
            sigma[sigma < 1e-20] = 1e-40
            tau = tau_sg[0]
            tau[tau < 1e-20] = 1e-40

            # exchange
            e_x = x_tpss_para(n, sigma, tau, self.name)
            e_g[:] += e_x * n

        # spin-polarized:
        else:
            n = n_sg
            n[n < 1e-20] = 1e-40
            sigma = sigma_xg
            sigma[sigma < 1e-20] = 1e-40
            tau = tau_sg
            tau[tau < 1e-20] = 1e-40

            # The spin polarized version is handle using the exact spin scaling
            # Ex[n1, n2] = (Ex[2*n1] + Ex[2*n2])/2
            na = 2.0 * n[0]
            nb = 2.0 * n[1]

            e2na = x_tpss_para(na, 4. * sigma[0], 2. * tau[0], self.name)
            e2nb = x_tpss_para(nb, 4. * sigma[2], 2. * tau[1], self.name)

            ea = e2na * na
            eb = e2nb * nb

            e_x = (ea + eb) / 2.0
            e_g[:] += e_x


def x_tpss_7(p, alpha):
    b = 0.40
    h = 9.0 / 20.0
    a = np.sqrt(1.0 + b * alpha * (alpha - 1.0))
    qb = np.divide(h * (alpha - 1.0), a) + np.divide(2.0 * p, 3.0)
    return qb


def x_tpss_10(p, alpha, name):
    if name is 'pyTPSSx':
        c = 1.59096
        e = 1.537
        mu = 0.21951
    elif name in ['pyrevTPSSx', 'pyBEErevTPSSx']:
        c = 2.35204
        e = 2.1677
        mu = 0.14
    else:
        raise NotImplementedError('unknown MGGA exchange: %s' % name)

    kappa = 0.804

    # TPSS equation 7:
    qb = x_tpss_7(p, alpha)

    # TPSS equation 10:
    p2 = p * p
    p2 = np.minimum(p2, 1e10)
    aux1 = 10.0 / 81.0
    ap = (3.0 * alpha + 5.0 * p) * (3.0 * alpha + 5.0 * p)
    apsr = (3.0 * alpha + 5.0 * p)

    # first the numerator
    x1 = np.zeros((np.shape(p)))

    # first term
    a = 9.0 * alpha * alpha + 30.0 * alpha * p + 50.0 * p2
    a2 = a * a
    ind = (a2 != 0.).nonzero()
    x1 += aux1 * p
    if name is 'pyTPSSx':
        x1[ind] += np.divide(25.0 * c
            * p2[ind] * p[ind] * ap[ind], a2[ind])
    elif name in ['pyrevTPSSx', 'pyBEErevTPSSx']:
        x1[ind] += np.divide(125.0 * c
            * p2[ind] * p2[ind] * apsr[ind], a2[ind])
    else:
        raise NotImplementedError('unknown MGGA exchange: %s' % name)

    # second term
    a = 146.0 / 2025.0 * qb
    x1 += a * qb

    # third term
    h = 73.0 / (405.0 * np.sqrt(2.0))
    ind = (ap != 0.).nonzero()
    x1[ind] -= np.divide(h * qb[ind] * p[ind],
        apsr[ind]) * np.sqrt(ap[ind] + 9.0)

    # forth term
    a = aux1 * aux1 / kappa
    x1 += a * p2

    # fifth term
    x1[ind] += np.divide(20.0 * np.sqrt(e) * p2[ind], 9.0 * ap[ind])

    # sixth term
    a = e * mu
    x1 += a * p * p2

    # then the denominator
    a = 1.0 + np.sqrt(e) * p
    a2 = a * a
    ind = (a2 != 0.).nonzero()
    x = np.zeros((np.shape(x1)))
    x[ind] = x1[ind] / a2[ind]
    return x


def x_tpss_para(n, sigma, tau_, name):
    C1 = -0.45816529328314287
    C2 = 0.26053088059892404
    kappa = 0.804

    aux = (3. / 10.) * (3.0 * pi * pi) ** (2. / 3.)

    # uniform gas energy and potential
    exunif = lda_x(n)

    # calculate |nabla rho|^2
    gdms = np.maximum(1e-40, sigma)

    # Eq. (4)
    ind = (n != 0.).nonzero()
    p = np.zeros((np.shape(n)))
    p[ind] = np.divide(gdms[ind], (4.0 * (3.0 * pi * pi) ** (2.0 / 3.0)
        * n[ind] ** (8.0 / 3.0)))

    # von Weisaecker kinetic energy density
    tauw = np.zeros((np.shape(n)))
    tauw[ind] = np.maximum(np.divide(gdms[ind], 8.0 * n[ind]), 1e-20)
    tau = np.maximum(tau_, tauw)

    tau_lsda = aux * n ** (5. / 3.)
    dtau_lsdadd = aux * 5. / 3. * n ** (2. / 3.)

    ind = (tau_lsda != 0.).nonzero()
    alpha = np.zeros((np.shape(n)))
    alpha[ind] = np.divide(tau[ind] - tauw[ind], tau_lsda[ind])

    # TPSS equation 10:
    x = x_tpss_10(p, alpha, name)

    # TPSS equation 5:
    Fx = get_Fx(kappa, x, name)

    energy = exunif * Fx
    return energy


def get_Fx(kappa, x, name):

    a = np.divide(kappa, kappa + x)
    Fx = 1.0 + kappa * (1.0 - a)
    return Fx


def lda_x(n):
    C0I = 0.238732414637843
    C1 = -0.45816529328314287

    rs = (C0I / n) ** (1 / 3.)
    ex = C1 / rs
    return ex
