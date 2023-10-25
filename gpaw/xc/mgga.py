from math import sqrt, pi

import numpy as np
from scipy.special import eval_legendre

from gpaw.xc.gga import (add_gradient_correction, gga_vars,
                         GGARadialExpansion, GGARadialExpander,
                         GGAPotentialExpansion,
                         add_radial_gradient_correction,
                         get_gradient_ops)
from gpaw.xc.functional import XCFunctional
from gpaw.sphere.lebedev import weight_n

class MGGARadialExpansion(GGARadialExpansion):
    def __init__(self, expander, *, n_qg, nc_g, tau_npg, tauc_g):
        GGARadialExpansion.__init__(self, expander, n_qg=n_qg, nc_g=nc_g)
        tau_sng = np.einsum('sp,npg->sng', expander.D_sp, tau_npg).copy()
        tau_sng += tauc_g

        self.tau_sng = tau_sng
        self.tau_npg = tau_npg
        
    def integrate(self, potential, sign=1.0, dEdD_sp=None):
        E = GGARadialExpansion.integrate(self, potential, sign=sign, dEdD_sp=dEdD_sp)
        if dEdD_sp is not None:
            dEdD_sp += sign * np.einsum('n, sng, g, npg->sp',
                                        weight_n, potential.dedtau_sng,
                                        self.rgd.dv_g, self.tau_npg)
        return E


class MGGARadialExpander(GGARadialExpander):
    def __init__(self, setup, D_sp):
        GGARadialExpander.__init__(self, setup, D_sp)
        xcc = self.xcc
        if xcc.tau_npg is None:
            xcc.tau_npg, xcc.taut_npg = self.initialize_kinetic(xcc)
    
    def initialize_kinetic(self, xcc):
        nii = xcc.nii
        nn = len(xcc.rnablaY_nLv)
        ng = len(xcc.phi_jg[0])

        tau_npg = np.zeros((nn, nii, ng))
        taut_npg = np.zeros((nn, nii, ng))
        create_kinetic(xcc, nn, xcc.phi_jg, tau_npg)
        create_kinetic(xcc, nn, xcc.phit_jg, taut_npg)
        return tau_npg, taut_npg

    def expansion_cls(self, *args, **kwargs):
        return MGGARadialExpansion

    def expansion_vars(self, ae=True, addcoredensity=True):
        dct = GGARadialExpander.expansion_vars(self, ae, addcoredensity)
        tau_npg = self.xcc.tau_npg if ae else self.xcc.taut_npg
        tauc_g = self.xcc.tauc_g if ae else self.xcc.tauct_g
        tauc_g = tauc_g / (sqrt(4 * pi) * self.nspins)
        dct.update(tau_npg=tau_npg, tauc_g=tauc_g)
        return dct


class MGGAPotentialExpansion(GGAPotentialExpansion):
    def __init__(self, dedn_sng, e_ng, dedsigma_xng, dedtau_sng):
        GGAPotentialExpansion.__init__(self, dedn_sng, e_ng, dedsigma_xng)
        self.dedtau_sng = dedtau_sng

    def empty_like(radial_expansion):
        s = radial_expansion.nspins
        n = len(weight_n)
        g = radial_expansion.rgd.N
        x = 2 * s - 1
        dedn_sng = np.zeros((s, n, g))
        e_ng = np.empty((n, g))
        dedsigma_xng = np.zeros((x, n, g))
        dedtau_sng = np.zeros((s,n,g))
        return MGGAPotentialExpansion(dedn_sng, e_ng, dedsigma_xng, dedtau_sng)


class MGGARadialCalculator:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, expansion):
        assert isinstance(expansion, GGARadialExpansion)
        potential = MGGAPotentialExpansion.empty_like(expansion)
        potential.dedn_sng[:] = 0.0
        potential.dedtau_sng[:] = 0.0
        assert potential.e_ng.flags.c_contiguous
        assert potential.dedn_sng.flags.c_contiguous
        nspins = expansion.nspins

        self.kernel.calculate(potential.e_ng.ravel(), 
                              expansion.n_sng.reshape((nspins, -1)),
                              potential.dedn_sng.reshape((nspins, -1)), 
                              expansion.sigma_xng.reshape((nspins*2-1, -1)), 
                              potential.dedsigma_xng.reshape((nspins*2-1, -1)),
                              expansion.tau_sng.reshape((nspins, -1)),
                              potential.dedtau_sng.reshape((nspins, -1)))
        potential.dedn_sng += add_radial_gradient_correction(expansion.rgd, expansion.sigma_xng, potential.dedsigma_xng, expansion.a_sng)
        return potential


class MGGA(XCFunctional):
    orbital_dependent = True

    def __init__(self, kernel, stencil=2):
        """Meta GGA functional."""
        XCFunctional.__init__(self, kernel.name, kernel.type)
        self.kernel = kernel
        self.stencil_range = stencil
        self.fixed_ke = False

    def set_grid_descriptor(self, gd):
        self.grad_v = get_gradient_ops(gd, self.stencil_range, np)
        XCFunctional.set_grid_descriptor(self, gd)

    def get_setup_name(self):
        return 'PBE'

    # This method exists on GGA class as well.  Try to solve this
    # kind of problem when refactoring MGGAs one day.
    def get_description(self):
        return ('{} with {} nearest neighbor stencil'
                .format(self.name, self.stencil_range))

    def get_radial_expander(self, setup, D_sp):
        return MGGARadialExpander(setup, D_sp)

    def get_radial_calculator(self):
        return MGGARadialCalculator(self.kernel)

    def initialize(self, density, hamiltonian, wfs):
        self.wfs = wfs
        self.tauct = density.get_pseudo_core_kinetic_energy_density_lfc()
        self.tauct_G = None
        self.dedtaut_sG = None
        if ((not hasattr(hamiltonian, 'xc_redistributor'))
                or (hamiltonian.xc_redistributor is None)):
            self.restrict_and_collect = hamiltonian.restrict_and_collect
            self.distribute_and_interpolate = \
                density.distribute_and_interpolate
        else:
            def _distribute_and_interpolate(in_xR, out_xR=None):
                tmp_xR = density.interpolate(in_xR)
                if hamiltonian.xc_redistributor.enabled:
                    out_xR = hamiltonian.xc_redistributor.distribute(tmp_xR,
                                                                     out_xR)
                elif out_xR is None:
                    out_xR = tmp_xR
                else:
                    out_xR[:] = tmp_xR
                return out_xR

            def _restrict_and_collect(in_xR, out_xR=None):
                if hamiltonian.xc_redistributor.enabled:
                    in_xR = hamiltonian.xc_redistributor.collect(in_xR)
                return hamiltonian.restrict(in_xR, out_xR)
            self.restrict_and_collect = _restrict_and_collect
            self.distribute_and_interpolate = _distribute_and_interpolate

    def set_positions(self, spos_ac):
        self.tauct.set_positions(spos_ac, self.wfs.atom_partition)
        if self.tauct_G is None:
            self.tauct_G = self.wfs.gd.empty()
        self.tauct_G[:] = 0.0
        self.tauct.add(self.tauct_G)

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        sigma_xg, dedsigma_xg, gradn_svg = gga_vars(gd, self.grad_v, n_sg)
        self.process_mgga(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        add_gradient_correction(self.grad_v, gradn_svg, sigma_xg,
                                dedsigma_xg, v_sg)

    def fix_kinetic_energy_density(self, taut_sG):
        self.fixed_ke = True
        self._taut_gradv_init = False
        self._fixed_taut_sG = taut_sG.copy()

    def process_mgga(self, e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg):
        if self.fixed_ke:
            taut_sG = self._fixed_taut_sG
            if not self._taut_gradv_init:
                self._taut_gradv_init = True
                # ensure initialization for calculation potential
                self.wfs.calculate_kinetic_energy_density()
        else:
            taut_sG = self.wfs.calculate_kinetic_energy_density()

        if taut_sG is None:
            taut_sG = self.wfs.gd.zeros(len(nt_sg))

        if 0:  # taut_sG is None:
            # Below code disabled because it produces garbage in at least
            # some cases.
            #
            # See https://gitlab.com/gpaw/gpaw/issues/124
            #
            # Initialize with von Weizsaecker kinetic energy density:
            nt0_sg = nt_sg.copy()
            nt0_sg[nt0_sg < 1e-10] = np.inf
            taut_sg = sigma_xg[::2] / 8 / nt0_sg
            nspins = self.wfs.nspins
            taut_sG = self.wfs.gd.empty(nspins)
            for taut_G, taut_g in zip(taut_sG, taut_sg):
                self.restrict_and_collect(taut_g, taut_G)
        else:
            taut_sg = np.empty_like(nt_sg)

        for taut_G, taut_g in zip(taut_sG, taut_sg):
            taut_G += 1.0 / self.wfs.nspins * self.tauct_G
            self.distribute_and_interpolate(taut_G, taut_g)

        # bad = taut_sg < tautW_sg + 1e-11
        # taut_sg[bad] = tautW_sg[bad]

        # m = 12.0
        # taut_sg = (taut_sg**m + (tautW_sg / 2)**m)**(1 / m)

        dedtaut_sg = np.empty_like(nt_sg)
        self.kernel.calculate(e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg,
                              taut_sg, dedtaut_sg)

        self.dedtaut_sG = self.wfs.gd.empty(self.wfs.nspins)
        self.ekin = 0.0
        for s in range(self.wfs.nspins):
            self.restrict_and_collect(dedtaut_sg[s], self.dedtaut_sG[s])
            self.ekin -= self.wfs.gd.integrate(
                self.dedtaut_sG[s] * (taut_sG[s] -
                                      self.tauct_G / self.wfs.nspins))

    def stress_tensor_contribution(self, n_sg):
        sigma_xg, dedsigma_xg, gradn_svg = gga_vars(self.gd, self.grad_v, n_sg)
        taut_sG = self.wfs.calculate_kinetic_energy_density()
        if taut_sG is None:
            taut_sG = self.wfs.gd.zeros(len(n_sg))
        taut_sg = np.empty_like(n_sg)
        for taut_G, taut_g in zip(taut_sG, taut_sg):
            taut_G += 1.0 / self.wfs.nspins * self.tauct_G
            self.distribute_and_interpolate(taut_G, taut_g)

        nspins = len(n_sg)
        dedtaut_sg = np.empty_like(n_sg)
        v_sg = self.gd.zeros(nspins)
        e_g = self.gd.empty()
        self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg,
                              taut_sg, dedtaut_sg)

        def integrate(a1_g, a2_g=None):
            return self.gd.integrate(a1_g, a2_g, global_integral=False)

        P = integrate(e_g)
        for v_g, n_g in zip(v_sg, n_sg):
            P -= integrate(v_g, n_g)
        for sigma_g, dedsigma_g in zip(sigma_xg, dedsigma_xg):
            P -= 2 * integrate(sigma_g, dedsigma_g)
        for taut_g, dedtaut_g in zip(taut_sg, dedtaut_sg):
            P -= integrate(taut_g, dedtaut_g)

        tau_svvG = self.wfs.calculate_kinetic_energy_density_crossterms()

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
        tau_cross_g = self.gd.empty()
        for s in range(nspins):
            for v1 in range(3):
                for v2 in range(3):
                    self.distribute_and_interpolate(
                        tau_svvG[s, v1, v2], tau_cross_g)
                    stress_vv[v1, v2] -= integrate(tau_cross_g, dedtaut_sg[s])

        self.dedtaut_sG = self.wfs.gd.empty(self.wfs.nspins)
        for s in range(self.wfs.nspins):
            self.restrict_and_collect(dedtaut_sg[s], self.dedtaut_sG[s])

        self.gd.comm.sum(stress_vv)
        return stress_vv

    def apply_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                            Htpsit_xG, dH_asp=None):
        self.wfs.apply_mgga_orbital_dependent_hamiltonian(
            kpt, psit_xG,
            Htpsit_xG, dH_asp,
            self.dedtaut_sG[kpt.s])

    def add_forces(self, F_av):
        dF_av = self.tauct.dict(derivative=True)
        self.tauct.derivative(self.dedtaut_sG.sum(0), dF_av)
        for a, dF_v in dF_av.items():
            F_av[a] += dF_v[0] / self.wfs.nspins

    def estimate_memory(self, mem):
        bytecount = self.wfs.gd.bytecount()
        mem.subnode('MGGA arrays', (1 + self.wfs.nspins) * bytecount)


def create_kinetic(xcc, ny, phi_jg, tau_ypg):
    r"""Short title here.

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
    dphidr_jg = np.zeros(np.shape(phi_jg))
    for j in range(nj):
        phi_g = phi_jg[j]
        xcc.rgd.derivative(phi_g, dphidr_jg[j])

    # Second term:
    for y in range(ny):
        i1 = 0
        p = 0
        Y_L = xcc.Y_nL[y]
        for j1, l1, L1 in xcc.jlL:
            for j2, l2, L2 in xcc.jlL[i1:]:
                c = Y_L[L1] * Y_L[L2]
                temp = c * dphidr_jg[j1] * dphidr_jg[j2]
                tau_ypg[y, p, :] += temp
                p += 1
            i1 += 1
    # first term
    for y in range(ny):
        i1 = 0
        p = 0
        rnablaY_Lv = xcc.rnablaY_nLv[y, :xcc.Lmax]
        Ax_L = rnablaY_Lv[:, 0]
        Ay_L = rnablaY_Lv[:, 1]
        Az_L = rnablaY_Lv[:, 2]
        for j1, l1, L1 in xcc.jlL:
            for j2, l2, L2 in xcc.jlL[i1:]:
                temp = (Ax_L[L1] * Ax_L[L2] + Ay_L[L1] * Ay_L[L2] +
                        Az_L[L1] * Az_L[L2])
                temp *= phi_jg[j1] * phi_jg[j2]
                temp[1:] /= xcc.rgd.r_g[1:]**2
                temp[0] = temp[1]
                tau_ypg[y, p, :] += temp
                p += 1
            i1 += 1
    tau_ypg *= 0.5


class PurePython2DMGGAKernel:
    def __init__(self, name, pars=None):
        self.name = name
        self.pars = pars
        self.type = 'MGGA'
        assert self.pars is not None

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
            e_x = twodexchange(n, sigma, tau, self.pars)
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

            e2na_x = twodexchange(na, 4. * sigma[0], 2. * tau[0], self.pars)
            e2nb_x = twodexchange(nb, 4. * sigma[2], 2. * tau[1], self.pars)
            ea_x = e2na_x * na
            eb_x = e2nb_x * nb

            e_g[:] += (ea_x + eb_x) / 2.0


def twodexchange(n, sigma, tau, pars):
    # parameters for 2 Legendre polynomials
    parlen_i = int(pars[0])
    parlen_j = pars[2 + 2 * parlen_i]
    assert parlen_i == parlen_j
    pars_i = pars[1:2 + 2 * parlen_i]
    pars_j = pars[3 + 2 * parlen_i:]
    trans_i = pars_i[0]
    trans_j = pars_j[0]
    orders_i, coefs_i = np.split(pars_i[1:], 2)
    orders_j, coefs_j = np.split(pars_j[1:], 2)
    assert len(coefs_i) == len(orders_i)
    assert len(coefs_j) == len(orders_j)
    assert len(orders_i) == len(orders_j)

    # product Legendre expansion of Fx(s, alpha)
    e_x_ueg, rs = ueg_x(n)
    Fx = LegendreFx2(n, rs, sigma, tau,
                     trans_i, orders_i, coefs_i, trans_j, orders_j, coefs_j)
    return e_x_ueg * Fx


def LegendreFx2(n, rs, sigma, tau,
                trans_i, orders_i, coefs_i, trans_j, orders_j, coefs_j):
    # Legendre polynomial basis expansion in 2D

    # reduced density gradient in transformation t1(s)
    C2 = 0.26053088059892404
    s2 = sigma * (C2 * np.divide(rs, n))**2.
    x_i = transformation(s2, trans_i)
    assert x_i.all() >= -1.0 and x_i.all() <= 1.0

    # kinetic energy density parameter alpha in transformation t2(s)
    alpha = get_alpha(n, sigma, tau)
    x_j = transformation(alpha, trans_j)
    assert x_j.all() >= -1.0 and x_j.all() <= 1.0

    # product exchange enhancement factor
    Fx_i = legendre_polynomial(x_i, orders_i, coefs_i)
    # print(Fx_i);asdf
    Fx_j = legendre_polynomial(x_j, orders_j, coefs_j)
    Fx = Fx_i * Fx_j
    return Fx


def transformation(x, t):
    if t > 0:
        tmp = t + x
        x = 2.0 * np.divide(x, tmp) - 1.0
    elif int(t) == -1:
        tmp1 = (1.0 - x**2.0)**3.0
        tmp2 = (1.0 + x**3.0 + x**6.0)
        x = -1.0 * np.divide(tmp1, tmp2)
    else:
        raise KeyError('transformation %i unknown!' % t)
    return x


def get_alpha(n, sigma, tau):
    # tau LSDA
    aux = (3. / 10.) * (3.0 * np.pi * np.pi)**(2. / 3.)
    tau_lsda = aux * n**(5. / 3.)

    # von Weisaecker
    ind = (n != 0.).nonzero()
    gdms = np.maximum(sigma, 1e-40)  # |nabla rho|^2
    tau_w = np.zeros(np.shape(n))
    tau_w[ind] = np.maximum(np.divide(gdms[ind], 8.0 * n[ind]), 1e-40)

    # z and alpha
    tau_ = np.maximum(tau_w, tau)
    alpha = np.divide(tau_ - tau_w, tau_lsda)
    assert alpha.all() >= 0.0
    return alpha


def ueg_x(n):
    C0I = 0.238732414637843
    C1 = -0.45816529328314287
    rs = (C0I / n)**(1 / 3.)
    ex = C1 / rs
    return ex, rs


def legendre_polynomial(x, orders, coefs):
    assert len(orders) == len(coefs) == 1
    return eval_legendre(orders[0], x) * coefs[0]
