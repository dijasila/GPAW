import numpy as np

from gpaw.xc.lda import LDA
from gpaw.utilities.blas import axpy
from gpaw.fd_operators import Gradient


class GGA(LDA):
    def set_grid_descriptor(self, gd):
        LDA.set_grid_descriptor(self, gd)
        self.grad_v = [Gradient(gd, v, allocate=not False).apply
                       for v in range(3)]

    def calculate_lda(self, e_g, n_sg, v_sg):
        gradn_vg = self.gd.empty(3)
        sigma_xg = self.gd.zeros(1)
        dedsigma_xg = self.gd.empty(1)
        for v in range(3):
            self.grad_v[v](n_sg[0], gradn_vg[v])
            axpy(1.0, gradn_vg[v]**2, sigma_xg[0])
        self.calculate_gga(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        tmp_g = gradn_vg[0]
        for v in range(3):
            self.grad_v[v](dedsigma_xg[0] * gradn_vg[v], tmp_g)
            axpy(-2.0, tmp_g, v_sg[0])

    def calculate_gga(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        self.xckernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        
    def calculate_radial(self, rgd, n_sLg, Y_L, v_sg,
                         dndr_sLg, rnablaY_Lv):
        nspins = len(n_sLg)
        e_g = rgd.empty()
        n_sg = np.dot(Y_L, n_sLg)
        rd_vsg = np.dot(rnablaY_Lv.T, n_sLg)
        sigma_xg = rgd.empty(2 * nspins - 1)
        sigma_xg[::2] = (rd_vsg**2).sum(0)
        if nspins == 2:
            sigma_xg[1] = (rd_vsg[:, 0] * rd_vsg[:, 1]).sum(0)
        sigma_xg[:, 1:] /= rgd.r_g[1:]**2
        sigma_xg[:, 0] = sigma_xg[:, 1]
        d_sg = np.dot(Y_L, dndr_sLg)
        sigma_xg[::2] += d_sg**2
        if nspins == 2:
            sigma_xg[1] += d_sg[0] * d_sg[1]
        dedsigma_xg = rgd.zeros(2 * nspins - 1)
        self.xckernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        vv_sg = d_sg  # reuse array
        for s in range(nspins):
            rgd.derivative2(-2 * rgd.dv_g * dedsigma_xg[2 * s] * d_sg[s],
                            vv_sg[s])
        if nspins == 2:
            w_sg = sigma_xg[:2]
            rgd.derivative2(-2 * rgd.dv_g * dedsigma_xg[1] * d_sg[1], w_sg[0])
            rgd.derivative2(-2 * rgd.dv_g * dedsigma_xg[1] * d_sg[0], w_sg[1])
            vv_sg += w_sg
        vv_sg[:, 1:] /= rgd.dv_g[1:]
        vv_sg[:, 0] = vv_sg[:, 1]
        v_sg += vv_sg
        return rgd.integrate(e_g), rd_vsg, dedsigma_xg

    def calculate_spherical(self, rgd, n_sg, v_sg):
        dndr_sg = np.empty_like(n_sg)
        for n_g, dndr_g in zip(n_sg, dndr_sg):
            rgd.derivative(n_g, dndr_g)
        return self.calculate_radial(rgd, n_sg[:, np.newaxis], [1.0], v_sg,
                                     dndr_sg[:, np.newaxis],
                                     np.zeros((1, 3)))[0]
