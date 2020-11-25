from math import sqrt, pi
import numpy as np

from gpaw.xc.functional import XCFunctional
from gpaw.sphere.lebedev import Y_nL, weight_n


class SFRadialExpansion:

    def __init__(self, rcalc, collinear=True):
        self.rcalc = rcalc
        self.collinear = collinear

    def __call__(self, rgd, D_sLq, n_qg, nc0_sg, D_sLq_total, spin):
        n_sLg = np.dot(D_sLq, n_qg)
        n_sLg_total = np.dot(D_sLq_total, n_qg)

        if self.collinear:
            n_sLg_total[spin, 0] += nc0_sg[spin]
        else:
            n_sLg_total[0, 0] += 4 * nc0_sg[0]

        dEdD_sqL = np.zeros_like(np.transpose(D_sLq, (0, 2, 1)))
        dEdD_sqL_com = np.zeros_like(np.transpose(D_sLq, (0, 2, 1)))

        Lmax = n_sLg.shape[1]
        E = 0.0
        for n, Y_L in enumerate(Y_nL[:, :Lmax]):
            w = weight_n[n]

            e_g, dedn_sg, v_scom = self.rcalc(rgd, n_sLg, Y_L,
                                                 n_sLg_total, spin)

            dEdD_sqL += np.dot(rgd.dv_g * dedn_sg,
                               n_qg.T)[:, :, np.newaxis] * (w * Y_L)
            dEdD_sqL_com += np.dot(rgd.dv_g * v_scom,
                                   n_qg.T)[:, :, np.newaxis] * (w * Y_L)

            E += w * rgd.integrate(e_g)

        return E, dEdD_sqL, dEdD_sqL_com


def calculate_paw_correction(expansion,
                             setup, D_sp, dEdD_sp=None,
                             addcoredensity=True, a=None,
                             D_sp_total=None,
                             dEdD_sp_tot=None,
                             spin=None):

    xcc = setup.xc_correction
    if xcc is None:
        return 0.0

    rgd = xcc.rgd
    nspins = len(D_sp)

    if addcoredensity:
        nc0_sg = rgd.empty(nspins)
        nct0_sg = rgd.empty(nspins)
        nc0_sg[:] = sqrt(4 * pi) / nspins * xcc.nc_g
        nct0_sg[:] = sqrt(4 * pi) / nspins * xcc.nct_g
        if xcc.nc_corehole_g is not None and nspins == 2:
            nc0_sg[0] -= 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
            nc0_sg[1] += 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
    else:
        nc0_sg = 0
        nct0_sg = 0

    D_sLq = np.inner(D_sp, xcc.B_pqL.T)
    D_sLq_total = np.inner(D_sp_total, xcc.B_pqL.T)

    e, dEdD_sqL, dEdD_scomL = expansion(rgd, D_sLq, xcc.n_qg,
                                        nc0_sg, D_sLq_total, spin)

    et, dEtdD_sqL, dEtdD_scomL = expansion(rgd, D_sLq, xcc.nt_qg,
                                           nct0_sg, D_sLq_total, spin)

    if dEdD_sp is not None:
        dEdD_sp += np.inner((dEdD_sqL - dEtdD_sqL).reshape((nspins, -1)),
                            xcc.B_pqL.reshape((len(xcc.B_pqL), -1)))
        dEdD_sp += np.inner((dEdD_sqL - dEtdD_sqL).reshape((nspins, -1)),
                            xcc.B_pqL.reshape((len(xcc.B_pqL), -1)))

    if dEdD_sp_tot is not None:
        dEdD_sp_tot += np.inner((dEdD_scomL - dEtdD_scomL).reshape((nspins, -1)),
                               xcc.B_pqL.reshape((len(xcc.B_pqL), -1)))
        dEdD_sp_tot += np.inner((dEdD_scomL - dEtdD_scomL).reshape((nspins, -1)),
                               xcc.B_pqL.reshape((len(xcc.B_pqL), -1)))

    return e - et


class SFRadialCalculator:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, rgd, n_sLg, Y_L, n_sLg_total, spin):
        nspins = len(n_sLg)
        n_sg = np.dot(Y_L, n_sLg)
        n_sg_total = np.dot(Y_L, n_sLg_total)
        e_g = rgd.empty()
        dedn_sg = rgd.zeros(nspins)
        dedn_stot = rgd.zeros(nspins)

        self.kernel.calculate(e_g, n_sg, dedn_sg,
                              n_stot=n_sg_total,
                              v_scom=dedn_stot,
                              spin=spin)

        return e_g, dedn_sg, dedn_stot


class SF(XCFunctional):

    def __init__(self, kernel):
        self.kernel = kernel
        XCFunctional.__init__(self, kernel.name, kernel.type)

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

    def calculate_impl(self, gd, n_sg, v_sg, e_g,
                       n_stot, v_scom, spin):

        self.kernel.calculate(e_g, n_sg, v_sg,
                              n_stot=n_stot, v_scom=v_scom,
                              spin=spin)

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None,
                                 D_sp_tot=None, dEdD_sp_tot=None,
                                 spin=None):

        from gpaw.xc.noncollinear import NonCollinearLDAKernel
        collinear = not isinstance(self.kernel, NonCollinearLDAKernel)
        assert collinear is True
        rcalc = SFRadialCalculator(self.kernel)
        expansion = SFRadialExpansion(rcalc, collinear)
        return calculate_paw_correction(expansion,
                                        setup, D_sp, dEdD_sp,
                                        addcoredensity, a,
                                        D_sp_total=D_sp_tot,
                                        dEdD_sp_tot=dEdD_sp_tot,
                                        spin=spin)

    def calculate_radial(self, rgd, n_sLg, Y_L,
                         n_sLg_total=None, spin=None):

        rcalc = SFRadialCalculator(self.kernel)

        return rcalc(rgd, n_sLg, Y_L,
                     n_sLg_total=n_sLg_total, spin=spin)

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None,
                            n_sg_total=None, v_scom=None, spin=None):

        if e_g is None:
            e_g = rgd.empty()
        rcalc = SFRadialCalculator(self.kernel)
        e_g[:], dedn_sg, dedn_stot = rcalc(
            rgd, n_sg[:, np.newaxis], [1.0],
            n_sLg_total=n_sg_total[:, np.newaxis], spin=spin)
        v_sg[:] = dedn_sg
        v_scom[:] += dedn_stot

        return rgd.integrate(e_g)

    def calculate_fxc(self, gd, n_sg, f_sg):

        raise NotImplementedError

        if gd is not self.gd:
            self.set_grid_descriptor(gd)

        assert len(n_sg) == 1
        assert n_sg.shape == f_sg.shape
        assert n_sg.flags.contiguous and n_sg.dtype == float
        assert f_sg.flags.contiguous and f_sg.dtype == float
        self.kernel.xc.calculate_fxc_spinpaired(n_sg.ravel(), f_sg)

    def stress_tensor_contribution(self, n_sg):

        raise NotImplementedError

        nspins = len(n_sg)
        v_sg = self.gd.zeros(nspins)
        e_g = self.gd.empty()
        self.calculate_impl(self.gd, n_sg, v_sg, e_g)
        stress = self.gd.integrate(e_g, global_integral=False)
        for v_g, n_g in zip(v_sg, n_sg):
            stress -= self.gd.integrate(v_g, n_g, global_integral=False)
        stress = self.gd.comm.sum(stress)
        return np.eye(3) * stress


class PurePythonSFKernel:

    def __init__(self, F, dFu, dFv):
        """
        scaling factor F(u, v),
        where
        u = n_i - orbital density,
        v = n_tot - total density

        :param F:
        :param dFu: derivative of F with respect to u
        :param dFv: derivative of F with respect to v

        """

        self.name = 'SF'
        self.type = 'LDA'
        self.F = F
        self.dFu = dFu
        self.dFv = dFv

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None,
                  n_stot=None, v_scom=None, spin=None):
        """
        :param e_g:
        :param n_sg: orbital spin-densities
        :param dedn_sg: orbital depended-potential
        :param sigma_xg:
        :param dedsigma_xg:
        :param tau_sg:
        :param dedtau_sg:
        :param n_tot: total density for a given spin
        :param v_com: common potential for a given spin
        :param spin: 0 or 1, spin index
        :return:
        """

        e_g[:] = 0.
        sf(e_g, n_sg[spin], dedn_sg[spin], n_stot[spin], v_scom[spin],
           self.F, self.dFu, self.dFv)

        if spin == 0:
            op_spin = 1
        else:
            op_spin = 0

        dedn_sg[op_spin] = np.zeros_like(dedn_sg[spin])
        v_scom[op_spin] = np.zeros_like(v_scom[spin])


REGULARIZATION = 1.0e-16


def sf(e, n_i, v_i, n_tot, v_com, F, dFu, dFv):
    """
    v_i = F(n_i, n_tot) + n_i dFu
    v_com = n_i dFv

    :param spin:
    :param e:
    :param n_i:
    :param v_i:
    :param n_tot:
    :param v_com:
    :param F:
    :param dFu:
    :param dFv:
    :return:
    """

    # for another spin channel,
    # potential equals zero

    eps = REGULARIZATION
    n_i[n_i < eps] = 1.0e-40
    n_tot[n_tot < eps] = 1.0e-40

    e[:] += F(n_i, n_tot)

    v_i += dFu(n_i, n_tot)

    v_com += dFv(n_i, n_tot)

    # eps = REGULARIZATION
    # e[:] += F(n_i + eps, n_tot + eps)
    #
    # v_i += dFu(n_i + eps, n_tot+ eps)
    #
    # v_com += dFv(n_i + eps, n_tot + eps)
