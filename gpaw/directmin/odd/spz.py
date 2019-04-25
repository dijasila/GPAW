"""
Potentials for orbital density dependent energy functionals
"""
import numpy as np
from gpaw.utilities import pack, unpack
from gpaw.lfc import LFC
from gpaw.transformers import Transformer
from gpaw.directmin.fd.tools import d_matrix
from gpaw.poisson import PoissonSolver
from gpaw.directmin.tools import get_n_occ

# from gpaw.xc.scaling_factor import SF, PurePythonSFKernel
from gpaw.xc.scaling_factor_gga import SFG, PurePythonSFGKernel


class SPzCorrectionsLcao:
    """
    Perdew-Zunger self-interaction corrections scaled with
    a function F(n_i, n)

    """
    def __init__(self, wfs, dens, ham, scaling_factor=(1.0, 1.0),
                 sic_coarse_grid=True, store_potentials=False,
                 poisson_solver='FPS'):

        self.name = 'SPZ_SIC'
        # what we need from wfs
        self.setups = wfs.setups
        spos_ac = wfs.spos_ac
        self.cgd = wfs.gd

        # what we need from dens
        self.finegd = dens.finegd
        self.sic_coarse_grid = sic_coarse_grid

        if self.sic_coarse_grid:
            self.ghat = LFC(self.cgd,
                            [setup.ghat_l for setup
                             in self.setups],
                            integral=np.sqrt(4 * np.pi),
                            forces=True)
            self.ghat.set_positions(spos_ac)
        else:
            self.ghat = dens.ghat  # we usually solve poiss. on finegd

        # what we need from ham
        self.xc = ham.xc

        # initialize scaling function xc.
        # self.sf_xc = SF(PurePythonSFKernel(F, dFu, dFv))
        self.sf_xc = SFG(PurePythonSFGKernel())

        if poisson_solver == 'FPS':
            self.poiss = PoissonSolver(eps=1.0e-16,
                                       use_charge_center=True,
                                       use_charged_periodic_corrections=True)
        elif poisson_solver == 'GS':
            self.poiss = PoissonSolver(name='fd',
                                       relax=poisson_solver,
                                       eps=1.0e-16,
                                       use_charge_center=True,
                                       use_charged_periodic_corrections=True)

        if self.sic_coarse_grid is True:
            self.poiss.set_grid_descriptor(self.cgd)
        else:
            self.poiss.set_grid_descriptor(self.finegd)

        self.interpolator = Transformer(self.cgd, self.finegd, 3)
        self.restrictor = Transformer(self.finegd, self.cgd, 3)
        # self.timer = wfs.timer
        self.dtype = wfs.dtype
        self.eigv_s = {}
        self.lagr_diag_s = {}
        self.e_sic_by_orbitals = {}
        self.counter = 0  # number of calls of this class
        # Scaling factor:
        self.beta_c = scaling_factor[0]
        self.beta_x = scaling_factor[1]

        self.n_kps = wfs.kd.nks // wfs.kd.nspins

        self.store_potentials = store_potentials
        if store_potentials:
            self.old_pot = {}
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                n_occ = 0
                nbands = len(kpt.f_n)
                while n_occ < nbands and kpt.f_n[n_occ] > 1e-10:
                    n_occ += 1
                self.old_pot[k] = self.cgd.zeros(n_occ, dtype=float)

        self.v_com = None
        self.dH_ap_com = None
        self.scalingf = {}
        self.nspins = wfs.nspins

    def get_pot_en_dh_and_sf_i_kpt(self, wfs, kpt, dens, m):

        """

        :param wfs:
        :param kpt:
        :param dens:
        :param m:
        :return:

        To calculate this, we need to calculate
        orbital-depended potential and PAW corrections to it.
        """

        u = self.n_kps * kpt.s + kpt.q
        # get orbital-density

        # here we don't scale with occupation numbers
        wfs.timer.start('Construct Density, Comp. Charge, and DM')
        nt_G, Q_aL, D_ap = self.get_orbdens_compcharge_dm_kpt(kpt, m)
        wfs.timer.stop('Construct Density, Comp. Charge, and DM')

        # here we don't scale with occupation numbers
        e_pz, vt_pz_G, dH_pz_ap = \
            self.get_pz_sic_ith(nt_G, Q_aL, D_ap, m, u, wfs.timer)

        # here we don't scale with occupation numbers
        e_sf, vt_sf_G, vt_G_com, dH_sf_ap, dH_ap_com = \
            self.get_scaling_contribution(
                nt_G, dens, D_ap, kpt.s, wfs.timer)

        # now scale with occupation numbers
        vt_mG = (e_sf * vt_pz_G + e_pz.sum() * vt_sf_G) * kpt.f_n[m]
        e_sic_m = e_sf * e_pz * kpt.f_n[m]
        dH_ap = {}
        for a in dH_pz_ap.keys():
            dH_ap[a] = e_sf * dH_pz_ap[a] + e_pz.sum() * dH_sf_ap[a]
            dH_ap[a] *= kpt.f_n[m]
            self.dH_ap_com[a][:] += \
                kpt.f_n[m] * e_pz.sum() * dH_ap_com[a]
        self.v_com[:] += kpt.f_n[m] * e_pz.sum() * vt_G_com

        return vt_mG, dH_ap, e_sic_m , e_sf

    def get_orbdens_compcharge_dm_kpt(self, kpt, n):

        nt_G = np.absolute(kpt.psit_nG[n]**2)

        # paw
        Q_aL = {}
        D_ap = {}
        for a, P_ni in kpt.P_ani.items():
            P_i = P_ni[n]
            D_ii = np.outer(P_i, P_i.conj()).real
            D_ap[a] = D_p = pack(D_ii)
            Q_aL[a] = np.dot(D_p, self.setups[a].Delta_pL)

        return nt_G, Q_aL, D_ap

    def get_pseudo_pot(self, nt, Q_aL, i, kpoint=None):

        if self.sic_coarse_grid is False:
            # fine grid
            vt_sg = self.finegd.zeros(2)
            v_ht_g = self.finegd.zeros()
            nt_sg = self.finegd.zeros(2)
        else:
            # coarse grid
            vt_sg = self.cgd.zeros(2)
            v_ht_g = self.cgd.zeros()
            nt_sg = self.cgd.zeros(2)

        if self.sic_coarse_grid is False:
            self.interpolator.apply(nt, nt_sg[0])
            nt_sg[0] *= self.cgd.integrate(nt) / \
                        self.finegd.integrate(nt_sg[0])
            e_xc = self.xc.calculate(self.finegd, nt_sg, vt_sg)
        else:
            nt_sg[0] = nt
            e_xc = self.xc.calculate(self.cgd, nt_sg, vt_sg)

        vt_sg[0] *= -self.beta_x

        self.ghat.add(nt_sg[0], Q_aL)

        if self.store_potentials:
            if self.sic_coarse_grid:
                v_ht_g = self.old_pot[kpoint][i]
            else:
                self.interpolator.apply(self.old_pot[kpoint][i],
                                        v_ht_g)

        self.poiss.solve(v_ht_g, nt_sg[0],
                         zero_initial_phi=False)

        if self.store_potentials:
            if self.sic_coarse_grid is True:
                self.old_pot[kpoint][i] = v_ht_g.copy()
            else:
                self.restrictor.apply(v_ht_g, self.old_pot[kpoint][i])

        if self.sic_coarse_grid is False:
            ec = 0.5 * self.finegd.integrate(nt_sg[0] * v_ht_g)
        else:
            ec = 0.5 * self.cgd.integrate(nt_sg[0] * v_ht_g)

        vt_sg[0] -= v_ht_g * self.beta_c

        if self.sic_coarse_grid is False:
            vt_G = self.cgd.zeros()
            self.restrictor.apply(vt_sg[0], vt_G)
        else:
            vt_G = vt_sg[0]

        return np.array([-ec*self.beta_c, -e_xc*self.beta_x]), \
               vt_G, v_ht_g

    def get_paw_corrections(self, D_ap, vHt_g):

        # XC-PAW
        dH_ap = {}

        exc = 0.0
        for a, D_p in D_ap.items():
            setup = self.setups[a]

            dH_sp = np.zeros((2, len(D_p)))
            D_sp = np.array([D_p, np.zeros_like(D_p)])

            exc += self.xc.calculate_paw_correction(setup, D_sp,
                                                    dH_sp,
                                                    addcoredensity=False)

            dH_ap[a] = -dH_sp[0] * self.beta_x

        # Hartree-PAW
        ec = 0.0
        W_aL = self.ghat.dict()
        self.ghat.integrate(vHt_g, W_aL)

        for a, D_p in D_ap.items():
            setup = self.setups[a]
            M_p = np.dot(setup.M_pp, D_p)
            ec += np.dot(D_p, M_p)

            dH_ap[a] += -(2.0 * M_p + np.dot(setup.Delta_pL,
                                             W_aL[a])) * self.beta_c

        if self.sic_coarse_grid is False:
            ec = self.finegd.comm.sum(ec)
            exc = self.finegd.comm.sum(exc)
        else:
            ec = self.cgd.comm.sum(ec)
            exc = self.cgd.comm.sum(exc)

        return np.array([-ec*self.beta_c, -exc * self.beta_x]), dH_ap

    def get_pz_sic_ith(self, nt_G, Q_aL, D_ap, m, u, timer):

        # calculate sic energy,
        # sic pseudo-potential and Hartree
        timer.start('Get Pseudo Potential')
        # calculate sic energy, sic pseudo-potential and Hartree
        e_pz, vt_G, v_ht_g = \
            self.get_pseudo_pot(nt_G, Q_aL, m, kpoint=u)
        timer.stop('Get Pseudo Potential')

        # calculate PAW corrections
        timer.start('PAW')
        # calculate PAW corrections
        e_pz_paw_m, dH_ap = self.get_paw_corrections(D_ap, v_ht_g)
        timer.stop('PAW')

        # total sic:
        e_pz += e_pz_paw_m

        return e_pz, vt_G, dH_ap

    def get_scaling_contribution(self, nt_G, dens, D_ap, spin, timer):

        timer.start('Get Pseudo Potential')
        e_sf, vt_G, vt_G_com = self.get_scaling_ps_pot(nt_G, dens, spin, timer)
        timer.stop('Get Pseudo Potential')

        # calculate PAW corrections
        timer.start('PAW')
        e_sf_paw, dH_ap, dH_ap_com = \
            self.get_scaling_paw_corrections(dens, D_ap, spin, timer)
        timer.stop('PAW')

        # total sf:
        e_sf += e_sf_paw

        return e_sf, vt_G, vt_G_com, dH_ap, dH_ap_com

    def get_scaling_ps_pot(self, nt, dens, spin, timer):

        if self.sic_coarse_grid is False:
            # change to fine grid
            vt_sg = self.finegd.zeros(2)
            vt_sg_com = self.finegd.zeros(2)
            nt_sg = self.finegd.zeros(2)
        else:
            vt_sg = self.cgd.zeros(2)
            vt_sg_com = self.cgd.zeros(2)
            nt_sg = self.cgd.zeros(2)

        if self.sic_coarse_grid is False:
            self.interpolator.apply(nt, nt_sg[spin])
            nt_sg[spin] *= self.cgd.integrate(nt) / \
                        self.finegd.integrate(nt_sg[spin])
        else:
            nt_sg[spin] = nt

        timer.start('ODD XC 3D grid')
        if self.sic_coarse_grid is False:
            e_xc = self.sf_xc.calculate(
                self.finegd, nt_sg, vt_sg,
                n_stot=dens.nt_sg,
                v_scom=vt_sg_com, spin=spin)
        else:
            e_xc = self.sf_xc.calculate(
                self.cgd, nt_sg, vt_sg,
                n_stot=dens.nt_sG,
                v_scom=vt_sg_com, spin=spin)
        timer.stop('ODD XC 3D grid')

        if self.sic_coarse_grid is False:
            vt_G = self.cgd.zeros()
            vt_G_com = self.cgd.zeros()
            self.restrictor.apply(vt_sg[spin], vt_G)
            self.restrictor.apply(vt_sg_com[spin], vt_G_com)

        else:
            vt_G = vt_sg[spin]
            vt_G_com = vt_sg_com[spin]

        return e_xc, vt_G, vt_G_com

    def get_scaling_paw_corrections(self, dens, D_ap, spin, timer):
        # XC-PAW
        timer.start('xc-PAW')
        dH_ap = {}
        dH_ap_com = {}
        e_sf = 0.0
        for a, D_p in D_ap.items():
            setup = self.setups[a]
            dH_sp = np.zeros((2, len(D_p)))
            dH_sp_com = np.zeros_like(dH_sp)

            if spin == 0:
                D_sp = np.array([D_p, np.zeros_like(D_p)])
                D_t = np.array([
                    dens.D_asp[a][0] / (3.0 - self.nspins),
                    np.zeros_like(dens.D_asp[a][0])])
            if spin == 1:
                D_sp = np.array([np.zeros_like(D_p), D_p])
                D_t = np.array([
                    np.zeros_like(dens.D_asp[a][1]),
                    dens.D_asp[a][1] / (3.0 - self.nspins)])

            e_sf += self.sf_xc.calculate_paw_correction(setup, D_sp,
                                                        dH_sp,
                                                        addcoredensity=True,
                                                        a=a,
                                                        D_sp_tot=D_t,
                                                        dEdD_sp_tot=dH_sp_com,
                                                        spin=spin)
            dH_ap[a] = dH_sp[spin]
            dH_ap_com[a] = dH_sp_com[spin]

        timer.stop('xc-PAW')

        timer.start('Wait for sum')
        if self.sic_coarse_grid is False:
            e_sf = self.finegd.comm.sum(e_sf)
        else:
            e_sf = self.cgd.comm.sum(e_sf)
        timer.stop('Wait for sum')

        return e_sf, dH_ap, dH_ap_com

    def get_energy_and_gradients_kpt(self, wfs, kpt, grad_knG, dens):

        wfs.timer.start('SIC e/g grid calculations')
        k = self.n_kps * kpt.s + kpt.q
        n_occ = get_n_occ(kpt)

        self.v_com = self.cgd.zeros(dtype=self.dtype)
        self.dH_ap_com = {}
        for a in dens.D_asp.keys():
            p = dens.D_asp[a][kpt.s].shape[0]
            self.dH_ap_com[a] = np.zeros(shape=p, dtype=self.dtype)

        e_total_sic = np.array([])
        sf = []

        for i in range(n_occ):
            # this values are scaled with i-th occupation numbers
            # but not e_sf
            v_m, dH_ap, e_sic, e_sf = self.get_pot_en_dh_and_sf_i_kpt(
                wfs, kpt, dens, i)
            e_total_sic = np.append(e_total_sic, e_sic, axis=0)
            sf.append(e_sf)

            grad_knG[k][i] += kpt.psit_nG[i] * v_m
            c_axi = {}
            for a in kpt.P_ani.keys():
                dH_ii = unpack(dH_ap[a])
                c_xi = np.dot(kpt.P_ani[a][i], dH_ii)
                c_axi[a] = c_xi  # * kpt.f_n[i]
            # add projectors to
            wfs.pt.add(grad_knG[k][i], c_axi, kpt.q)

        # common potential:
        for i in range(n_occ):
            grad_knG[k][i] += kpt.psit_nG[i] * self.v_com
            c_axi = {}
            for a in kpt.P_ani.keys():
                dH_ii = unpack(self.dH_ap_com[a])
                c_xi = np.dot(kpt.P_ani[a][i], dH_ii)
                c_axi[a] = c_xi #* kpt.f_n[i]
            # add projectors to
            wfs.pt.add(grad_knG[k][i], c_axi, kpt.q)

        self.e_sic_by_orbitals[k] = \
            e_total_sic.reshape(e_total_sic.shape[0] // 2, 2)
        self.scalingf[k] = np.asarray(sf)

        wfs.timer.stop('SIC e/g grid calculations')
        return e_total_sic.sum()

    def get_energy_and_gradients_inner_loop(self, wfs, kpt, a_mat,
                                            evals, evec, dens):

        n_occ = 0
        for f in kpt.f_n:
            if f > 1.0e-10:
                n_occ += 1

        k = self.n_kps * kpt.s + kpt.q
        grad = {k: np.zeros_like(kpt.psit_nG[:n_occ])}

        e_sic = self.get_energy_and_gradients_kpt(wfs, kpt, grad, dens)

        wfs.timer.start('Unitary gradients')
        l_odd = self.cgd.integrate(kpt.psit_nG[:n_occ],
                                   grad[k][:n_occ], False)
        l_odd = np.ascontiguousarray(l_odd)
        self.cgd.comm.sum(l_odd)

        f = np.ones(n_occ)
        l_odd = f[:, np.newaxis] * l_odd.T.conj() - f * l_odd

        if a_mat is None:
            wfs.timer.stop('Unitary gradients')
            return l_odd.T, e_sic
        else:
            g_mat = evec.T.conj() @ l_odd.T.conj() @ evec
            g_mat = g_mat * d_matrix(evals)
            g_mat = evec @ g_mat @ evec.T.conj()

            for i in range(g_mat.shape[0]):
                g_mat[i][i] *= 0.5

            wfs.timer.stop('Unitary gradients')

            if a_mat.dtype == float:
                return 2.0 * g_mat.real, e_sic
            else:
                return 2.0 * g_mat, e_sic

    def get_odd_corrections_to_forces(self, F_av, wfs, kpt):

        raise NotImplementedError

        n_occ = get_n_occ(kpt)
        n_kps = self.n_kps

        dP_amiv = wfs.pt.dict(n_occ, derivative=True)
        wfs.pt.derivative(kpt.psit_nG[:n_occ], dP_amiv)
        k = n_kps * kpt.s + kpt.q
        for m in range(n_occ):
            # calculate Hartree pot, compans. charge and PAW corrects
            nt_G, Q_aL, D_ap = self.get_orbdens_compcharge_dm_kpt(kpt, m)
            e_sic, vt_G, v_ht_g = \
                self.get_pseudo_pot(nt_G, Q_aL, m, kpoint=k)
            e_sic_paw_m, dH_ap = \
                self.get_paw_corrections(D_ap, v_ht_g)

            # Force from compensation charges:
            dF_aLv = self.ghat.dict(derivative=True)
            self.ghat.derivative(v_ht_g, dF_aLv)
            for a, dF_Lv in dF_aLv.items():
                F_av[a] -= kpt.f_n[m] * self.beta_c * \
                    np.dot(Q_aL[a], dF_Lv)

            # Force from projectors
            for a, dP_miv in dP_amiv.items():
                dP_vi = dP_miv[m].T.conj()
                dH_ii = unpack(dH_ap[a])
                P_i = kpt.P_ani[a][m]
                F_v = np.dot(np.dot(dP_vi, dH_ii), P_i)
                F_av[a] += kpt.f_n[m] * 2.0 * F_v.real

    def get_energy_and_gradients_kpt_2(self, wfs, kpt, grad_knG,
                                       dens=None, U=None):

        wfs.timer.start('SIC e/g grid calculations')
        k = self.n_kps * kpt.s + kpt.q
        n_occ = get_n_occ(kpt)
        grad = np.zeros_like(kpt.psit_nG[:n_occ])
        self.v_com = self.cgd.zeros(dtype=self.dtype)
        self.dH_ap_com = {}
        for a in dens.D_asp.keys():
            p = dens.D_asp[a][kpt.s].shape[0]
            self.dH_ap_com[a] = np.zeros(shape=p, dtype=self.dtype)

        e_total_sic = np.array([])
        sf = []

        for i in range(n_occ):
            # this values are scaled with i-th occupation numbers
            # but not e_sf
            v_m, dH_ap, e_sic, e_sf = self.get_pot_en_dh_and_sf_i_kpt(
                wfs, kpt, dens, i)
            e_total_sic = np.append(e_total_sic, e_sic, axis=0)
            sf.append(e_sf)

            grad[i] = kpt.psit_nG[i] * v_m
            c_axi = {}
            for a in kpt.P_ani.keys():
                dH_ii = unpack(dH_ap[a])
                c_xi = np.dot(kpt.P_ani[a][i], dH_ii)
                c_axi[a] = c_xi  # * kpt.f_n[i]
            # add projectors to
            wfs.pt.add(grad[i], c_axi, kpt.q)

        # common potential:
        for i in range(n_occ):
            grad[i] += kpt.psit_nG[i] * self.v_com
            c_axi = {}
            for a in kpt.P_ani.keys():
                dH_ii = unpack(self.dH_ap_com[a])
                c_xi = np.dot(kpt.P_ani[a][i], dH_ii)
                c_axi[a] = c_xi  # * kpt.f_n[i]
            # add projectors to
            wfs.pt.add(grad[i], c_axi, kpt.q)

        grad_knG[k][:n_occ] += np.tensordot(U.conj(), grad, axes=1)

        self.e_sic_by_orbitals[k] = \
            e_total_sic.reshape(e_total_sic.shape[0] // 2, 2)
        self.scalingf[k] = np.asarray(sf)

        wfs.timer.stop('SIC e/g grid calculations')
        return e_total_sic.sum()


# def F(n, n_tot):
#
#     x = n / n_tot
#     x[x > 1.0] = 1.0
#     x[x < 0.0] = 0.0
#
#     return (3.0 * x**2.0 - 2.0 * x**3.0) * n
#
#
# def dFu(n, n_tot):
#
#     x = n / n_tot
#     x[x > 1.0] = 1.0
#     x[x < 0.0] = 0.0
#
#     return 9.0 * x**3.0 - 8.0 * x**2.0
#
#
# def dFv(n, n_tot):
#
#     x = n / n_tot
#     x[x > 1.0] = 1.0
#     x[x < 0.0] = 0.0
#
#     return - 6.0 * (x - x**2.0) * x**2.0

#

def F(n, n_tot):
    x = n / n_tot
    x[x > 1.0] = 1.0
    x[x < 0.0] = 0.0

    return x * n


def dFu(n, n_tot):
    x = n / n_tot
    x[x > 1.0] = 1.0
    x[x < 0.0] = 0.0

    return 2.0 * x


def dFv(n, n_tot):
    x = n / n_tot
    x[x > 1.0] = 1.0
    x[x < 0.0] = 0.0

    return - x**2.0
