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
from gpaw.xc.scaling_factor_gga_2 import PurePythonSFG2Kernel
from gpaw.directmin.odd.pz import PzCorrections


class SPzCorrections(PzCorrections):
    """
    Perdew-Zunger self-interaction corrections scaled with
    a function F(n_i, n)

    """
    def __init__(self, wfs, dens, ham, scaling_factor=(1.0, 1.0),
                 sic_coarse_grid=True, store_potentials=False,
                 poisson_solver='FPS', sftype='I'):

        super(SPzCorrections, self).__init__(
            wfs, dens, ham,
            scaling_factor=scaling_factor,
            sic_coarse_grid=sic_coarse_grid,
            store_potentials=store_potentials,
            poisson_solver=poisson_solver)

        self.name = 'SPZ_SIC'

        if sftype == 'I':
            self.sf_xc = SFG(PurePythonSFGKernel())
        elif sftype == 'II':
            self.sf_xc = SFG(PurePythonSFG2Kernel())

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
             self.get_pz_sic_ith_kpt(nt_G, Q_aL, D_ap, m, u, wfs.timer)

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

    def get_energy_and_gradients_kpt(self, wfs, kpt, grad_knG,
                                     dens=None, U_k=None):

        wfs.timer.start('SIC e/g grid calculations')
        k = self.n_kps * kpt.s + kpt.q
        n_occ = get_n_occ(kpt)
        e_total_sic = np.array([])
        grad = np.zeros_like(kpt.psit_nG[:n_occ])

        self.v_com = self.cgd.zeros(dtype=self.dtype)
        self.dH_ap_com = {}
        for a in dens.D_asp.keys():
            p = dens.D_asp[a][kpt.s].shape[0]
            self.dH_ap_com[a] = np.zeros(shape=p, dtype=self.dtype)
        sf = []

        for i in range(n_occ):
            # this values are scaled with i-th occupation numbers
            # but not e_sf
            v_m, dH_ap, e_sic, e_sf = self.get_pot_en_dh_and_sf_i_kpt(
                wfs, kpt, dens, i)
            e_total_sic = np.append(e_total_sic, e_sic, axis=0)
            sf.append(e_sf)

            grad[i] += kpt.psit_nG[i] * v_m
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
                c_axi[a] = c_xi #* kpt.f_n[i]
            # add projectors to
            wfs.pt.add(grad[i], c_axi, kpt.q)

        if U_k is not None:
            grad_knG[k][:n_occ] += \
                np.tensordot(U_k[k].conj(), grad, axes=1)
        else:
            grad_knG[k][:n_occ] += grad

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
