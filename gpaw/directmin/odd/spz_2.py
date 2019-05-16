"""
Potentials for orbital density dependent energy functionals
"""
import numpy as np
from gpaw.utilities import unpack
from gpaw.directmin.fd.tools import d_matrix
from gpaw.directmin.tools import get_n_occ
from gpaw.xc.scaling_factor_gga import SFG
from gpaw.xc.scaling_factor_gga_3 import PurePythonSFG3Kernel
from gpaw.directmin.odd.pz import PzCorrections


class SPzCorrections2(PzCorrections):

    """
    Perdew-Zunger self-interaction corrections scaled with
    a function F(n_i, n)

    """
    def __init__(self, wfs, dens, ham, scaling_factor=(1.0, 1.0),
                 sic_coarse_grid=True, store_potentials=False,
                 poisson_solver='FPS'):

        super(SPzCorrections2, self).__init__(
            wfs, dens, ham,
            scaling_factor=scaling_factor,
            sic_coarse_grid=sic_coarse_grid,
            store_potentials=store_potentials,
            poisson_solver=poisson_solver)

        self.name = 'SPZ_SIC2'

        self.sf_xc = SFG(PurePythonSFG3Kernel())

        self.v_scom = None
        self.dH_asp_com = None
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
            self.dH_asp_com[a][kpt.s][:] += \
                kpt.f_n[m] * e_pz.sum() * dH_ap_com[a]
        self.v_scom[kpt.s][:] += kpt.f_n[m] * e_pz.sum() * vt_G_com

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

        raise NotImplementedError

    def get_energy_and_gradients_inner_loop(self, wfs, kpt, a_mat,
                                            evals, evec, dens):

        raise NotImplementedError

    def get_odd_corrections_to_forces(self, F_av, wfs, kpt):

        raise NotImplementedError

    def get_energy_and_gradients(self, wfs, grad_knG,
                                 dens=None, U_k=None):

        wfs.timer.start('SIC e/g grid calculations')

        self.v_scom = self.cgd.zeros(wfs.nspins, dtype=self.dtype)
        grad_s = {}
        self.dH_asp_com = {}
        e_tot = 0.0

        for kpt in wfs.kpt_u:
            n_occ = get_n_occ(kpt)
            grad_s[kpt.s] = np.zeros_like(kpt.psit_nG[:n_occ])
            for a in dens.D_asp.keys():
                self.dH_asp_com[a] = np.zeros_like(dens.D_asp[a])
            e_total_sic = np.array([])
            sf = []
            for i in range(n_occ):
                # these values are scaled with i-th occupation numbers
                # but not e_sf
                v_m, dH_ap, e_sic, e_sf = self.get_pot_en_dh_and_sf_i_kpt(
                    wfs, kpt, dens, i)
                e_total_sic = np.append(e_total_sic, e_sic, axis=0)
                sf.append(e_sf)

                grad_s[kpt.s][i] = kpt.psit_nG[i] * v_m
                c_axi = {}
                for a in kpt.P_ani.keys():
                    dH_ii = unpack(dH_ap[a])
                    c_xi = np.dot(kpt.P_ani[a][i], dH_ii)
                    c_axi[a] = c_xi  # * kpt.f_n[i]
                # add projectors to
                wfs.pt.add(grad_s[kpt.s][i], c_axi, kpt.q)

                grad_s[kpt.s][i] = kpt.psit_nG[i] * v_m
                c_axi = {}
                for a in kpt.P_ani.keys():
                    dH_ii = unpack(dH_ap[a])
                    c_xi = np.dot(kpt.P_ani[a][i], dH_ii)
                    c_axi[a] = c_xi  # * kpt.f_n[i]
                # add projectors to
                wfs.pt.add(grad_s[kpt.s][i], c_axi, kpt.q)

            k = self.n_kps * kpt.s + kpt.q
            self.e_sic_by_orbitals[k] = \
                e_total_sic.reshape(e_total_sic.shape[0] // 2, 2)
            self.scalingf[k] = np.asarray(sf)
            wfs.kd.comm.sum(self.v_scom)
            for a in dens.D_asp.keys():
                wfs.kd.comm.sum(self.dH_asp_com[a])

            e_tot += e_total_sic.sum()

        # common potential:
        for kpt in wfs.kpt_u:
            n_occ = get_n_occ(kpt)
            k = self.n_kps * kpt.s + kpt.q
            for i in range(n_occ):
                grad_s[kpt.s][i] += \
                    kpt.psit_nG[i] * self.v_scom.sum(axis=0)
                c_axi = {}
                for a in kpt.P_ani.keys():
                    dH_ii = unpack(self.dH_asp_com[a].sum(axis=0))
                    c_xi = np.dot(kpt.P_ani[a][i], dH_ii)
                    c_axi[a] = c_xi  # * kpt.f_n[i]
                # add projectors to
                wfs.pt.add(grad_s[kpt.s][i], c_axi, kpt.q)

            if U_k is not None:
                grad_knG[k][:n_occ] += np.tensordot(
                    U_k[k].conj(), grad_s[kpt.s], axes=1)
            else:
                grad_knG[k][:n_occ] += grad_s[kpt.s]

        e_tot = wfs.kd.comm.sum(e_tot)
        wfs.timer.stop('SIC e/g grid calculations')

        return e_tot

    def get_energy_and_gradients_inner_loop2(
            self, wfs, a_mat_k, evals_k, evec_k, dens):

        grad_knG = {}
        for kpt in wfs.kpt_u:
            n_occ = get_n_occ(kpt)
            k = self.n_kps * kpt.s + kpt.q
            grad_knG[k] = np.zeros_like(kpt.psit_nG[:n_occ])

        e_sic = self.get_energy_and_gradients(
            wfs, grad_knG, dens, U_k=None)

        wfs.timer.start('Unitary gradients')
        g_mat_k = {}
        for kpt in wfs.kpt_u:
            n_occ = get_n_occ(kpt)
            if n_occ == 0:
                g_mat_k[k] = np.zeros_like(a_mat_k[k])
                continue

            k = self.n_kps * kpt.s + kpt.q
            l_odd = self.cgd.integrate(kpt.psit_nG[:n_occ],
                                       grad_knG[k][:n_occ], False)
            l_odd = np.ascontiguousarray(l_odd)
            self.cgd.comm.sum(l_odd)

            f = np.ones(n_occ)
            l_odd = f[:, np.newaxis] * l_odd.T.conj() - f * l_odd
            g_mat = evec_k[k].T.conj() @ l_odd.T.conj() @ evec_k[k]
            g_mat = g_mat * d_matrix(evals_k[k])
            g_mat = evec_k[k] @ g_mat @ evec_k[k].T.conj()
            for i in range(g_mat.shape[0]):
                g_mat[i][i] *= 0.5

            if a_mat_k[k].dtype == float:
                g_mat = g_mat.real
            g_mat_k[k] = g_mat * 2.0

        wfs.timer.stop('Unitary gradients')

        return e_sic, g_mat_k