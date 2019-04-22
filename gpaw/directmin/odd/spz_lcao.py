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
        self.scalingf = {}
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

    def get_gradients(self, h_mm, c_nm, f_n, evec, evals, kpt,
                      wfs, timer, matrix_exp, repr_name,
                      ind_up, occupied_only=False, dens=None):
        """
        :param C_nM: coefficients of orbitals
        :return: matrix G - gradients, and orbital SI energies

        which is G_{ij} = (1 - delta_{ij}/2)*( \int_0^1 e^{tA} L e^{-tA} dt )_{ji}

        Lambda_ij = (C_i, F_j C_j )

        L_{ij} = Lambda_ji^{cc} - Lambda_ij

        """

        u = kpt.s * self.n_kps + kpt.q
        n_occ = 0
        nbands = len(f_n)
        while n_occ < nbands and f_n[n_occ] > 1e-10:
            n_occ += 1

        self.v_com = self.cgd.zeros(dtype=self.dtype)
        self.dH_ap_com = {}
        for a in dens.D_asp.keys():
            p = dens.D_asp[a][kpt.s].shape[0]
            self.dH_ap_com[a] = np.zeros(shape=p, dtype=self.dtype)

        if occupied_only is True:
            nbs = n_occ
        else:
            nbs = c_nm.shape[0]
        n_set = c_nm.shape[1]

        timer.start('Construct Gradient Matrix')
        hc_mn = np.dot(h_mm.conj(), c_nm[:nbs].T)
        h_mm = np.dot(c_nm[:nbs].conj(), hc_mn)
        # odd part
        b_mn = np.zeros(shape=(n_set, nbs), dtype=self.dtype)
        e_total_sic = np.array([])
        sf = []
        for n in range(n_occ):
            F_MM, sic_energy_n, e_sf =\
                self.get_orbital_potential_matrix(f_n, c_nm, kpt,
                                                  wfs, wfs.setups,
                                                  n, timer, dens)

            b_mn[:, n] = np.dot(c_nm[n], F_MM.conj()).T
            e_total_sic = np.append(e_total_sic, sic_energy_n, axis=0)
            sf.append(e_sf)

        # common contribution to all orbitals
        F_MM = self.potential_matrix(wfs, self.v_com, self.dH_ap_com,
                                     kpt, timer)
        # for n in range(n_occ):
        #     b_mn[:, n] += np.dot(c_nm[n], F_MM.conj()).T

        b_mn[:,:n_occ] += np.dot(F_MM.conj().T, c_nm[:n_occ].T)

        l_odd = np.dot(c_nm[:nbs].conj(), b_mn)


        f = f_n[:nbs]
        grad = f * (h_mm[:nbs, :nbs] + l_odd) - \
            f[:, np.newaxis] * (h_mm[:nbs, :nbs] + l_odd.T.conj())

        if matrix_exp in ['pade_approx', 'egdecomp2']:
            # timer.start('Frechet derivative')
            # frechet derivative, unfortunately it calculates unitary
            # matrix which we already calculated before. Could it be used?
            # it also requires a lot of memory so don't use it now
            # u, grad = expm_frechet(a_mat, h_mm,
            #                        compute_expm=True,
            #                        check_finite=False)
            # grad = grad @ u.T.conj()
            # timer.stop('Frechet derivative')
            grad = np.ascontiguousarray(grad)
        elif matrix_exp == 'egdecomp':
            timer.start('Use Eigendecomposition')
            grad = np.dot(evec.T.conj(), np.dot(grad, evec))
            grad = grad * D_matrix(evals)
            grad = np.dot(evec, np.dot(grad, evec.T.conj()))
            for i in range(grad.shape[0]):
                grad[i][i] *= 0.5
            timer.stop('Use Eigendecomposition')
        else:
            raise NotImplementedError('Check the keyword '
                                      'for matrix_exp. \n'
                                      'Must be '
                                      '\'pade_approx\' or '
                                      '\'egdecomp\'')
        if self.dtype == float:
            grad = grad.real
        if repr_name in ['sparse', 'u_invar']:
            grad = grad[ind_up]

        timer.stop('Construct Gradient Matrix')

        u = kpt.s * self.n_kps + kpt.q
        self.e_sic_by_orbitals[u] = \
            e_total_sic.reshape(e_total_sic.shape[0] // 2, 2)
        self.scalingf[u] = np.asarray(sf)

        timer.start('Residual')
        hc_mn += b_mn
        h_mm += l_odd
        # what if there are empty states between occupied?
        rhs = np.zeros(shape=(c_nm.shape[1], n_occ),
                       dtype=self.dtype)
        rhs2 = np.zeros(shape=(c_nm.shape[1], n_occ),
                        dtype=self.dtype)
        mmm(1.0, kpt.S_MM.conj(), 'N', c_nm[:n_occ], 'T', 0.0, rhs)
        mmm(1.0, rhs, 'N', h_mm[:n_occ, :n_occ], 'N', 0.0, rhs2)
        hc_mn = hc_mn[:, :n_occ] - rhs2[:, :n_occ]
        norm = []
        for i in range(n_occ):
            norm.append(np.dot(hc_mn[:, i].conj(),
                               hc_mn[:, i]).real * kpt.f_n[i])

        error = sum(norm) * Hartree ** 2 / wfs.nvalence
        del rhs, rhs2, hc_mn, norm
        timer.stop('Residual')

        if self.counter == 0:
            h_mm = 0.5 * (h_mm + h_mm.T.conj())
            kpt.eps_n[:nbs] = np.linalg.eigh(h_mm)[0]
        self.counter += 1

        return 2.0 * grad, error

    def get_orbital_potential_matrix(self, f_n, C_nM, kpt,
                                     wfs, setup, m, timer, dens=None):
        """
        :param f_n:
        :param C_nM:
        :param kpt:
        :param wfs:
        :param setup:
        :return:

        To calculate this, we need to calculate
        orbital-depended potential and PAW corrections to it.
        """

        kpoint = self.n_kps * kpt.s + kpt.q
        # get orbital-density
        timer.start('Construct Density, Comp. Charge, and DM')
        nt_G, Q_aL, D_ap = \
            self.get_density(f_n, C_nM, kpt, wfs, setup, m)
        timer.stop('Construct Density, Comp. Charge, and DM')

        e_pz, vt_pz_G, dH_pz_ap = \
            self.get_pz_sic_ith(nt_G, Q_aL, D_ap, m, kpoint, timer)

        e_sf, vt_sf_G, vt_G_com, dH_sf_ap, dH_ap_com = \
            self.get_scaling_contribution(nt_G, dens, D_ap, kpt.s, timer)

        vt_mG = e_sf * vt_pz_G + e_pz.sum() * vt_sf_G
        e_sic_m = e_sf * e_pz
        dH_ap = {}
        for a in dH_pz_ap.keys():
            dH_ap[a] = e_sf * dH_pz_ap[a] + e_pz.sum() * dH_sf_ap[a]
            self.dH_ap_com[a][:] += e_pz.sum() * dH_ap_com[a]
        self.v_com[:] += e_pz.sum() * vt_G_com
        F_MM = self.potential_matrix(wfs, vt_mG, dH_ap, kpt, timer)

        return F_MM, e_sic_m * f_n[m], e_sf

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

    def get_pz_sic_ith(self, nt_G, Q_aL, D_ap, m, kpoint, timer):

        # calculate sic energy,
        # sic pseudo-potential and Hartree
        timer.start('Get Pseudo Potential')
        e_pz, vt_G, vHt_g = \
            self.get_pseudo_pot(nt_G, Q_aL, m, kpoint, timer)
        timer.stop('Get Pseudo Potential')

        # calculate PAW corrections
        timer.start('PAW')
        e_pz_paw, dH_ap = \
            self.get_paw_corrections(D_ap, vHt_g, timer)
        timer.stop('PAW')

        # total sic:
        e_pz += e_pz_paw

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
            e_xc = self.sf_xc.calculate(self.finegd, nt_sg, vt_sg,
                                        n_stot=dens.nt_sg,
                                        v_scom=vt_sg_com,
                                        spin=spin)
        else:
            e_xc = self.sf_xc.calculate(self.cgd, nt_sg, vt_sg,
                                        n_stot=dens.nt_sG,
                                        v_scom=vt_sg_com,
                                        spin=spin)
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
                D_t = np.array([dens.D_asp[a][0],
                                np.zeros_like(dens.D_asp[a][0])])
            if spin == 1:
                D_sp = np.array([np.zeros_like(D_p), D_p])
                D_t = np.array([np.zeros_like(dens.D_asp[a][1]),
                                dens.D_asp[a][1]])

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
