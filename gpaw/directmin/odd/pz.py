"""
Potentials for orbital density dependent energy functionals
"""
from ase.units import Hartree
import numpy as np
from gpaw.utilities import pack, unpack
from gpaw.utilities.blas import gemm, gemv, gemmdot, mmm
from gpaw.utilities.lapack import diagonalize
from gpaw.lfc import LFC
from gpaw.transformers import Transformer
from gpaw.poisson import PoissonSolver
from gpaw.directmin.tools import D_matrix


class PzCorrectionsLcao:
    """
    Perdew-Zunger self-interaction corrections

    """
    def __init__(self, wfs, dens, ham, scaling_factor=(1.0, 1.0),
                 sic_coarse_grid=True, store_potentials=True,
                 poisson_solver='GS'):

        self.name = 'PZ_SIC'
        # what we need from wfs
        self.setups = wfs.setups
        self.bfs = wfs.basis_functions
        spos_ac = wfs.spos_ac

        # what we need from dens
        self.cgd = dens.gd
        self.finegd = dens.finegd
        self.sic_coarse_grid = sic_coarse_grid

        if self.sic_coarse_grid:
            self.ghat_cg = LFC(self.cgd,
                               [setup.ghat_l for setup
                                in self.setups],
                               integral=np.sqrt(4 * np.pi),
                               forces=True)
            self.ghat_cg.set_positions(spos_ac)
            self.ghat = None
        else:
            self.ghat = dens.ghat  # we usually solve poiss. on finegd
            self.ghat_cg = None

        # what we need from ham
        self.xc = ham.xc

        self.poiss = PoissonSolver(relax=poisson_solver,
                                   use_charge_center=True,
                                   eps=1.0e-16) #,
                                   # sic_gg=True)
        if self.sic_coarse_grid is True:
            self.poiss.set_grid_descriptor(self.cgd)
        else:
            self.poiss.set_grid_descriptor(self.finegd)

        self.interpolator = Transformer(self.cgd, self.finegd, 3)
        self.restrictor = Transformer(self.finegd, self.cgd, 3)
        self.timer = wfs.timer
        self.dtype = wfs.dtype
        self.eigv_s = {}
        self.lagr_diag_s = {}
        self.e_sic = {}
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

    def get_gradients(self, h_mm, c_nm, f_n, evec, evals, kpt,
                      wfs, timer, matrix_exp, sparse,
                      ind_up, occupied_only=False):
        """
        :param C_nM: coefficients of orbitals
        :return: matrix G - gradients, and orbital SI energies

        which is G_{ij} = (1 - delta_{ij}/2)*( \int_0^1 e^{tA} L e^{-tA} dt )_{ji}

        Lambda_ij = (C_i, F_j C_j )

        L_{ij} = Lambda_ji^{cc} - Lambda_ij

        """

        # 0.
        n_occ = 0
        nbands = len(f_n)
        while n_occ < nbands and f_n[n_occ] > 1e-10:
            n_occ += 1

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
        for n in range(n_occ):
            F_MM, sic_energy_n =\
                self.get_orbital_potential_matrix(f_n, c_nm, kpt,
                                                  wfs, wfs.setups, n)

            b_mn[:,n] = np.dot(c_nm[n], F_MM.conj()).T
            e_total_sic = np.append(e_total_sic, sic_energy_n, axis=0)
        l_odd = np.dot(c_nm[:nbs].conj(), b_mn)

        f = f_n[:nbs]
        grad = f * (h_mm[:nbs, :nbs] + l_odd) - \
            f[:, np.newaxis] * (h_mm[:nbs, :nbs] + l_odd.T.conj())

        if matrix_exp == 'pade_approx':
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
        elif matrix_exp == 'eigendecomposition':
            timer.start('Use Eigendecomposition')
            grad = evec.T.conj() @ grad @ evec
            grad = grad * D_matrix(evals)
            grad = evec @ grad @ evec.T.conj()
            for i in range(grad.shape[0]):
                grad[i][i] *= 0.5
            timer.stop('Use Eigendecomposition')
        else:
            raise NotImplementedError('Check the keyword '
                                      'for matrix_exp. \n'
                                      'Must be '
                                      '\'pade_approx\' or '
                                      '\'eigendecomposition\'')
        if self.dtype == float:
            grad = grad.real
        if sparse:
            grad = grad[ind_up]

        timer.stop('Construct Gradient Matrix')

        u = kpt.s * self.n_kps + kpt.q
        self.e_sic[u] = \
            e_total_sic.reshape(e_total_sic.shape[0] // 2, 2)

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
                                     wfs, setup, m):
        """
        :param f_n:
        :param C_nM:
        :param kpt:
        :param wfs:
        :param setup:
        :return: F_i = <m|v_i + PAW_i|n > for occupied
                 F_i = 0 for unoccupied,
                 SI energies

        To calculate this, we need to calculate
        orbital-depended potential and PAW corrections to it.

        Fot this, we need firstly to get orbitals densities.

        """
        kpoint = self.n_kps * kpt.s + kpt.q
        n_set = C_nM.shape[1]
        F_MM = np.zeros(shape=(n_set, n_set),
                        dtype=self.dtype)
        # get orbital-density
        nt_G, Q_aL, D_ap = \
            self.get_density(f_n,
                             C_nM, kpt,
                             wfs, setup, m)
        # calculate sic energy,
        # sic pseudo-potential and Hartree
        e_sic_m, vt_mG, vHt_g = \
            self.get_pseudo_pot(nt_G, Q_aL, m, kpoint)

        # calculate PAW corrections
        e_sic_paw_m, dH_ap = \
            self.get_paw_corrections(D_ap, vHt_g)

        # total sic:
        e_sic_m += e_sic_paw_m

        # now calculate potential matrix F_i
        # calculate pseudo-part
        # Vt_MM = \
        #     self.bfs.calculate_potential_matrices(vt_mG)[0]

        # TODO: sum over cell? see calculate_hamiltonian_matrix in
        # eigensolver.py
        Vt_MM = np.zeros_like(F_MM)
        self.bfs.calculate_potential_matrix(vt_mG, Vt_MM, kpt.q)
        # make matrix hermitian
        ind_l = np.tril_indices(Vt_MM.shape[0], -1)
        Vt_MM[(ind_l[1], ind_l[0])] = Vt_MM[ind_l].conj()

        # np.save('Vt_MM1', Vt_MM)
        #
        # wfs.timer.start('Potential matrix')
        # Vt_xMM = self.bfs.calculate_potential_matrices(vt_mG)
        # wfs.timer.stop('Potential matrix')
        #
        # if self.bfs.gamma and self.dtype is float:
        #     yy = 1.0
        #     Vt_MM = Vt_xMM[0]
        # else:
        #     wfs.timer.start('Sum over cells')
        #     yy = 0.5
        #     k_c = wfs.kd.ibzk_qc[kpt.q]
        #     Vt_MM = (0.5 + 0.0j) * Vt_xMM[0]
        #     for sdisp_c, Vt1_MM in zip(self.bfs.sdisp_xc[1:], Vt_xMM[1:]):
        #         Vt_MM += np.exp(2j * np.pi * np.dot(sdisp_c, k_c)) * Vt1_MM
        #     wfs.timer.stop('Sum over cells')
        #
        # # make matrix hermitian
        # ind_l = np.tril_indices(Vt_MM.shape[0], -1)
        # Vt_MM[(ind_l[1], ind_l[0])] = Vt_MM[ind_l].conj()
        #
        # np.save('Vt_MM2', Vt_MM)

        # PAW:
        for a, dH_p in dH_ap.items():
            P_Mj = wfs.P_aqMi[a][kpt.q]
            dH_ij = unpack(dH_p)
            # dH_ij = yy * unpack(dH_p)

            K_iM = np.zeros((dH_ij.shape[0], n_set),
                            dtype=self.dtype)

            if self.dtype is complex:
                gemm(1.0, P_Mj,
                     dH_ij.astype(complex),
                     0.0, K_iM, 'c')
                gemm(1.0, K_iM,
                     P_Mj,
                     1.0, F_MM)

                # K_iM = np.dot(dH_ij, P_Mj.T.conj())
                # F_MM += np.dot(P_Mj, K_iM)

            else:
                gemm(1.0, P_Mj, dH_ij, 0.0, K_iM, 't')
                gemm(1.0, K_iM, P_Mj, 1.0, F_MM)

                # K_iM = np.dot(dH_ij, P_Mj.T)
                # F_MM += np.dot(P_Mj, K_iM)

        if self.dtype is complex:
            F_MM += Vt_MM.astype(complex)
        else:
            F_MM += Vt_MM
        #
        # wfs.timer.start('Distribute overlap matrix')
        # F_MM = wfs.ksl.distribute_overlap_matrix(
        #     F_MM, root=-1, add_hermitian_conjugate=(yy == 0.5))
        # wfs.timer.stop('Distribute overlap matrix')
        #
        if self.sic_coarse_grid:
            self.cgd.comm.sum(F_MM)
        else:
            self.finegd.comm.sum(F_MM)

        return F_MM, e_sic_m

    def get_density(self, f_n, C_nM, kpt,
                    wfs, setup, m):

        # construct orbital density matrix
        if self.dtype is complex:
            rho_MM = f_n[m] * np.outer(C_nM[m].conj(), C_nM[m]) / \
                    (3 - wfs.nspins)
        else:
            rho_MM = f_n[m] * np.outer(C_nM[m], C_nM[m]) / \
                    (3 - wfs.nspins)

        nt_G = self.cgd.zeros()
        self.bfs.construct_density(rho_MM, nt_G, kpt.q)

        # calculate  atomic density matrix and
        # compensation charges
        D_ap = {}
        Q_aL = {}

        for a in wfs.P_aqMi.keys():
            P_Mi = wfs.P_aqMi[a][kpt.q]
            rhoP_Mi = np.zeros_like(P_Mi)

            D_ii = np.zeros((wfs.P_aqMi[a].shape[2],
                             wfs.P_aqMi[a].shape[2]),
                            dtype=self.dtype)

            gemm(1.0, P_Mi, rho_MM, 0.0, rhoP_Mi)
            if self.dtype is complex:
                gemm(1.0, rhoP_Mi, P_Mi.T.conj().copy(), 0.0, D_ii)
            else:
                gemm(1.0, rhoP_Mi, P_Mi.T.copy(), 0.0, D_ii)

            # FIXME: What to do with complex part, which are not zero
            if self.dtype is complex:
                D_ap[a] = D_p = pack(D_ii.real)
            else:
                D_ap[a] = D_p = pack(D_ii)

            Q_aL[a] = np.dot(D_p, setup[a].Delta_pL)

        return nt_G, Q_aL, D_ap

    def get_pseudo_pot(self, nt, Q_aL, i, kpoint):

        if self.sic_coarse_grid is False:
            # change to fine grid
            vt_sg = self.finegd.zeros(2)
            vHt_g = self.finegd.zeros()
            nt_sg = self.finegd.zeros(2)
        else:
            vt_sg = self.cgd.zeros(2)
            vHt_g = self.cgd.zeros()
            nt_sg = self.cgd.zeros(2)

        if self.sic_coarse_grid is False:
            self.interpolator.apply(nt, nt_sg[0])
            nt_sg[0] *= self.cgd.integrate(nt) / \
                        self.finegd.integrate(nt_sg[0])
        else:
            nt_sg[0] = nt

        self.timer.start('ODD XC 3D grid')
        if self.sic_coarse_grid is False:
            e_xc = self.xc.calculate(self.finegd, nt_sg, vt_sg)
        else:
            e_xc = self.xc.calculate(self.cgd, nt_sg, vt_sg)
        self.timer.stop('ODD XC 3D grid')
        vt_sg[0] *= -self.beta_x

        # Hartree
        if self.sic_coarse_grid is False:
            self.ghat.add(nt_sg[0], Q_aL)
        else:
            self.ghat_cg.add(nt_sg[0], Q_aL)

        self.timer.start('ODD Poisson')
        if self.store_potentials:
            if self.sic_coarse_grid:
                vHt_g = self.old_pot[kpoint][i]
            else:
                self.interpolator.apply(self.old_pot[kpoint][i],
                                        vHt_g)
        self.poiss.solve(vHt_g, nt_sg[0],
                         zero_initial_phi=False)
        if self.store_potentials:
            if self.sic_coarse_grid:
                self.old_pot[kpoint][i] = vHt_g.copy()
            else:
                self.restrictor.apply(vHt_g, self.old_pot[kpoint][i])

        self.timer.stop('ODD Poisson')

        self.timer.start('ODD Hartree integrate ODD')
        if self.sic_coarse_grid is False:
            ec = 0.5 * self.finegd.integrate(nt_sg[0] * vHt_g)
        else:
            ec = 0.5 * self.cgd.integrate(nt_sg[0] * vHt_g)

        self.timer.stop('ODD Hartree integrate ODD')
        vt_sg[0] -= vHt_g * self.beta_c
        if self.sic_coarse_grid is False:
            vt_G = self.cgd.zeros()
            self.restrictor.apply(vt_sg[0], vt_G)
        else:
            vt_G = vt_sg[0]

        return np.array([-ec*self.beta_c,
                         -e_xc*self.beta_x]),\
               vt_G, vHt_g

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
        if self.sic_coarse_grid is False:
            W_aL = self.ghat.dict()
            self.ghat.integrate(vHt_g, W_aL)
        else:
            W_aL = self.ghat_cg.dict()
            self.ghat_cg.integrate(vHt_g, W_aL)

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

    def update_eigenval(self, f_n, C_nM, kpt, wfs, setup, H_MM):
        n_kps = wfs.kd.nks // wfs.kd.nspins
        u = kpt.s * n_kps + kpt.q
        n_occ = 0
        for f in f_n:
            if f > 1.0e-10:
                n_occ += 1

        b_nM = np.zeros(shape=(n_occ, C_nM.shape[1]), dtype=self.dtype)

        for n in range(n_occ):
            F_MM = self.get_orbital_potential_matrix(f_n, C_nM, kpt,
                                                     wfs, setup, n,
                                                     )[0]
            gemv(1.0, F_MM, C_nM[n], 0.0, b_nM[n])

        L_occ = np.zeros((n_occ, n_occ), dtype=self.dtype)
        C_conj_nM = np.copy(C_nM.conj()[:n_occ])
        mmm(1.0, C_conj_nM, 'n', b_nM, 't', 0.0, L_occ)

        L_occ += C_conj_nM @ H_MM @ C_nM[:n_occ].T
        L_occ = 0.5 * (L_occ + L_occ.T.conj())
        del C_conj_nM

        L_unocc = C_nM.conj()[n_occ:] @ H_MM @ C_nM[n_occ:].T
        L_unocc = 0.5 * (L_unocc + L_unocc.T.conj())

        self.lagr_diag_s[u] = \
            np.append(np.diagonal(L_occ),
                      np.diagonal(L_unocc)).real

        # occupied eigenvalues
        # TODO: fix it, when there is no occ numbers
        if n_occ > 0:
            eig_occ = np.empty(L_occ.shape[0])
            diagonalize(L_occ, eig_occ)
            kpt.eps_n[:n_occ] = eig_occ

        # unoccupied eigenvalues
        if L_unocc.shape[0] > 0:
            eig_unocc = np.empty(L_unocc.shape[0])
            diagonalize(L_unocc, eig_unocc)
            kpt.eps_n[n_occ:] = eig_unocc

        self.eigv_s[u] = np.copy(kpt.eps_n)

    def get_odd_corrections_to_forces(self, wfs, dens):

        self.timer.start('LCAO forces')

        natoms = len(wfs.setups)
        F_av = np.zeros((natoms, 3))
        Ftheta_av = np.zeros_like(F_av)
        Frho_av = np.zeros_like(F_av)
        Fatom_av = np.zeros_like(F_av)
        Fpot_av = np.zeros_like(F_av)
        Fhart_av = np.zeros_like(F_av)

        # spos_ac = wfs.tci.atoms.get_scaled_positions() % 1.0

        spos_ac = wfs.spos_ac

        ksl = wfs.ksl
        nao = ksl.nao
        mynao = ksl.mynao
        nq = len(wfs.kd.ibzk_qc)
        dtype = wfs.dtype
        # tci = wfs.tci
        # newtci = wfs.newtci
        manytci = wfs.manytci
        gd = wfs.gd
        # bfs = wfs.basis_functions

        Mstart = ksl.Mstart
        Mstop = ksl.Mstop
        n_kps = wfs.kd.nks // wfs.kd.nspins

        # self.timer.start('TCI derivative')
        # dThetadR_qvMM = np.empty((nq, 3, mynao, nao), dtype)
        # dTdR_qvMM = np.empty((nq, 3, mynao, nao), dtype)
        # dPdR_aqvMi = {}
        # for a in self.bfs.my_atom_indices:
        #     ni = self.setups[a].ni
        #     dPdR_aqvMi[a] = np.empty((nq, 3, nao, ni), dtype)
        # tci.calculate_derivative(spos_ac, dThetadR_qvMM, dTdR_qvMM,
        #                          dPdR_aqvMi)
        # gd.comm.sum(dThetadR_qvMM)
        # gd.comm.sum(dTdR_qvMM)
        # self.timer.stop('TCI derivative')

        self.timer.start('TCI derivative')
        dThetadR_qvMM, dTdR_qvMM = manytci.O_qMM_T_qMM(
            gd.comm, Mstart, Mstop, False, derivative=True)

        dPdR_aqvMi = manytci.P_aqMi(
            self.bfs.my_atom_indices, derivative=True)
        gd.comm.sum(dThetadR_qvMM)
        gd.comm.sum(dTdR_qvMM)
        self.timer.stop('TCI derivative')

        my_atom_indices = self.bfs.my_atom_indices
        atom_indices = self.bfs.atom_indices

        def _slices(indices):
            for a in indices:
                M1 = self.bfs.M_a[a] - Mstart
                M2 = M1 + self.setups[a].nao
                if M2 > 0:
                    yield a, max(0, M1), M2

        def slices():
            return _slices(atom_indices)

        def my_slices():
            return _slices(my_atom_indices)

        #
        #         -----                    -----
        #          \    -1                  \    *
        # E      =  )  S     H    rho     =  )  c     eps  f  c
        #  mu nu   /    mu x  x z    z nu   /    n mu    n  n  n nu
        #         -----                    -----
        #          x z                       n
        #
        # We use the transpose of that matrix.  The first form is used
        # if rho is given, otherwise the coefficients are used.

        # rho_unMM = {}
        # F_unMM = {}

        for kpt in wfs.kpt_u:
            u = kpt.s * n_kps + kpt.q
            f_n = kpt.f_n
            n_occ = 0
            for f in f_n:
                if f > 1.0e-10:
                    n_occ += 1

            for m in range(n_occ):

                # calculate orbital-density matrix
                rho_xMM = \
                    kpt.f_n[m] * np.outer(kpt.C_nM[m].conj(),
                                          kpt.C_nM[m]) / (3.0 - wfs.nspins)
                # rho_unMM[u][m] = rho_xMM

                # calc S^{-1} F rho

                F_MM = \
                    self.get_orbital_potential_matrix(f_n, kpt.C_nM,
                                                      kpt,
                                                      wfs,
                                                      self.setups, m
                                                      )[0]

                sfrhoT_MM = np.linalg.solve(wfs.S_qMM[kpt.q],
                                           gemmdot(F_MM,
                                                   rho_xMM)).T.copy()

                del F_MM

                # Density matrix contribution due to basis overlap
                #
                #            ----- d Theta
                #  a          \           mu nu
                # F  += -2 Re  )   ------------  E
                #             /        d R        nu mu
                #            -----        mu nu
                #         mu in a; nu
                #
                dThetadRE_vMM = (dThetadR_qvMM[kpt.q] *
                                 sfrhoT_MM[np.newaxis]).real
                for a, M1, M2 in my_slices():
                    Ftheta_av[a, :] += \
                        -2.0 * dThetadRE_vMM[:, M1:M2].sum(-1).sum(-1)
                del dThetadRE_vMM

                # Density matrix contribution from PAW correction
                #
                #           -----                        -----
                #  a         \      a                     \     b
                # F +=  2 Re  )    Z      E        - 2 Re  )   Z      E
                #            /      mu nu  nu mu          /     mu nu  nu mu
                #           -----                        -----
                #           mu nu                    b; mu in a; nu
                #
                # with
                #                  b*
                #         -----  dP
                #   b      \       i mu    b   b
                #  Z     =  )   -------- dS   P
                #   mu nu  /     dR        ij  j nu
                #         -----    b mu
                #           ij
                #
                work_MM = np.zeros((mynao, nao), dtype)
                ZE_MM = None
                for b in my_atom_indices:
                    setup = self.setups[b]
                    dO_ii = np.asarray(setup.dO_ii, dtype)
                    dOP_iM = np.zeros((setup.ni, nao), dtype)
                    gemm(1.0, wfs.P_aqMi[b][kpt.q], dO_ii, 0.0,
                         dOP_iM, 'c')
                    for v in range(3):
                        gemm(1.0, dOP_iM,
                             dPdR_aqvMi[b][kpt.q][v][Mstart:Mstop],
                             0.0, work_MM, 'n')
                        ZE_MM = (work_MM * sfrhoT_MM).real
                        for a, M1, M2 in slices():
                            dE = 2 * ZE_MM[M1:M2].sum()
                            Frho_av[a, v] -= dE  # the "b; mu in a; nu" term
                            Frho_av[b, v] += dE  # the "mu nu" term
                del work_MM, ZE_MM

                # Potential contribution
                #
                #           -----      /  d Phi  (r)
                #  a         \        |        mu    ~
                # F += -2 Re  )       |   ---------- v (r)  Phi  (r) dr rho
                #            /        |     d R                nu          nu mu
                #           -----    /         a
                #        mu in a; nu
                #

                nt_G, Q_aL, D_ap = \
                    self.get_density(f_n,
                                     kpt.C_nM, kpt,
                                     wfs, self.setups, m)

                e_sic_m, vt_mG, vHt_g = \
                    self.get_pseudo_pot(nt_G, Q_aL, m, u)
                e_sic_paw_m, dH_ap = \
                    self.get_paw_corrections(D_ap, vHt_g)

                Fpot_av += \
                    self.bfs.calculate_force_contribution(vt_mG,
                                                     rho_xMM.T,
                                                     kpt.q)

                # Atomic density contribution
                #            -----                         -----
                #  a          \     a                       \     b
                # F  += -2 Re  )   A      rho       + 2 Re   )   A      rho
                #             /     mu nu    nu mu          /     mu nu    nu mu
                #            -----                         -----
                #            mu nu                     b; mu in a; nu
                #
                #                  b*
                #         ----- d P
                #  b       \       i mu   b   b
                # A     =   )   ------- dH   P
                #  mu nu   /    d R       ij  j nu
                #         -----    b mu
                #           ij
                #
                for b in my_atom_indices:
                    H_ii = np.asarray(unpack(dH_ap[b]),
                                      dtype)
                    HP_iM = gemmdot(H_ii,
                                    np.ascontiguousarray(
                                        wfs.P_aqMi[b][
                                            kpt.q].T.conj()))
                    for v in range(3):
                        dPdR_Mi = dPdR_aqvMi[b][kpt.q][v][
                                  Mstart:Mstop]
                        ArhoT_MM = (gemmdot(dPdR_Mi, HP_iM) *
                                    rho_xMM.T).real
                        for a, M1, M2 in slices():
                            dE = 2 * ArhoT_MM[M1:M2].sum()
                            Fatom_av[a, v] += dE  # the "b; mu in a; nu" term
                            Fatom_av[b, v] -= dE  # the "mu nu" term

                # contribution from hartree
                if self.sic_coarse_grid is False:
                    ghat_aLv = dens.ghat.dict(derivative=True)

                    dens.ghat.derivative(vHt_g, ghat_aLv)
                    for a, dF_Lv in ghat_aLv.items():
                        Fhart_av[a] -= self.beta_c * np.dot(Q_aL[a],
                                                            dF_Lv)
                else:
                    ghat_aLv = self.ghat_cg.dict(derivative=True)

                    self.ghat_cg.derivative(vHt_g, ghat_aLv)
                    for a, dF_Lv in ghat_aLv.items():
                        Fhart_av[a] -= self.beta_c * np.dot(Q_aL[a],
                                                            dF_Lv)

        # dens.finegd.comm.sum(Fhart_av, 0)

        F_av += Fpot_av + Ftheta_av + \
                Frho_av + Fatom_av + Fhart_av

        wfs.gd.comm.sum(F_av, 0)

        self.timer.start('Wait for sum')
        ksl.orbital_comm.sum(F_av)
        if wfs.bd.comm.rank == 0:
            wfs.kd.comm.sum(F_av, 0)

        wfs.world.broadcast(F_av, 0)
        self.timer.stop('Wait for sum')

        self.timer.stop('LCAO forces')

        return F_av * (3.0 - wfs.nspins)

    def get_hessian(self, kpt, H_MM, n_dim, wfs, setup, C_nM=None,
                    diag_heiss=False, h_type='ks'):

        if C_nM is None:
            C_nM = kpt.C_nM
        f_n = kpt.f_n

        if h_type == 'full':
            n_occ = 0
            for f in f_n:
                if f > 1.0e-10:
                    n_occ += 1

            b_nM = np.zeros(shape=(n_occ, C_nM.shape[1]), dtype=self.dtype)

            for n in range(n_occ):
                F_MM = self.get_orbital_potential_matrix(f_n, C_nM, kpt,
                                                         wfs, setup, n,
                                                         )[0]
                F_MM += H_MM
                gemv(1.0, F_MM.conj(), C_nM[n], 0.0, b_nM[n])

            L_occ = np.zeros((n_occ, n_occ), dtype=self.dtype)
            C_conj_nM = np.copy(C_nM.conj()[:n_occ])
            mmm(1.0, C_conj_nM, 'n', b_nM, 't', 0.0, L_occ)
            del C_conj_nM

            nrm_n = np.empty(L_occ.shape[0])
            L_occ = 0.5 * (L_occ + L_occ.T.conj())
            diagonalize(L_occ, nrm_n)
            kpt.eps_n[:n_occ] = nrm_n
            # kpt.eps_n[:n_occ] = np.einsum('ii->i', L_occ).real
            del L_occ

            L_unocc = np.einsum('jk,kl,il->ji',
                                C_nM.conj()[n_occ:],
                                H_MM.conj(), C_nM[n_occ:])
            L_unocc = 0.5 * (L_unocc + L_unocc.T.conj())
            nrm_n = np.empty(L_unocc.shape[0])
            diagonalize(L_unocc, nrm_n)
            kpt.eps_n[n_occ:] = nrm_n
            kpt.eps_n = sorted(kpt.eps_n)

        elif h_type == 'ks':

            HC_Mn = np.zeros_like(H_MM)
            mmm(1.0, H_MM.conj(), 'n', C_nM, 't', 0.0, HC_Mn)
            L = np.zeros_like(H_MM)
            if self.dtype is complex:
                mmm(1.0, C_nM.conj(), 'n', HC_Mn, 'n', 0.0, L)
            else:
                mmm(1.0, C_nM, 'n', HC_Mn, 'n', 0.0, L)

            nrm_n = np.empty(L.shape[0])
            diagonalize(L, nrm_n)
            kpt.eps_n = nrm_n
        elif h_type == 'kinetic':
            HC_Mn = np.zeros_like(H_MM)
            mmm(1.0, wfs.T_qMM[kpt.q].conj(),
                'n', C_nM,
                't', 0.0,
                HC_Mn)
            L = np.zeros_like(H_MM)
            if self.dtype is complex:
                mmm(1.0, C_nM.conj(), 'n', HC_Mn, 'n', 0.0, L)
            else:
                mmm(1.0, C_nM, 'n', HC_Mn, 'n', 0.0, L)

            nrm_n = np.empty(L.shape[0])
            diagonalize(L, nrm_n)
            kpt.eps_n = nrm_n


        else:
            raise NotImplementedError

        n1 = n_dim[kpt.s]
        if self.dtype == complex:
            n_d = n1 * n1
        else:
            n_d = n1 * (n1 - 1) // 2

        if self.dtype is complex:
            il1 = np.tril_indices(n1)
        else:
            il1 = np.tril_indices(n1, -1)
        il1 = list(il1)

        if diag_heiss:
            if self.dtype is float:
                Heiss = np.zeros(n_d)
                x = 0
                for l, m in zip(*il1):

                    df = f_n[l] - f_n[m]
                    Heiss[x] = -2.0 * (kpt.eps_n[l] - kpt.eps_n[m]) * df

                    if Heiss[x] < 0.0:
                        Heiss[x] = 0.0

                    x += 1
                return Heiss
            else:
                Heiss = np.zeros(len(il1[0]), dtype=self.dtype)
                x = 0

                for l, m in zip(*il1):

                    df = f_n[l] - f_n[m]
                    Heiss[x] = -2.0 * (kpt.eps_n[l] - kpt.eps_n[m]) * df

                    if Heiss[x] < 0.0:
                        Heiss[x] = 0.0
                    Heiss[x] += 1.0j * Heiss[x]

                    x += 1
                return Heiss

        else:
            raise NotImplementedError
