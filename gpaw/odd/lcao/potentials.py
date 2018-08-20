""" Potentials for orbital density dependent
    energy functionals

"""

import numpy as np
from gpaw.utilities import pack, unpack
from gpaw.utilities.blas import gemm, gemv, dotc, dotu, \
    gemmdot, mmm
# this is for lcao mode
from scipy.linalg import expm_frechet
from scipy.sparse import csc_matrix
from gpaw.odd.lcao.tools import get_grad_from_matrix_exponential, \
    D_matrix
from gpaw.utilities.lapack import diagonalize
from gpaw.lfc import LFC


class PZpotentialLcao:
    """
    Perdew-Zunger self-interaction corrections

    """

    def __init__(self, gd, xc, poisson,
                 ghat_fg, restrictor,
                 interpolator,
                 setups, beta, dtype, timer, bfs,
                 spos_ac=None,
                 sic_coarse_grid=False):

        self.cgd = gd
        self.finegd = ghat_fg.gd
        self.xc = xc
        self.poiss = poisson
        self.restrictor = restrictor
        self.interpolator = interpolator
        self.setups = setups
        self.bfs = bfs
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
            self.ghat = ghat_fg  # we usually solve poiss. on finegd
            self.ghat_cg = None

        self.timer = timer
        self.esic = 0.0
        self.dtype = dtype
        self.eigv_s = {}
        self.lagr_diag_s = {}
        self.counter = 0  # number of calls of this class
        # Scaling factor: len 1 or 2 array
        if len(beta) > 1:
            self.beta_x = beta[0]
            self.beta_c = beta[1]
        else:
            self.beta_x = self.beta_c = beta[0]

    def get_gradients_old(self, f_n, C_nM, kpt,
                      wfs, setup,
                      H_MM=None,
                      A=None, occupied_only=False):
        """
        :param C_nM: coefficients of orbitals
        :return: matrix G - gradients, and orbital SI energies

        which is G_{ij} = ( \int_0^1 e^{tA} L e^{-tA} dt )_{ji}

        L matrix is defined below:

        we use dotc which defined as:

        (a,b) =
            _
           \   cc
            ) a * b
           /_  i    i
           i

        ``cc`` denotes complex conjugation.

        Therefore:

        L_{ij} = (C_j, F_i C_i)^{cc} - (C_i, F_j C_j )  +

        if F_i is self-adjoint, +
        F_i = V^i, if i is occupied state and +
        F_i = -H, if i is unoccupied state    +

        L_{ij} is skew-hermitian matrix, so requies calculate
        upper triagonal part. Here, diagonal elements are zero

        Sequence:

              0. Initialize F_i, b_i, M_ij, L
                 call for F_i = get_F_i;

              1. Calculate b_i = F_i C_i for i = 0 .. n

              2. Calculate M[i][j] = (C_j, b_i) for 0 <= i < n
                                                    0 <= j < n
                                                    ???
                                                    can we use ???:
                                                    i + 1 <= j < n
                                                    Probably not

              3. Calculate L[i][j] = M[i][j]^{cc} - M[j][i]

              4. Calculate G

        """

        # 0.
        if occupied_only is True:
            nbs = 0
            for f in f_n:
                if f > 1.0e-10:
                    nbs += 1
            n_set = C_nM.shape[1]
        else:
            nbs = C_nM.shape[0]
            n_set = C_nM.shape[1]

        L = np.zeros(shape=(nbs, nbs), dtype=self.dtype)

        b_nM = np.zeros(shape=(nbs, n_set), dtype=self.dtype)

        M_ij = np.zeros(shape=(nbs, nbs), dtype=self.dtype)

        # 1.
        e_total_sic = np.array([])
        self.timer.start('ODD Construction of gradients')
        for n in range(nbs):
            if f_n[n] < 1.0e-10:
                # for unoccupied use minus KS hamiltonian
                gemv(-1.0, H_MM, C_nM[n], 0.0, b_nM[n])
                # b_nM[n] = -np.dot(H_MM, C_nM[n])
            else:
                F_MM, sic_energy_n = \
                     self.get_orbital_potential_matrix(f_n, C_nM,
                                                       kpt, wfs,
                                                       setup, n,
                                               occupied_only=occupied_only)

                gemv(1.0, F_MM, C_nM[n], 0.0, b_nM[n])
                # b_nM[n] = np.dot(F_nMM[n], C_nM[n])

                e_total_sic = np.append(e_total_sic, sic_energy_n,
                                        axis=0)

        del F_MM

        e_total_sic = e_total_sic.reshape(e_total_sic.shape[0] //
                                          2, 2)
        # 2.
        for i in range(nbs):
            for j in range(nbs):
                # unoccupied-unoccupied block is zero
                if f_n[i] > 1.0e-10 or f_n[j] > 1.0e-10:
                    if self.dtype is complex:
                        M_ij[i][j] = dotc(C_nM[j], b_nM[i])
                        # M_ij[i][j] = np.dot(C_nM[j].conj(), b_nM[i])
                    else:
                        M_ij[i][j] = dotu(C_nM[j], b_nM[i])
                        # M_ij[i][j] = np.dot(C_nM[j], b_nM[i])

        # 3.
        for i in range(nbs):
            for j in range(i+1, nbs):
                # unoccupied-unoccupied block is zero
                if f_n[i] > 1.0e-10 or f_n[j] > 1.0e-10:
                    if self.dtype is complex:
                        L[i][j] = \
                                np.conjugate(M_ij[i][j]) - M_ij[j][i]
                        L[j][i] = - np.conjugate(L[i][j])
                    else:
                        L[i][j] = M_ij[i][j] - M_ij[j][i]
                        L[j][i] = - L[i][j]

        self.timer.stop('ODD Construction of gradients')

        self.counter += 1

        if A is None:
            return L.T, e_total_sic  # sic_energy_n
        else:
            self.timer.start('ODD Matrix_integrals')
            G = get_grad_from_matrix_exponential(A, L)
            self.timer.stop('ODD Matrix_integrals')

            for i in range(nbs):
                G[i][i] *= 0.5

            # G = 0.5*(dE/dx - idE/dy)
            # let's return G = (dE/dx + idE/dy)

            return 2.0*G.conj(), e_total_sic  # sic_energy_n

    def get_gradients(self, f_n, C_nM, kpt,
                      wfs, setup, evec, eval,
                      H_MM=None,
                      A=None, occupied_only=False):
        """
        :param C_nM: coefficients of orbitals
        :return: matrix G - gradients, and orbital SI energies

        which is G_{ij} = (1 - delta_{ij}/2)*( \int_0^1 e^{tA} L e^{-tA} dt )_{ji}

        Lambda_ij = (C_i, F_j C_j )

        L_{ij} = Lambda_ji^{cc} - Lambda_ij

        """

        # 0.
        n_occ = 0
        for f in f_n:
            if f > 1.0e-10:
                n_occ += 1

        if occupied_only is True:
            nbs = n_occ
            n_set = C_nM.shape[1]
        else:
            nbs = C_nM.shape[0]
            n_set = C_nM.shape[1]

        self.timer.start('ODD Construction of gradients')

        HC_Mn = np.zeros_like(H_MM)
        mmm(1.0, H_MM.conj(), 'n', C_nM, 't', 0.0, HC_Mn)
        L = np.zeros_like(H_MM)

        if self.dtype is complex:
            mmm(1.0, C_nM.conj(), 'n', HC_Mn, 'n', 0.0, L)
        else:
            mmm(1.0, C_nM, 'n', HC_Mn, 'n', 0.0, L)
        del HC_Mn

        # odd part
        b_nM = np.zeros(shape=(nbs, n_set), dtype=self.dtype)
        e_total_sic = np.array([])
        for n in range(n_occ):
            F_MM, sic_energy_n =\
                self.get_orbital_potential_matrix(f_n, C_nM, kpt,
                                                  wfs, setup, n,
                                                  occupied_only=
                                                  occupied_only)

            gemv(1.0, F_MM.conj(), C_nM[n], 0.0, b_nM[n])

            e_total_sic = np.append(e_total_sic, sic_energy_n,
                                    axis=0)
        L_odd = C_nM[:nbs].conj() @ b_nM.T

        f = f_n[:nbs]
        L = f[:, np.newaxis] * (L[:nbs, :nbs] + L_odd.T.conj()) - \
            f * (L[:nbs, :nbs] + L_odd)

        e_total_sic = e_total_sic.reshape(e_total_sic.shape[0] //
                                          2, 2)
        self.timer.stop('ODD Construction of gradients')
        self.counter += 1

        if A is None:
            return L.T, e_total_sic
        else:

            self.timer.start('ODD matrix integrals')
            G = evec.T.conj() @ L.T.conj() @ evec
            G = G * D_matrix(eval)
            G = evec @ G @ evec.T.conj()
            self.timer.stop('ODD matrix integrals')

            for i in range(G.shape[0]):
                G[i][i] *= 0.5

            if A.dtype == float:
                return 2.0 * G.real, e_total_sic
            else:
                return 2.0 * G, e_total_sic

    def get_orbital_potential_matrix(self, f_n, C_nM, kpt,
                             wfs, setup, m, occupied_only=False):
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
            self.get_pseudo_pot(nt_G, Q_aL, m)

        # calculate PAW corrections
        e_sic_paw_m, dH_ap = \
            self.get_paw_corrections(D_ap, vHt_g, m)

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

    def get_pseudo_pot(self, nt, Q_aL, m):

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
        zero_initial_phi = True
        self.poiss.solve(vHt_g, nt_sg[0],
                         zero_initial_phi=zero_initial_phi)
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

    def get_paw_corrections(self, D_ap, vHt_g, m):

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

    def update_eigenval(self, f_n, C_nM, kpt,
                      wfs, setup,
                      H_MM, occupied_only=False):
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
                                                     occupied_only=
                                                     occupied_only)[0]
            F_MM += H_MM
            gemv(1.0, F_MM, C_nM[n], 0.0, b_nM[n])

        L_occ = np.zeros((n_occ, n_occ), dtype=self.dtype)
        C_conj_nM = np.copy(C_nM.conj()[:n_occ])
        mmm(1.0, C_conj_nM, 'n', b_nM, 't', 0.0, L_occ)
        L_occ = 0.5 * (L_occ + L_occ.T.conj())
        del C_conj_nM

        L_unocc = np.einsum('jk,kl,il->ji',
                            C_nM.conj()[n_occ:], H_MM, C_nM[n_occ:])
        L_unocc = 0.5 * (L_unocc + L_unocc.T.conj())

        self.lagr_diag_s[u] = \
            np.append(np.einsum('ii->i', L_occ),
                      np.einsum('ii->i', L_unocc)).real

        # occupied eigenvalues
        # TODO: fix it, when there is no occ numbers
        # Can_nM = np.zeros_like(C_nM)
        if n_occ > 0:
            eig_occ = np.empty(L_occ.shape[0])
            diagonalize(L_occ, eig_occ)
            kpt.eps_n[:n_occ] = eig_occ

            # Can_nM[:n_occ] = np.dot(L_occ.conj(), C_nM[:n_occ])

        # unoccupied eigenvalues
        if L_unocc.shape[0] > 0:
            eig_unocc = np.empty(L_unocc.shape[0])
            diagonalize(L_unocc, eig_unocc)
            kpt.eps_n[n_occ:] = eig_unocc

            # Can_nM[n_occ:] = np.dot(L_occ.conj(), C_nM[n_occ:])

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
        tci = wfs.tci
        gd = wfs.gd
        # bfs = wfs.basis_functions

        Mstart = ksl.Mstart
        Mstop = ksl.Mstop
        n_kps = wfs.kd.nks // wfs.kd.nspins

        self.timer.start('TCI derivative')
        dThetadR_qvMM = np.empty((nq, 3, mynao, nao), dtype)
        dTdR_qvMM = np.empty((nq, 3, mynao, nao), dtype)
        dPdR_aqvMi = {}
        for a in self.bfs.my_atom_indices:
            ni = self.setups[a].ni
            dPdR_aqvMi[a] = np.empty((nq, 3, nao, ni), dtype)
        tci.calculate_derivative(spos_ac, dThetadR_qvMM, dTdR_qvMM,
                                 dPdR_aqvMi)
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
                                          kpt.C_nM[m])
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
                    self.get_pseudo_pot(nt_G, Q_aL, m)
                e_sic_paw_m, dH_ap = \
                    self.get_paw_corrections(D_ap, vHt_g, m)

                Fpot_av += \
                    self.bfs.calculate_force_contribution(vt_mG,
                                                     rho_xMM,
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
                                    rho_xMM).real
                        for a, M1, M2 in slices():
                            dE = 2 * ArhoT_MM[M1:M2].sum()
                            Fatom_av[a, v] += dE  # the "b; mu in a; nu" term
                            Fatom_av[b, v] -= dE  # the "mu nu" term

                # contribution from hartree
                if self.sic_coarse_grid is False:
                    ghat_aLv = dens.ghat.dict(derivative=True)

                    dens.ghat.derivative(vHt_g, ghat_aLv)
                    for a, dF_Lv in ghat_aLv.items():
                        Fhart_av[a] += -1.0 * np.dot(Q_aL[a], dF_Lv)
                else:
                    ghat_aLv = self.ghat_cg.dict(derivative=True)

                    self.ghat_cg.derivative(vHt_g, ghat_aLv)
                    for a, dF_Lv in ghat_aLv.items():
                        Fhart_av[a] += -1.0 * np.dot(Q_aL[a],
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

        return F_av

    def get_hessian(self, kpt, H_MM, n_dim, wfs, setup, C_nM=None,
                    diag_heiss=False, occupied_only=False, h_type='ks'):

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
                                                         occupied_only=
                                                         occupied_only)[0]
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

    def get_canonical_orbitals_and_evals(self, wfs, kpt,
                                         H_MM, occupied_only=False):

        C_nM = kpt.C_nM
        f_n = kpt.f_n
        n_occ = 0
        for f in f_n:
            if f > 1.0e-10:
                n_occ += 1

        b_nM = np.zeros(shape=(n_occ, C_nM.shape[1]), dtype=self.dtype)

        for n in range(n_occ):
            F_MM = self.get_orbital_potential_matrix(f_n, C_nM, kpt,
                                                     wfs, wfs.setups, n,
                                                     occupied_only=
                                                     occupied_only)[0]
            F_MM += H_MM
            gemv(1.0, F_MM, C_nM[n], 0.0, b_nM[n])

        L_occ = np.zeros((n_occ, n_occ), dtype=self.dtype)
        C_conj_nM = np.copy(C_nM.conj()[:n_occ])
        mmm(1.0, C_conj_nM, 'n', b_nM, 't', 0.0, L_occ)
        L_occ = 0.5 * (L_occ + L_occ.T.conj())
        del C_conj_nM

        L_unocc = np.einsum('jk,kl,il->ji',
                            C_nM.conj()[n_occ:], H_MM, C_nM[n_occ:])
        L_unocc = 0.5 * (L_unocc + L_unocc.T.conj())

        a = L_occ.shape[0]
        b = L_unocc.shape[0]
        L_tot = np.vstack([np.hstack([L_occ,
                                      np.zeros(shape=(a, b),
                                               dtype=self.dtype)]),
                           np.hstack([np.zeros(shape=(b, a),
                                               dtype=self.dtype),
                                      L_unocc])])

        evals = np.empty(L_tot.shape[0])
        diagonalize(L_tot, evals)
        Can_nM = np.dot(L_tot.conj(), C_nM)

        return evals, Can_nM


class ZeroOddLcao:
    """
    Zero self-interaction corrections
    """

    def __init__(self, dtype, timer):

        self.timer = timer
        self.dtype = dtype
        self.eigv_s = {}

        self.counter = 0  # number of calls of this class

    def get_gradients(self, f_n, C_nM, kpt,
                      wfs, setup, evec, eval,
                      H_MM=None,
                      A=None, occupied_only=False):
        """
        :param C_nM: coefficients of orbitals
        :return: matrix G - gradients, and orbital SI energies
        """

        self.timer.start('ODD construction of L')

        sic_energy_n = np.zeros(shape=(1, 1),
                         dtype=float)
        HC_Mn = np.zeros_like(H_MM)
        mmm(1.0, H_MM.conj(), 'n', C_nM, 't', 0.0, HC_Mn)
        L = np.zeros_like(H_MM)

        if self.dtype is complex:
            mmm(1.0, C_nM.conj(), 'n', HC_Mn, 'n', 0.0, L)
        else:
            mmm(1.0, C_nM, 'n', HC_Mn, 'n', 0.0, L)

        L = f_n[:, np.newaxis] * L - f_n * L

        self.counter += 1

        self.timer.stop('ODD construction of L')

        if A is None:

            return L.T, sic_energy_n
        else:

            self.timer.start('ODD matrix integrals')
            G = evec.T.conj() @ L.T.conj() @ evec
            G = G * D_matrix(eval)
            G = evec @ G @ evec.T.conj()
            self.timer.stop('ODD matrix integrals')

            for i in range(G.shape[0]):
                G[i][i] *= 0.5

            if A.dtype == float:
                return 2.0 * G.real, sic_energy_n
            else:
                return 2.0 * G, sic_energy_n

    def update_eigenval(self, f_n, C_nM, kpt,
                        wfs, setup,
                        H_MM, occupied_only=False):

        n_kps = wfs.kd.nks // wfs.kd.nspins
        u = kpt.s * n_kps + kpt.q

        HC_Mn = np.zeros_like(H_MM)
        mmm(1.0, H_MM.conj(), 'n', C_nM, 't', 0.0, HC_Mn)
        L = np.zeros_like(H_MM)

        if self.dtype is complex:
            mmm(1.0, C_nM.conj(), 'n', HC_Mn, 'n', 0.0, L)
        else:
            mmm(1.0, C_nM, 'n', HC_Mn, 'n', 0.0, L)

        nrm_n = np.empty(L.shape[0])
        diagonalize(L, nrm_n)

        kpt.C_nM = \
            np.dot(L.conj(), kpt.C_nM)

        kpt.eps_n = nrm_n

        # FIXME:
        # wfs.gd.comm.broadcast(kpt.C_nM, 0)
        # wfs.gd.comm.broadcast(kpt.eps_n, 0)

        self.eigv_s[u] = np.copy(kpt.eps_n)

    def update_eigenval_2(self, C_nM, kpt, H_MM):

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

        C_nM = \
            np.dot(L.conj(), C_nM)

        return C_nM

    def get_gradients_wrt_coeff(self, f_n, C_nM, kpt,
                                wfs, setup, H_MM,
                                S_inv_MM=None, orthonorm_grad=False):

        """
        :return:
        matrix
        g_ni = \sum_{l} C_il ( H_nl + V^i_nl)
        """

        self.counter += 1

        nbs = 0
        for f in f_n:
            if f > 1.0e-10:
                nbs += 1
        n_set = C_nM.shape[1]

        g_nM = np.zeros_like(C_nM[:nbs])

        if S_inv_MM is None:
            for i in range(nbs):
                g_nM[i] = np.dot(H_MM, C_nM[i])
        else:
            for i in range(nbs):
                g_nM[i] = np.dot(S_inv_MM, np.dot(H_MM, C_nM[i]))

        if orthonorm_grad:
            for i in range(nbs):
                for j in range(nbs):
                    g_nM[i] -= 0.5 * C_nM[j] * \
                                   (np.dot(C_nM[i], np.dot(H_MM,
                                                           C_nM[j])) +
                                    np.dot(np.dot(H_MM, C_nM[i]),
                                           C_nM[j])
                                    )


        return g_nM, np.array([0.0])

    def update_orbital_energies(self, C_nM, kpt, H_MM):

        HC_Mn = np.zeros_like(H_MM)
        mmm(1.0, H_MM.conj(), 'n', C_nM, 't', 0.0, HC_Mn)
        L = np.zeros_like(H_MM)

        if self.dtype is complex:
            mmm(1.0, C_nM.conj(), 'n', HC_Mn, 'n', 0.0, L)
        else:
            mmm(1.0, C_nM, 'n', HC_Mn, 'n', 0.0, L)

        kpt.eps_n = np.diagonal(L.real).copy()
        # nrm_n = np.empty(L.shape[0])
        # diagonalize(L, nrm_n)
        # kpt.eps_n = nrm_n

    def get_hessian(self, kpt, H_MM, C_nM=None):

        if C_nM is None:
            C_nM = kpt.C_nM

        f_n = kpt.f_n
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

        if self.dtype is complex:
            il1 = np.tril_indices(H_MM.shape[0])
        else:
            il1 = np.tril_indices(H_MM.shape[0], -1)

        il1 = list(il1)

        heiss = np.zeros(len(il1[0]), dtype=self.dtype)
        x = 0
        for l, m in zip(*il1):
            df = f_n[l] - f_n[m]
            heiss[x] = -2.0 * (kpt.eps_n[l] - kpt.eps_n[m]) * df
            if self.dtype is complex:
                heiss[x] += 1.0j * heiss[x]
                if abs(heiss[x]) < 1.0e-10:
                    heiss[x] = 0.0 + 0.0j
            else:
                if abs(heiss[x]) < 1.0e-10:
                    heiss[x] = 0.0
            x += 1

        return heiss
