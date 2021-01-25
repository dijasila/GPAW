from gpaw.directmin.fdpw.tools import get_n_occ, get_indices, expm_ed
from gpaw.directmin.fdpw.sd_inner import LBFGS_P
from gpaw.directmin.fdpw.ls_inner import StrongWolfeConditions as SWC
from ase.units import Hartree
import numpy as np
import time

from ase.parallel import parprint

class InnerLoop:

    def __init__(self, odd_pot, wfs, tol=1.0e-3, maxiter=50,
                 g_tol=5.0e-4):

        self.odd_pot = odd_pot
        self.n_kps = wfs.kd.nks // wfs.kd.nspins
        self.g_tol = g_tol / Hartree
        self.tol = tol
        self.dtype = wfs.dtype
        self.get_en_and_grad_iters = 0
        self.method = 'LBFGS'
        self.line_search_method = 'AwcSwc'
        self.max_iter_line_search = 6
        self.n_counter = maxiter
        self.eg_count = 0
        self.total_eg_count = 0
        self.run_count = 0
        self.U_k = {}
        self.Unew_k = {}
        self.e_total = 0.0
        self.n_occ = {}
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            self.n_occ[k] = get_n_occ(kpt)
            dim1 = self.n_occ[k]
            dim2 = self.n_occ[k]
            self.U_k[k] = np.eye(dim1, dtype=self.dtype)
            self.Unew_k[k] = np.eye(dim2, dtype=self.dtype)

    def get_energy_and_gradients(self, a_k, wfs, dens):
        """
        Energy E = E[A]. Gradients G_ij[A] = dE/dA_ij
        Returns E[A] and G[A] at psi = exp(A).T kpt.psi
        :param a_k: A
        :return:
        """

        if self.odd_pot.name == 'SPZ_SIC2':
            return self.get_energy_and_gradients2(a_k, wfs, dens)

        g_k = {}
        self.e_total = 0.0

        self.kappa = 0.0
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            # n_occ = self.n_occ[k]
            dim2 = self.Unew_k[k].shape[0]
            if dim2 == 0:
                g_k[k] = np.zeros_like(a_k[k])
                continue
            wfs.timer.start('Unitary matrix')
            u_mat, evecs, evals = expm_ed(a_k[k], evalevec=True)
            wfs.timer.stop('Unitary matrix')
            self.Unew_k[k] = u_mat.copy()
            if wfs.mode == 'lcao':
                kpt.C_nM[:dim2] = u_mat.T @ self.c_knm[k][:dim2]
                wfs.atomic_correction.calculate_projections(wfs, kpt)
            else:
                kpt.psit_nG[:dim2] = \
                    np.tensordot(u_mat.T, self.psit_knG[k][:dim2],
                                 axes=1)
                # calc projectors
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

            del u_mat
            wfs.timer.start('Energy and gradients')
            g_k[k], e_sic, kappa1 = \
                self.odd_pot.get_energy_and_gradients_inner_loop(
                    wfs, kpt, a_k[k], evals, evecs, dens)
            wfs.timer.stop('Energy and gradients')
            if kappa1 > self.kappa:
                self.kappa = kappa1
            self.e_total += e_sic

        self.kappa = wfs.kd.comm.max(self.kappa)
        self.e_total = wfs.kd.comm.sum(self.e_total)
        self.eg_count += 1
        self.total_eg_count += 1

        return self.e_total, g_k

    def evaluate_phi_and_der_phi(self, a_k, p_k, alpha, wfs, dens,
                                 phi=None, g_k=None):
        """
        phi = f(x_k + alpha_k*p_k)
        der_phi = \grad f(x_k + alpha_k*p_k) \cdot p_k
        :return:  phi, der_phi, \grad f(x_k + alpha_k*p_k)
        """
        if phi is None or g_k is None:
            x_k = {k: a_k[k] + alpha * p_k[k] for k in a_k.keys()}
            phi, g_k = \
                self.get_energy_and_gradients(x_k, wfs, dens)
            del x_k
        else:
            pass

        der_phi = 0.0
        for k in p_k.keys():
            il1 = get_indices(p_k[k].shape[0], self.dtype)
            der_phi += np.dot(g_k[k][il1].conj(), p_k[k][il1]).real

        der_phi = wfs.kd.comm.sum(der_phi)

        return phi, der_phi, g_k

    def get_search_direction(self, a_k, g_k, wfs):

        # structure of vector is
        # (x_1_up, x_2_up,..,y_1_up, y_2_up,..,
        #  x_1_down, x_2_down,..,y_1_down, y_2_down,.. )

        a = {}
        g = {}

        for k in a_k.keys():
            il1 = get_indices(a_k[k].shape[0], self.dtype)
            a[k] = a_k[k][il1]
            g[k] = g_k[k][il1]

        p = self.sd.update_data(wfs, a, g)
        del a, g

        p_k = {}
        for k in p.keys():
            p_k[k] = np.zeros_like(a_k[k])
            il1 = get_indices(a_k[k].shape[0], self.dtype)
            p_k[k][il1] = p[k]
            # make it skew-hermitian
            ind_l = np.tril_indices(p_k[k].shape[0], -1)
            p_k[k][(ind_l[1], ind_l[0])] = -p_k[k][ind_l].conj()
        del p

        return p_k

    def run(self, e_ks, wfs, dens, log, outer_counter=0,
            small_random=True):

        log = log
        self.run_count += 1
        self.counter = 0
        self.eg_count = 0
        # initial things
        self.c_knm = {}
        self.psit_knG = {}

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            dim1 = self.U_k[k].shape[0]
            if wfs.mode == 'lcao':
                self.c_knm[k] = self.U_k[k].T @ kpt.C_nM[:dim1]
            else:
                self.psit_knG[k] = np.tensordot(self.U_k[k].T,
                                                kpt.psit_nG[:dim1],
                                                axes=1)

        a_k = {}
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            d = self.Unew_k[k].shape[0]
            # a_k[k] = np.zeros(shape=(d, d), dtype=self.dtype)
            if self.run_count == 1 and self.dtype is complex\
                    and small_random:
                a = 0.01 * np.random.rand(d, d) * 1.0j
                a = a - a.T.conj()
                # a = np.zeros(shape=(d, d), dtype=self.dtype)
                wfs.gd.comm.broadcast(a, 0)
                a_k[k] = a
            else:
                a_k[k] = np.zeros(shape=(d, d), dtype=self.dtype)

        self.sd = LBFGS_P(wfs, memory=20)
        # self.ls = US(self.evaluate_phi_and_der_phi)
        self.ls = SWC(
            self.evaluate_phi_and_der_phi,
            method=self.method, awc=True,
            max_iter=self.max_iter_line_search)

        threelasten = []
        # get initial energy and gradients
        self.e_total, g_k = self.get_energy_and_gradients(a_k, wfs, dens)
        threelasten.append(self.e_total)
        g_max = g_max_norm(g_k, wfs)
        if g_max < self.g_tol:
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                # n_occ = self.n_occ[k]
                dim1 = self.U_k[k].shape[0]
                if wfs.mode == 'lcao':
                    kpt.C_nM[:dim1] = self.U_k[k].conj() @ self.c_knm[k]
                    wfs.atomic_correction.calculate_projections(wfs,
                                                                kpt)
                else:
                    kpt.psit_nG[:dim1] = np.tensordot(self.U_k[k].conj(),
                                                       self.psit_knG[k],
                                                       axes=1)
                    # calc projectors
                    wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

                dim2 = self.Unew_k[k].shape[0]
                if dim1 == dim2:
                    self.U_k[k] = self.U_k[k] @ self.Unew_k[k]
                else:
                    u_oo = self.Unew_k[k]
                    u_ov = np.zeros(shape=(dim2, dim1-dim2),
                                    dtype=self.dtype)
                    u_vv = np.eye(dim1-dim2,dtype=self.dtype)
                    unew = np.vstack([np.hstack([u_oo, u_ov]),
                                      np.hstack([u_ov.T, u_vv])])
                    self.U_k[k] = self.U_k[k] @ unew

            del self.psit_knG
            del self.c_knm
            if outer_counter is None:
                return self.e_total, self.counter
            else:
                return self.e_total, outer_counter

        # get maximum of gradients
        # max_norm = []
        # for kpt in wfs.kpt_u:
        #     k = self.n_kps * kpt.s + kpt.q
        #     if self.n_occ[k] == 0:
        #         continue
        #     max_norm.append(np.max(np.absolute(g_k[k])))
        # max_norm = np.max(np.asarray(max_norm))
        # g_max = wfs.world.max(max_norm)

        # stuff which are needed for minim.
        phi_0 = self.e_total
        phi_old = None
        der_phi_old = None
        phi_old_2 = None
        der_phi_old_2 = None

        outer_counter += 1
        if log is not None:
            log_f(log, self.counter, self.kappa, e_ks, self.e_total,
                  outer_counter, g_max)

        alpha = 1.0
        # if self.kappa < self.tol:
        #     not_converged = False
        # else:
        #     not_converged = True
        # not_converged = \
        #     g_max > self.g_tol and counter < self.n_counter
        not_converged = True
        while not_converged:

            # calculate search direction fot current As and Gs
            p_k = self.get_search_direction(a_k, g_k, wfs)

            # calculate derivative along the search direction
            phi_0, der_phi_0, g_k = \
                self.evaluate_phi_and_der_phi(a_k, p_k,
                                              0.0, wfs, dens,
                                              phi=phi_0, g_k=g_k)
            if self.counter > 1:
                phi_old = phi_0
                der_phi_old = der_phi_0

            # choose optimal step length along the search direction
            # also get energy and gradients for optimal step
            alpha, phi_0, der_phi_0, g_k = \
                self.ls.step_length_update(
                    a_k, p_k, wfs, dens,
                    phi_0=phi_0, der_phi_0=der_phi_0,
                    phi_old=phi_old_2, der_phi_old=der_phi_old_2,
                    alpha_max=3.0, alpha_old=alpha)

            # broadcast data is gd.comm > 1
            if wfs.gd.comm.size > 1:
                alpha_phi_der_phi = np.array([alpha, phi_0,
                                              der_phi_0])
                wfs.gd.comm.broadcast(alpha_phi_der_phi, 0)
                alpha = alpha_phi_der_phi[0]
                phi_0 = alpha_phi_der_phi[1]
                der_phi_0 = alpha_phi_der_phi[2]
                for kpt in wfs.kpt_u:
                    k = self.n_kps * kpt.s + kpt.q
                    if self.n_occ[k] == 0:
                        continue
                    wfs.gd.comm.broadcast(g_k[k], 0)

            phi_old_2 = phi_old
            der_phi_old_2 = der_phi_old

            if alpha > 1.0e-10:
                # calculate new matrices at optimal step lenght
                a_k = {k: a_k[k] + alpha * p_k[k] for k in a_k.keys()}
                g_max = g_max_norm(g_k, wfs)

                # output
                self.counter += 1
                if outer_counter is not None:
                    outer_counter += 1
                if log is not None:
                    log_f(
                        log, self.counter, self.kappa, e_ks, phi_0,
                        outer_counter, g_max)

                not_converged = g_max > self.g_tol and \
                                self.counter < self.n_counter

                threelasten.append(phi_0)
                if len(threelasten) > 2:
                    threelasten = threelasten[-3:]
                    if threelasten[0] < threelasten[1] < \
                            threelasten[2]:
                        if log is not None:
                           log(
                                'Could not converge, leave the loop',
                                flush=True)
                        break
                        # reset things:
                        # threelasten = []
                        # self.sd = LBFGS_P(wfs, memory=20)
                        # self.ls = SWC(
                        #     self.evaluate_phi_and_der_phi,
                        #     method=self.method, awc=True,
                        #     max_iter=self.max_iter_line_search)
                        # for kpt in wfs.kpt_u:
                        #     k = self.n_kps * kpt.s + kpt.q
                        #     n_occ = self.n_occ[k]
                        #     self.psit_knG[k] = np.tensordot(
                        #         self.Unew_k[k].T,
                        #         self.psit_knG[k],
                        #         axes=1)
                        #     self.U_k[k] = self.U_k[k] @ self.Unew_k[k]
                        #     d = self.n_occ[k]
                        #     if a_k[k].dtype == complex:
                        #         a_k[k] = 1.0e-5 * np.random.rand(d, d) * 1.0j + \
                        #                  1.0e-5 * np.random.rand(d, d)
                        #     else:
                        #         a_k[k] = 1.0e-5 * np.random.rand(d, d)
                        #
                        # self.e_total, g_k = self.get_energy_and_gradients(
                        #     a_k, wfs, dens)
                        #
                        # phi_0 = self.e_total
                        # phi_old = None
                        # der_phi_old = None
                        # phi_old_2 = None
                        # der_phi_old_2 = None

                # if not not_converged and self.counter < 2:
                #     not_converged = True

            else:
                break

        if log is not None:
            log('INNER LOOP FINISHED.\n')
            log('Total number of e/g calls:' + str(self.eg_count))

        # for kpt in wfs.kpt_u:
        #     k = self.n_kps * kpt.s + kpt.q
        #     n_occ = self.n_occ[k]
        #     if n_occ == 0:
        #         g_k[k] = np.zeros_like(a_k[k])
        #         continue
        #     wfs.timer.start('Unitary matrix')
        #     u_mat, evecs, evals = expm_ed(a_k[k], evalevec=True)
        #     wfs.timer.stop('Unitary matrix')
        #     self.U_k[k] = u_mat.copy()
        #     kpt.psit_nG[:n_occ] = \
        #         np.tensordot(u_mat.T, self.psit_knG[k][:n_occ],
        #                      axes=1)
        #     # calc projectors
        #     wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            # n_occ = self.n_occ[k]
            dim1 = self.U_k[k].shape[0]
            if wfs.mode == 'lcao':
                kpt.C_nM[:dim1] = self.U_k[k].conj() @ self.c_knm[k][:dim1]
                wfs.atomic_correction.calculate_projections(wfs, kpt)
            else:
                kpt.psit_nG[:dim1] = np.tensordot(self.U_k[k].conj(),
                                                   self.psit_knG[k][:dim1],
                                                   axes=1)
                # calc projectors
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
            dim2 = self.Unew_k[k].shape[0]
            if dim1 == dim2:
                self.U_k[k] = self.U_k[k] @ self.Unew_k[k]
            else:
                u_oo = self.Unew_k[k]
                u_ov = np.zeros(shape=(dim2, dim1 - dim2),
                                dtype=self.dtype)
                u_vv = np.eye(dim1 - dim2, dtype=self.dtype)
                unew = np.vstack([np.hstack([u_oo, u_ov]),
                                  np.hstack([u_ov.T, u_vv])])
                self.U_k[k] = self.U_k[k] @ unew

        del self.psit_knG
        del self.c_knm
        if outer_counter is None:
            return self.e_total, self.counter
        else:
            return self.e_total, outer_counter

    def get_numerical_gradients(self, A_s, wfs, dens, log, eps=1.0e-5,
                                dtype=None):
        dtype=self.dtype
        h = [eps, -eps]
        coef = [1.0, -1.0]
        Gr_n_x = {}
        Gr_n_y = {}
        E_0, G = self.get_energy_and_gradients(A_s, wfs, dens)
        log("Estimating gradients using finite differences..")
        log(flush=True)

        if dtype is complex:
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                dim = A_s[k].shape[0]
                iut = np.triu_indices(dim, 1)
                dim_gr = iut[0].shape[0]

                for z in range(2):
                    grad_num = np.zeros(shape=dim_gr,
                                        dtype=self.dtype)
                    igr = 0
                    for i, j in zip(*iut):
                        log(igr + 1, 'out of', dim_gr, 'for a', k,
                            'kpt and', z, 'real/compl comp.')
                        log(flush=True)
                        A = A_s[k][i][j]
                        for l in range(2):
                            if z == 1:
                                if i == j:
                                    A_s[k][i][j] = A + 1.0j * h[l]
                                else:
                                    A_s[k][i][j] = A + 1.0j * h[
                                        l]
                                    A_s[k][j][i] = -np.conjugate(
                                        A + 1.0j * h[l])
                            else:
                                if i == j:
                                    A_s[k][i][j] = A + 0.0j * h[l]
                                else:
                                    A_s[k][i][j] = A + h[
                                        l]
                                    A_s[k][j][i] = -np.conjugate(
                                        A + h[l])
                            E =\
                                self.get_energy_and_gradients(
                                    A_s, wfs, dens)[0]
                            grad_num[igr] += E * coef[l]
                        grad_num[igr]*= 1.0 / (2.0 * eps)
                        if i == j:
                            A_s[k][i][j] = A
                        else:
                            A_s[k][i][j] = A
                            A_s[k][j][i] = -np.conjugate(A)
                        igr += 1
                    if z == 0:
                        Gr_n_x[k] = grad_num.copy()
                    else:
                        Gr_n_y[k] = grad_num.copy()
                G[k] = G[k][iut]

            Gr_n = {k: (Gr_n_x[k] + 1.0j * Gr_n_y[k]) for k in
                    Gr_n_x.keys()}
        else:
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                dim = A_s[k].shape[0]
                iut = np.triu_indices(dim, 1)
                dim_gr = iut[0].shape[0]
                grad_num = np.zeros(shape=dim_gr, dtype=self.dtype)

                igr = 0
                for i, j in zip(*iut):
                    # log(k, i, j)
                    log(igr+1, 'out of ', dim_gr, 'for a', k, 'kpt')
                    log(flush=True)
                    A = A_s[k][i][j]
                    for l in range(2):
                        A_s[k][i][j] = A + h[l]
                        A_s[k][j][i] = -(A + h[l])
                        E = self.get_energy_and_gradients(A_s, wfs, dens)[0]
                        grad_num[igr] += E * coef[l]
                    grad_num[igr] *= 1.0 / (2.0 * eps)
                    A_s[k][i][j] = A
                    A_s[k][j][i] = -A
                    igr += 1

                Gr_n_x[k] = grad_num.copy()
                G[k] = G[k][iut]

            Gr_n = {k: (Gr_n_x[k]) for k in Gr_n_x.keys()}

        return G, Gr_n

    def get_numerical_hessian(self, A_s, wfs, dens, log, eps=1.0e-5,
                                dtype=None):

        dtype=self.dtype
        h = [eps, -eps]
        coef = [1.0, -1.0]
        log("Estimating Hessian using finite differences..")
        log(flush=True)
        num_hes = {}

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            dim = A_s[k].shape[0]
            iut = np.tril_indices(dim, -1)
            dim_gr = iut[0].shape[0]
            hessian = np.zeros(shape=(dim_gr,dim_gr),
                               dtype=self.dtype)
            ih = 0
            for i, j in zip(*iut):
                # log(k, i, j)
                log(ih + 1, 'out of ', dim_gr, 'for a', k, 'kpt')
                log(flush=True)
                A = A_s[k][i][j]
                for l in range(2):
                    A_s[k][i][j] = A + h[l]
                    A_s[k][j][i] = -(A + h[l])
                    g = self.get_energy_and_gradients(A_s, wfs, dens)[1]
                    g = g[k][iut]
                    hessian[ih, :] += g * coef[l]

                hessian[ih, :] *= 1.0 / (2.0 * eps)

                A_s[k][i][j] = A
                A_s[k][j][i] = -A
                ih += 1

            num_hes[k] = hessian.copy()

        return num_hes

    def get_energy_and_gradients2(self, a_k, wfs, dens):

        """
        Energy E = E[A]. Gradients G_ij[A] = dE/dA_ij
        Returns E[A] and G[A] at psi = exp(A).T kpt.psi
        :param a_k: A
        :return:
        """

        g_k = {}
        evals_k = {}
        evecs_k = {}

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            n_occ = self.n_occ[k]
            if n_occ == 0:
                g_k[k] = np.zeros_like(a_k[k])
                continue
            wfs.timer.start('Unitary matrix')
            u_mat, evecs, evals = expm_ed(a_k[k], evalevec=True)
            wfs.timer.stop('Unitary matrix')
            self.U_k[k] = u_mat.copy()

            if wfs.mode == 'lcao':
                kpt.C_nM[:n_occ] = u_mat.T @ self.c_knm[k][:n_occ]
                wfs.atomic_correction.calculate_projections(wfs, kpt)
            else:
                kpt.psit_nG[:n_occ] = \
                    np.tensordot(u_mat.T, self.psit_knG[k][:n_occ],
                                 axes=1)
                # calc projectors
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
            evals_k[k] = evals
            evecs_k[k] = evecs
            del u_mat

        wfs.timer.start('Energy and gradients')
        e_sic, g_k = \
            self.odd_pot.get_energy_and_gradients_inner_loop2(
                wfs, a_k, evals_k, evecs_k, dens)
        self.eg_count += 1
        wfs.timer.stop('Energy and gradients')

        return e_sic, g_k


def log_f(log, niter, kappa, e_ks, e_sic, outer_counter=None, g_max=np.inf):

    t = time.localtime()

    if niter == 0:
        header0 = '\nINNER LOOP:\n'
        header = '                      Kohn-Sham          SIC' \
                 '        Total             \n' \
                 '           time         energy:      energy:' \
                 '      energy:       Error:       G_max:'
        log(header0 + header)

    if outer_counter is not None:
        niter = outer_counter

    log('iter: %3d  %02d:%02d:%02d ' %
        (niter,
         t[3], t[4], t[5]
         ), end='')
    log('%11.6f  %11.6f  %11.6f  %11.1e  %11.1e' %
        (Hartree * e_ks,
         Hartree * e_sic,
         Hartree * (e_ks + e_sic),
         kappa,
         Hartree * g_max), end='')
    log(flush=True)


def g_max_norm(g_k, wfs):
    # get maximum of gradients
    n_kps = wfs.kd.nks // wfs.kd.nspins

    max_norm = []
    for kpt in wfs.kpt_u:
        k = n_kps * kpt.s + kpt.q
        dim = g_k[k].shape[0]
        if dim == 0:
            continue
        max_norm.append(np.max(np.absolute(g_k[k])))
    max_norm = np.max(np.asarray(max_norm))
    g_max = wfs.world.max(max_norm)

    return g_max
