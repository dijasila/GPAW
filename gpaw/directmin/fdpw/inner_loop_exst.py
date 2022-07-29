"""
Optimization of orbitals
among occupied and a few virtual states
represented on a grid or with plane waves
in order to calculate and excited state

arXiv:2102.06542 [physics.comp-ph]
"""

from gpaw.directmin.fdpw.tools import get_n_occ, get_indices, expm_ed
from gpaw.directmin.lcao.sd_lcao import LSR1P
from gpaw.directmin.lcao.ls_lcao import UnitStepLength
from ase.units import Hartree
import numpy as np
import time


class InnerLoop:

    def __init__(self, odd_pot, wfs, nstates='all',
                 tol=1.0e-3, maxiter=50, maxstepxst=0.2,
                 g_tol=5.0e-4, useprec=False):

        self.odd_pot = odd_pot
        self.n_kps = wfs.kd.nibzkpts
        self.g_tol = g_tol / Hartree
        self.tol = tol
        self.dtype = wfs.dtype
        self.get_en_and_grad_iters = 0
        self.precond = {}
        # self.method = 'LBFGS'
        # self.line_search_method = 'AwcSwc'
        self.max_iter_line_search = 6
        self.n_counter = maxiter
        self.maxstep = maxstepxst
        self.eg_count = 0
        self.total_eg_count = 0
        self.run_count = 0
        self.U_k = {}
        self.Unew_k = {}
        self.e_total = 0.0
        self.n_occ = {}
        self.useprec = useprec
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            if nstates == 'all':
                self.n_occ[k] = wfs.bd.nbands
            elif nstates == 'occupied':
                self.n_occ[k] = get_n_occ(kpt)
            else:
                raise NotImplementedError
            self.U_k[k] = np.eye(self.n_occ[k], dtype=self.dtype)
            self.Unew_k[k] = np.eye(self.n_occ[k], dtype=self.dtype)

    def get_energy_and_gradients(self, a_k, wfs, dens, ham):
        """
        Energy E = E[A]. Gradients G_ij[A] = dE/dA_ij
        Returns E[A] and G[A] at psi = exp(A).T kpt.psi
        :param a_k: A
        :return:
        """

        g_k = {}
        self.e_total = 0.0
        self.kappa = 0.0
        evals = {}
        evecs = {}
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            n_occ = self.n_occ[k]
            if n_occ == 0:
                g_k[k] = np.zeros_like(a_k[k])
                continue
            wfs.timer.start('Unitary matrix')
            u_mat, evecs[k], evals[k] = expm_ed(a_k[k], evalevec=True)
            wfs.timer.stop('Unitary matrix')
            self.Unew_k[k] = u_mat.copy()
            kpt.psit_nG[:n_occ] = \
                np.tensordot(u_mat.T, self.psit_knG[k][:n_occ],
                             axes=1)

            # calc projectors
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

            del u_mat

        wfs.timer.start('Energy and gradients')
        g_k, e_inner, kappa1 = \
            self.odd_pot.get_energy_and_gradients_inner_loop(
                wfs, a_k, evals, evecs, dens, ham=ham)
        wfs.timer.stop('Energy and gradients')
        if kappa1 > self.kappa:
            self.kappa = kappa1
        self.e_total = e_inner

        self.kappa = wfs.kd.comm.max(self.kappa)
        # self.e_total = wfs.kd.comm.sum(self.e_total)
        self.eg_count += 1
        self.total_eg_count += 1

        return self.e_total, g_k

    def evaluate_phi_and_der_phi(self, a_k, p_k, n_dim, alpha,
                                 wfs, dens, ham,
                                 phi=None, g_k=None):
        """
        phi = f(x_k + alpha_k*p_k)
        der_phi = grad f(x_k + alpha_k*p_k) cdot p_k
        :return:  phi, der_phi, grad f(x_k + alpha_k*p_k)
        """
        if phi is None or g_k is None:
            x_k = {k: a_k[k] + alpha * p_k[k] for k in a_k.keys()}
            phi, g_k = \
                self.get_energy_and_gradients(x_k, wfs, dens, ham)
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

        p = self.sd.update_data(wfs, a, g, self.precond)
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
            small_random=True, ham=None):

        log = log
        self.run_count += 1
        self.counter = 0
        self.eg_count = 0
        self.odd_pot.momcounter = 1
        self.converged = False
        # initial things
        self.psit_knG = {}
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            n_occ = self.n_occ[k]
            self.psit_knG[k] = np.tensordot(self.U_k[k].T,
                                            kpt.psit_nG[:n_occ],
                                            axes=1)

        a_k = {}
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            d = self.n_occ[k]
            a_k[k] = np.zeros(shape=(d, d), dtype=self.dtype)

        self.sd = LSR1P(wfs, memory=50)
        self.ls = UnitStepLength(self.evaluate_phi_and_der_phi,
                                 max_step=self.maxstep)

        threelasten = []
        # get initial energy and gradients
        self.e_total, g_k = self.get_energy_and_gradients(a_k, wfs, dens, ham)
        threelasten.append(self.e_total)
        g_max = g_max_norm(g_k, wfs)
        if g_max < self.g_tol:
            self.converged = True
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                n_occ = self.n_occ[k]
                kpt.psit_nG[:n_occ] = np.tensordot(self.U_k[k].conj(),
                                                   self.psit_knG[k],
                                                   axes=1)
                # calc projectors
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

                self.U_k[k] = self.U_k[k] @ self.Unew_k[k]
            del self.psit_knG
            if outer_counter is None:
                return self.e_total, self.counter
            else:
                return self.e_total, outer_counter

        if self.odd_pot.restart:
            del self.psit_knG
            return 0.0, 0

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
            # esic = self.e_total
            esic = self.odd_pot.total_sic
            e_ks = self.odd_pot.eks
            log_f(log, self.counter, self.kappa, e_ks, esic,
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
            self.precond = self.update_preconditioning(wfs, self.useprec)

            # calculate search direction fot current As and Gs
            p_k = self.get_search_direction(a_k, g_k, wfs)

            # calculate derivative along the search direction
            phi_0, der_phi_0, g_k = \
                self.evaluate_phi_and_der_phi(a_k, p_k, None,
                                              0.0, wfs, dens,
                                              ham=ham,
                                              phi=phi_0, g_k=g_k)
            if self.counter > 1:
                phi_old = phi_0
                der_phi_old = der_phi_0

            # choose optimal step length along the search direction
            # also get energy and gradients for optimal step
            alpha, phi_0, der_phi_0, g_k = \
                self.ls.step_length_update(
                    a_k, p_k, None, wfs, dens, ham,
                    phi_0=phi_0, der_phi_0=der_phi_0,
                    phi_old=phi_old_2, der_phi_old=der_phi_old_2,
                    alpha_max=3.0, alpha_old=alpha, kpdescr=wfs.kd)

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

            if self.odd_pot.restart:
                break
            if alpha > 1.0e-10:
                # calculate new matrices at optimal step lenght
                a_k = {k: a_k[k] + alpha * p_k[k] for k in a_k.keys()}
                g_max = g_max_norm(g_k, wfs)

                # output
                self.counter += 1
                if outer_counter is not None:
                    outer_counter += 1
                if log is not None:
                    # esic = phi_0
                    esic = self.odd_pot.total_sic
                    e_ks = self.odd_pot.eks
                    log_f(
                        log, self.counter, self.kappa, e_ks, esic,
                        outer_counter, g_max)

                not_converged = \
                    g_max > self.g_tol and \
                    self.counter < self.n_counter
                if not g_max > self.g_tol:
                    self.converged = True
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
            n_occ = self.n_occ[k]
            kpt.psit_nG[:n_occ] = np.tensordot(self.U_k[k].conj(),
                                               self.psit_knG[k],
                                               axes=1)
            # calc projectors
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
            self.U_k[k] = self.U_k[k] @ self.Unew_k[k]

        del self.psit_knG
        if outer_counter is None:
            return self.e_total, self.counter
        else:
            return self.e_total, outer_counter

    def get_numerical_gradients(self, A_s, wfs, dens, ham, log,
                                eps=1.0e-5):
        # initial things
        self.psit_knG = {}
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            n_occ = self.n_occ[k]
            self.psit_knG[k] = np.tensordot(self.U_k[k].T,
                                            kpt.psit_nG[:n_occ],
                                            axes=1)

        dtype = self.dtype
        h = [eps, -eps]
        coef = [1.0, -1.0]
        Gr_n_x = {}
        Gr_n_y = {}
        E_0, G = self.get_energy_and_gradients(A_s, wfs, dens, ham)
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
                        grad_num[igr] *= 1.0 / (2.0 * eps)
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
                    log(igr + 1, 'out of ', dim_gr, 'for a', k, 'kpt')
                    log(flush=True)
                    A = A_s[k][i][j]
                    for l in range(2):
                        A_s[k][i][j] = A + h[l]
                        A_s[k][j][i] = -(A + h[l])
                        E = self.get_energy_and_gradients(
                            A_s, wfs, dens, ham)[0]
                        grad_num[igr] += E * coef[l]
                    grad_num[igr] *= 1.0 / (2.0 * eps)
                    A_s[k][i][j] = A
                    A_s[k][j][i] = -A
                    igr += 1

                Gr_n_x[k] = grad_num.copy()
                G[k] = G[k][iut]

            Gr_n = {k: (Gr_n_x[k]) for k in Gr_n_x.keys()}

        return G, Gr_n

    def get_numerical_hessian(self, A_s, wfs, dens, ham, log, eps=1.0e-5):

        # initial things
        self.psit_knG = {}
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            n_occ = self.n_occ[k]
            self.psit_knG[k] = np.tensordot(self.U_k[k].T,
                                            kpt.psit_nG[:n_occ],
                                            axes=1)

        dtype = self.dtype
        assert dtype is float
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
            hessian = np.zeros(shape=(dim_gr, dim_gr),
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
                    g = self.get_energy_and_gradients(A_s, wfs, dens, ham)[1]
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

    def update_preconditioning(self, wfs, use_prec):
        counter = 30
        if use_prec:
            if self.counter % counter == 0 or self.counter == 1:
                for kpt in wfs.kpt_u:
                    k = self.n_kps * kpt.s + kpt.q
                    hess = self.get_hessian(kpt)
                    if self.dtype is float:
                        self.precond[k] = np.zeros_like(hess)
                        for i in range(hess.shape[0]):
                            if abs(hess[i]) < 1.0e-4:
                                self.precond[k][i] = 1.0
                            else:
                                self.precond[k][i] = \
                                    1.0 / (hess[i].real)
                    else:
                        self.precond[k] = np.zeros_like(hess)
                        for i in range(hess.shape[0]):
                            if abs(hess[i]) < 1.0e-4:
                                self.precond[k][i] = 1.0 + 1.0j
                            else:
                                self.precond[k][i] = \
                                    1.0 / hess[i].real + \
                                    1.0j / hess[i].imag
                return self.precond
            else:
                return self.precond
        else:
            return None

    def get_hessian(self, kpt):

        f_n = kpt.f_n
        eps_n = kpt.eps_n
        il1 = get_indices(eps_n.shape[0], self.dtype)
        il1 = list(il1)

        hess = np.zeros(len(il1[0]), dtype=self.dtype)
        x = 0
        for l, m in zip(*il1):
            df = f_n[l] - f_n[m]
            hess[x] = -2.0 * (eps_n[l] - eps_n[m]) * df
            if self.dtype is complex:
                hess[x] += 1.0j * hess[x]
                if abs(hess[x]) < 1.0e-10:
                    hess[x] = 0.0 + 0.0j
            else:
                if abs(hess[x]) < 1.0e-10:
                    hess[x] = 0.0
            x += 1

        return hess


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
    n_kps = wfs.kd.nibzkpts

    max_norm = []
    for kpt in wfs.kpt_u:
        k = n_kps * kpt.s + kpt.q
        dim = g_k[k].shape[0]
        if dim == 0:
            max_norm.append(0.0)
        else:
            max_norm.append(np.max(np.absolute(g_k[k])))
    max_norm = np.max(np.asarray(max_norm))
    g_max = wfs.world.max(max_norm)

    return g_max
