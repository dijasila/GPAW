from gpaw.directmin.tools import get_n_occ, get_indices
from gpaw.directmin.tools import expm_ed
from gpaw.directmin.fd.sd_inner import LBFGS
from gpaw.directmin.fd.ls_inner import StrongWolfeConditions as SWC
from ase.units import Hartree
import numpy as np
import time


class InnerLoop:

    def __init__(self, odd_pot, wfs, g_tol=5.0e-4):

        self.odd_pot = odd_pot
        self.n_kps = wfs.kd.nks // wfs.kd.nspins
        self.g_tol = g_tol / Hartree
        self.dtype = wfs.dtype
        self.get_en_and_grad_iters = 0
        self.method = 'LBFGS'
        self.line_search_method = 'AwcSwc'
        self.max_iter_line_search = 3
        self.n_counter = 15
        self.eg_count = 0
        self.run_count = 0

        self.n_occ = {}
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            self.n_occ[k] = get_n_occ(kpt)

    def get_energy_and_gradients(self, a_k, wfs, dens):
        """
        Energy E = E[A]. Gradients G_ij[A] = dE/dA_ij
        Returns E[A] and G[A] at psi = exp(A).T kpt.psi
        :param a_k: A
        :return:
        """
        g_k = {}
        e_total = 0.0

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            n_occ = self.n_occ[k]
            if n_occ == 0:
                g_k[k] = np.zeros_like(a_k[k])
                continue
            wfs.timer.start('Unitary matrix')
            u_mat, evecs, evals = expm_ed(a_k[k], evalevec=True)
            wfs.timer.stop('Unitary matrix')

            kpt.psit_nG[:n_occ] = \
                np.tensordot(u_mat.T, self.psit_knG[k][:n_occ],
                             axes=1)

            # calc projectors
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

            del u_mat
            wfs.timer.start('Energy and gradients')
            g_k[k], e_sic = \
                self.odd_pot.get_energy_and_gradients_inner_loop(
                    wfs, kpt, a_k[k], evals, evecs, dens)
            wfs.timer.stop('Energy and gradients')

            e_total += e_sic
        e_total = wfs.kd.comm.sum(e_total)

        self.eg_count += 1

        return e_total, g_k

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

    def run(self, e_ks, psit_knG, wfs, dens, log, outer_counter=0):
        self.run_count += 1

        counter = 0
        # initial things
        self.psit_knG = psit_knG
        a_k = {}
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            d = self.n_occ[k]
            # a_k[k] = np.zeros(shape=(d, d), dtype=self.dtype)
            if self.run_count == 1 and self.dtype is complex:
                a = 0.01 * np.random.rand(d, d) * 1.0j
                # a = np.zeros(shape=(d, d), dtype=self.dtype)
                wfs.gd.comm.broadcast(a, 0)
                a_k[k] = a
            else:
                a_k[k] = np.zeros(shape=(d, d), dtype=self.dtype)

        self.sd = LBFGS(wfs, memory=3)
        self.ls = SWC(
            self.evaluate_phi_and_der_phi,
            method=self.method, awc=True,
            max_iter=self.max_iter_line_search)

        # get initial energy and gradients
        e_total, g_k = self.get_energy_and_gradients(a_k, wfs, dens)

        # get maximum of gradients
        max_norm = []
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            if self.n_occ[k] == 0:
                continue
            max_norm.append(np.max(np.absolute(g_k[k])))
        max_norm = np.max(np.asarray(max_norm))
        g_max = wfs.world.max(max_norm)

        # stuff which are needed for minim.
        phi_0 = e_total
        phi_old = None
        der_phi_old = None
        phi_old_2 = None
        der_phi_old_2 = None

        outer_counter += 1
        if log is not None:
            log_f(log, counter, g_max, e_ks, e_total, outer_counter)

        alpha = 1.0
        if g_max < 1.0e-8:
            not_converged = False
        else:
            not_converged = True

        # not_converged = \
        #     g_max > self.g_tol and counter < self.n_counter

        while not_converged:

            # calculate search direction fot current As and Gs
            p_k = self.get_search_direction(a_k, g_k, wfs)

            # calculate derivative along the search direction
            phi_0, der_phi_0, g_k = \
                self.evaluate_phi_and_der_phi(a_k, p_k,
                                              0.0, wfs, dens,
                                              phi=phi_0, g_k=g_k)
            if counter > 1:
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

                # get maximum of gradients
                max_norm = []
                for kpt in wfs.kpt_u:
                    k = self.n_kps * kpt.s + kpt.q
                    if self.n_occ[k] == 0:
                        continue
                    max_norm.append(np.max(np.absolute(g_k[k])))
                max_norm = np.max(np.asarray(max_norm))
                g_max = wfs.world.max(max_norm)
                # output
                counter += 1
                if outer_counter is not None:
                    outer_counter += 1
                if log is not None:
                    log_f(
                        log, counter, g_max, e_ks, phi_0,
                        outer_counter)

                not_converged = g_max > self.g_tol and \
                                counter < self.n_counter
                if not not_converged and counter < 2:
                    not_converged = True

            else:
                break

        if log is not None:
            log('INNER LOOP FINISHED.\n')
        del self.psit_knG
        if outer_counter is None:

            return counter
        else:
            return outer_counter

    def get_numerical_gradients(self, A_s, wfs, dens, log, eps=1.0e-5,
                                dtype=complex):
        h = [eps, -eps]
        coef = [1.0, -1.0]
        Gr_n_x = {}
        Gr_n_y = {}
        E_0, G = self.get_energy_and_gradients(A_s, wfs, dens)
        log("Estimating gradients using finite differences..")
        log(flush=True)
        if dtype == complex:
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                dim = A_s[k].shape[0]
                for z in range(2):
                    grad = np.zeros(shape=(dim * dim),
                                    dtype=self.dtype)
                    for i in range(dim):
                        for j in range(dim):
                            log(k, z, i, j)
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
                                grad[i * dim + j] += E * coef[l]
                            grad[i * dim + j] *= 1.0 / (2.0 * eps)
                            if i == j:
                                A_s[k][i][j] = A
                            else:
                                A_s[k][i][j] = A
                                A_s[k][j][i] = -np.conjugate(A)
                    if z == 0:
                        Gr_n_x[k] = grad[:].reshape(
                            int(np.sqrt(grad.shape[0])),
                            int(np.sqrt(grad.shape[0])))
                    else:
                        Gr_n_y[k] = grad[:].reshape(
                            int(np.sqrt(grad.shape[0])),
                            int(np.sqrt(grad.shape[0])))
            Gr_n = {k: (Gr_n_x[k] + 1.0j * Gr_n_y[k]) for k in
                    Gr_n_x.keys()}
        else:
            for kpt in self.wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                dim = A_s[k].shape[0]
                grad = np.zeros(shape=(dim * dim), dtype=self.dtype)
                for i in range(dim):
                    for j in range(dim):
                        log(k, i, j)
                        log(flush=True)
                        A = A_s[k][i][j]
                        for l in range(2):
                            if i == j:
                                A_s[k][i][j] = A
                            else:
                                A_s[k][i][j] = A + h[l]
                                A_s[k][j][i] = -(A + h[l])
                            E = self.get_energy_and_gradients(
                                A_s, wfs, dens)[0]
                            grad[i * dim + j] += E * coef[l]
                        grad[i * dim + j] *= 1.0 / (2.0 * eps)
                        if i == j:
                            A_s[k][i][j] = A
                        else:
                            A_s[k][i][j] = A
                            A_s[k][j][i] = -A
                Gr_n_x[k] = grad[:].reshape(
                    int(np.sqrt(grad.shape[0])),
                    int(np.sqrt(grad.shape[0])))
            Gr_n = {k: (Gr_n_x[k]) for k in Gr_n_x.keys()}
        return G, Gr_n


def log_f(log, niter, g_max, e_ks, e_sic, outer_counter=None):

    t = time.localtime()

    if niter == 0:
        header0 = 'INNER LOOP:\n'
        header = '                      Kohn-Sham          SIC' \
                 '        Total    ||g||_inf\n' \
                 '           time         energy:      energy:' \
                 '      energy:    gradients:'
        log(header0 + header)

    if outer_counter is not None:
        niter = outer_counter

    log('iter: %3d  %02d:%02d:%02d ' %
        (niter,
         t[3], t[4], t[5]
         ), end='')
    log('%11.6f  %11.6f  %11.6f  %11.1e' %
        (Hartree * e_ks,
         Hartree * e_sic,
         Hartree * (e_ks + e_sic),
         Hartree * g_max), end='')
    log(flush=True)
