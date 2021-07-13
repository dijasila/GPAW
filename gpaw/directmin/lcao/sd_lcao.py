"""
Search directions in space of skew-hermitian matrices

LSR1 algorithm and application to excited states:
arXiv:2006.15922 [physics.chem-ph]
J. Chem. Theory Comput. 16, 6968 (2020).
https://pubs.acs.org/doi/10.1021/acs.jctc.0c00597
"""

import numpy as np
import copy
from ase.parallel import parprint


class SteepestDescent(object):
    """
    Steepest descent algorithm
    """

    def __init__(self, wfs):
        """
        """
        self.n_kps = wfs.kd.nibzkpts
        self.iters = 0

    def __str__(self):
        return 'Steepest Descent algorithm'

    def update_data(self, wfs, x_k1, g_k1, precond=None):

        if precond is None:
            p_k = self.minus(g_k1)
        else:
            p_k = self.apply_prec(precond, g_k1, -1.0)
        self.iters += 1
        return p_k

    def zeros(self, x):

        y = {}
        for k in x.keys():
            y[k] = np.zeros_like(x[k])

        return y

    def minus(self, x):

        p = {}

        for k in x.keys():
            p[k] = - x[k].copy()

        return p

    def calc_diff(self, x1, x2, wfs, const_0=1.0, const=1.0):
        y_k = {}
        for kpt in wfs.kpt_u:
            y_k[self.n_kps * kpt.s + kpt.q] = \
                const_0 * x1[self.n_kps * kpt.s + kpt.q] - \
                const * x2[self.n_kps * kpt.s + kpt.q]

        return y_k

    def dot_all_k_and_b(self, x1, x2, wfs):

        dot_pr_x1x2 = 0.0

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            # dot_pr_x1x2 += 2.0 * dotc(x1[k], x2[k]).real
            dot_pr_x1x2 += np.dot(x1[k].conj(), x2[k]).real

        dot_pr_x1x2 = wfs.kd.comm.sum(dot_pr_x1x2)

        return dot_pr_x1x2

    def multiply(self, x, const=1.0):

        y = {}
        for k in x.keys():
            y[k] = const * x[k]

        return y

    def apply_prec(self, prec, x, const=1.0):

        y = {}
        for k in x.keys():
            if prec[k].dtype == complex:
                y[k] = const * (prec[k].real * x[k].real +
                                1.0j * prec[k].imag * x[k].imag)
            else:
                y[k] = const * prec[k] * x[k]

        return y


class FRcg(SteepestDescent):
    """
    The Fletcher-Reeves conj. grad. method
    See Jorge Nocedal and Stephen J. Wright 'Numerical
    Optimization' Second Edition, 2006 (p. 121)
    """

    def __init__(self, wfs):
        super(FRcg, self).__init__(wfs)

    def __str__(self):
        return 'Fletcher-Reeves conjugate gradient method'

    def update_data(self, wfs, x_k1, g_k1, precond=None):

        if precond is not None:
            g_k1 = self.apply_prec(precond, g_k1, 1.0)

        if self.iters == 0:
            self.p_k = self.minus(g_k1)
            # save the step
            self.g_k = copy.deepcopy(g_k1)
            self.iters += 1

            return self.p_k
        else:

            dot_g_k1_g_k1 = self.dot_all_k_and_b(g_k1, g_k1, wfs)
            dot_g_g = self.dot_all_k_and_b(self.g_k, self.g_k, wfs)
            beta_k = dot_g_k1_g_k1 / dot_g_g

            self.p_k = self.calc_diff(self.p_k, g_k1, wfs,
                                      const_0=beta_k)
            # save this step
            self.g_k = copy.deepcopy(g_k1)
            self.iters += 1

            return self.p_k


class HZcg(SteepestDescent):

    """
    conjugate gradient method from paper of
    William W. Hager and Hongchao Zhang
    SIAM J. optim., 16(1), 170-192. (23 pages)
    """

    def __init__(self, wfs):

        super(HZcg, self).__init__(wfs)
        self.eta = 0.01

    def __str__(self):

        return 'Hager-Zhang conjugate gradient method'

    def update_data(self, wfs, x_k1, g_k1, precond=None):

        if precond is not None:
            g_k1 = self.apply_prec(precond, g_k1, 1.0)

        if self.iters == 0:
            self.p_k = self.minus(g_k1)
            # save the step
            self.g_k = g_k1
            self.iters += 1

            return self.p_k
        else:
            y_k = self.calc_diff(g_k1, self.g_k, wfs)
            try:
                dot_yp = self.dot_all_k_and_b(y_k, self.p_k, wfs)
                rho = 1.0 / dot_yp
            except ZeroDivisionError:
                rho = 1.0e10

            norm2 = self.dot_all_k_and_b(y_k, y_k, wfs)

            y1 = self.calc_diff(y_k, self.p_k, wfs,
                                const=2.0 * rho * norm2)

            beta_k = rho * self.dot_all_k_and_b(y1, g_k1, wfs)

            try:
                norm_p = np.sqrt(self.dot_all_k_and_b(self.p_k,
                                                      self.p_k, wfs))
                norm_g = np.sqrt(self.dot_all_k_and_b(self.g_k,
                                                      self.g_k, wfs))
                eta_k = - 1.0 / (norm_p * min(self.eta, norm_g))
            except ZeroDivisionError:
                eta_k = 1.0e10
            beta_k = max(beta_k, eta_k)
            self.p_k = self.calc_diff(self.p_k, g_k1, wfs, beta_k)
            # save this step
            self.g_k = g_k1
            self.iters += 1

            return self.p_k


class QuickMin(SteepestDescent):

    def __init__(self, wfs):
        super(QuickMin, self).__init__(wfs)
        self.dt = 0.01
        self.m = 0.01

    def __str__(self):

        return 'QuickMin'

    def update_data(self, wfs, x_k1, g_k1, precond=None):

        if precond is not None:
            g_k1 = self.apply_prec(precond, g_k1, 1.0)

        dt = self.dt
        m = self.m

        if self.iters == 0:
            self.v = self.multiply(g_k1, -dt / m)
            p = self.multiply(self.v, dt)
            self.iters += 1
            return p
        else:
            dot_gv = self.dot_all_k_and_b(g_k1, self.v, wfs)
            dot_gg = self.dot_all_k_and_b(g_k1, g_k1, wfs)
            if dot_gv > 0.0:
                dot_gv = 0.0
            gamma = (dt / m - dot_gv / dot_gg)
            self.v = self.multiply(g_k1, -gamma)
            p = self.multiply(self.v, dt)
            self.iters += 1
            return p


class LBFGS(SteepestDescent):

    def __init__(self, wfs, memory=3):
        """
        :param m: memory (amount of previous steps to use)
        """
        super(LBFGS, self).__init__(wfs)

        self.s_k = {i: None for i in range(memory)}
        self.y_k = {i: None for i in range(memory)}

        self.rho_k = np.zeros(shape=memory)

        self.kp = {}
        self.p = 0
        self.k = 0

        self.m = memory

        self.stable = True

    def __str__(self):

        return 'LBFGS'

    def update_data(self, wfs, x_k1, g_k1, precond=None):

        if precond is not None:
            g_k1 = self.apply_prec(precond, g_k1, 1.0)

        if self.k == 0:

            self.kp[self.k] = self.p
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = g_k1

            self.s_k[self.kp[self.k]] = self.zeros(g_k1)
            self.y_k[self.kp[self.k]] = self.zeros(g_k1)

            self.k += 1
            self.p += 1

            self.kp[self.k] = self.p

            p = self.minus(g_k1)
            self.iters += 1

            return p

        else:

            if self.p == self.m:
                self.p = 0
                self.kp[self.k] = self.p

            s_k = self.s_k
            x_k = self.x_k
            y_k = self.y_k
            g_k = self.g_k

            x_k1 = copy.deepcopy(x_k1)

            rho_k = self.rho_k

            kp = self.kp
            k = self.k
            m = self.m

            s_k[kp[k]] = self.calc_diff(x_k1, x_k, wfs)
            y_k[kp[k]] = self.calc_diff(g_k1, g_k, wfs)

            dot_ys = self.dot_all_k_and_b(y_k[kp[k]],
                                          s_k[kp[k]],
                                          wfs)
            if abs(dot_ys) > 1.0e-15:
                rho_k[kp[k]] = 1.0 / dot_ys
            else:
                rho_k[kp[k]] = 1.0e15

            # try:
            #     dot_ys = self.dot_all_k_and_b(y_k[kp[k]],
            #                                   s_k[kp[k]],
            #                                   wfs)
            #     rho_k[kp[k]] = 1.0 / dot_ys
            # except ZeroDivisionError:
            #     rho_k[kp[k]] = 1.0e12

            if dot_ys < 0.0:
                # raise Exception('y_k^Ts_k is not positive!')
                # print("y_k^Ts_k is not positive!")
                self.stable = False

            # q = np.copy(g_k1)
            q = copy.deepcopy(g_k1)

            alpha = np.zeros(np.minimum(k + 1, m))
            j = np.maximum(-1, k - m)

            for i in range(k, j, -1):
                dot_sq = self.dot_all_k_and_b(s_k[kp[i]],
                                              q, wfs)

                alpha[kp[i]] = rho_k[kp[i]] * dot_sq

                q = self.calc_diff(q, y_k[kp[i]],
                                   wfs, const=alpha[kp[i]])

                # q -= alpha[kp[i]] * y_k[kp[i]]

            t = k
            dot_yy = self.dot_all_k_and_b(y_k[kp[t]],
                                          y_k[kp[t]], wfs)
            if abs(dot_yy) > 1.0e-15:
                r = self.multiply(q, 1.0 / (rho_k[kp[t]] * dot_yy))
            else:
                r = self.multiply(q, 1.0e15)
            # try:
            #     # t = np.maximum(1, k - m + 1)
            #
            #     t = k
            #
            #     dot_yy = self.dot_all_k_and_b(y_k[kp[t]],
            #                                   y_k[kp[t]], wfs)
            #
            #     r = self.multiply(q, 1.0 / (rho_k[kp[t]] * dot_yy))
            #
            # except ZeroDivisionError:
            #     # r = 1.0e12 * q
            #     r = self.multiply(q, 1.0e12)

            for i in range(np.maximum(0, k - m + 1), k + 1):
                dot_yr = self.dot_all_k_and_b(y_k[kp[i]], r, wfs)

                beta = rho_k[kp[i]] * dot_yr

                r = self.calc_diff(r, s_k[kp[i]], wfs,
                                   const=(beta - alpha[kp[i]]))

                # r += s_k[kp[i]] * (alpha[kp[i]] - beta)

            # save this step:
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p
            self.iters += 1

            return self.multiply(r, const=-1.0)


class LBFGS_P(SteepestDescent):

    def __init__(self, wfs, memory=3):
        """
        :param m: memory (amount of previous steps to use)
        """
        super(LBFGS_P, self).__init__(wfs)
        # self.n_kps = wfs.kd.nibzkpts
        self.s_k = {i: None for i in range(memory)}
        self.y_k = {i: None for i in range(memory)}
        self.rho_k = np.zeros(shape=memory)
        self.kp = {}
        self.p = 0
        self.k = 0
        self.m = memory
        self.stable = True
        self.beta_0 = 1.0

    def __str__(self):

        return 'LBFGS_P'

    def update_data(self, wfs, x_k1, g_k1, hess_1=None):

        if self.k == 0:
            self.kp[self.k] = self.p
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)
            self.s_k[self.kp[self.k]] = self.zeros(g_k1)
            self.y_k[self.kp[self.k]] = self.zeros(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p
            if hess_1 is None:
                p = self.minus(g_k1)
            else:
                p = self.apply_prec(hess_1, g_k1, -1.0)
            self.beta_0 = 1.0
            return p

        else:

            if self.p == self.m:
                self.p = 0
                self.kp[self.k] = self.p

            s_k = self.s_k
            x_k = self.x_k
            y_k = self.y_k
            g_k = self.g_k

            # x_k1 = copy.deepcopy(x_k1)

            rho_k = self.rho_k

            kp = self.kp
            k = self.k
            m = self.m

            s_k[kp[k]] = self.calc_diff(x_k1, x_k, wfs)
            y_k[kp[k]] = self.calc_diff(g_k1, g_k, wfs)

            dot_ys = self.dot_all_k_and_b(y_k[kp[k]],
                                          s_k[kp[k]],
                                          wfs)
            if abs(dot_ys) > 1.0e-20:
                rho_k[kp[k]] = 1.0 / dot_ys
            else:
                rho_k[kp[k]] = 1.0e20

            # try:
            #     dot_ys = self.dot_all_k_and_b(y_k[kp[k]],
            #                                   s_k[kp[k]],
            #                                   wfs)
            #     rho_k[kp[k]] = 1.0 / dot_ys
            # except ZeroDivisionError:
            #     rho_k[kp[k]] = 1.0e12

            if rho_k[kp[k]] < 0.0:
                # raise Exception('y_k^Ts_k is not positive!')
                # parprint("y_k^Ts_k is not positive!")
                self.stable = False
                self.__init__(wfs, memory=self.m)
                return self.update_data(wfs, x_k1, g_k1, hess_1)

            # q = np.copy(g_k1)
            q = copy.deepcopy(g_k1)

            alpha = np.zeros(np.minimum(k + 1, m))
            j = np.maximum(-1, k - m)

            for i in range(k, j, -1):
                dot_sq = self.dot_all_k_and_b(s_k[kp[i]],
                                              q, wfs)

                alpha[kp[i]] = rho_k[kp[i]] * dot_sq

                q = self.calc_diff(q, y_k[kp[i]],
                                   wfs, const=alpha[kp[i]])

                # q -= alpha[kp[i]] * y_k[kp[i]]

            t = k
            dot_yy = self.dot_all_k_and_b(y_k[kp[t]],
                                          y_k[kp[t]], wfs)

            rhoyy = rho_k[kp[t]] * dot_yy
            if hess_1 is not None:
                r = self.apply_prec(hess_1, q)
            else:
                if abs(rhoyy) > 1.0e-20:
                    self.beta_0 = 1.0 / rhoyy
                else:
                    self.beta_0 = 1.0e20
                r = self.multiply(q, self.beta_0)

            for i in range(np.maximum(0, k - m + 1), k + 1):
                dot_yr = self.dot_all_k_and_b(y_k[kp[i]], r, wfs)

                beta = rho_k[kp[i]] * dot_yr

                r = self.calc_diff(r, s_k[kp[i]], wfs,
                                   const=(beta - alpha[kp[i]]))

                # r += s_k[kp[i]] * (alpha[kp[i]] - beta)

            # save this step:
            del s_k
            del x_k
            del y_k
            del g_k
            del rho_k

            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)

            self.k += 1
            self.p += 1

            self.kp[self.k] = self.p

            del q

            return self.multiply(r, const=-1.0)


class LBFGS_P2(SteepestDescent):

    def __init__(self, wfs, memory=3):
        """
        :param m: memory (amount of previous steps to use)
        """
        super(LBFGS_P2, self).__init__(wfs)
        self.n_kps = wfs.kd.nibzkpts
        self.s_k = {i: None for i in range(memory)}
        self.y_k = {i: None for i in range(memory)}
        self.rho_k = np.zeros(shape=memory)
        self.kp = {}
        self.p = 0
        self.k = 0
        self.m = memory
        self.stable = True
        self.beta_0 = 1.0

    def __str__(self):

        return 'LBFGS_P2'

    def update_data(self, wfs, x_k1, g_k1, hess_1=None):

        if self.k == 0:
            self.kp[self.k] = self.p
            self.g_k = copy.deepcopy(g_k1)
            self.s_k[self.kp[self.k]] = self.zeros(g_k1)
            self.y_k[self.kp[self.k]] = self.zeros(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p
            p = self.apply_prec(hess_1, g_k1, -1.0)
            self.beta_0 = 1.0
            return p

        else:

            if self.p == self.m:
                self.p = 0
                self.kp[self.k] = self.p

            y_k = self.y_k
            g_k = self.g_k
            rho_k = self.rho_k

            kp = self.kp
            k = self.k
            m = self.m

            self.s_k[kp[k]] = x_k1
            s_k = self.s_k

            y_k[kp[k]] = self.calc_diff(g_k1, g_k, wfs)

            try:
                dot_ys = self.dot_all_k_and_b(y_k[kp[k]],
                                              s_k[kp[k]],
                                              wfs)
                rho_k[kp[k]] = 1.0 / dot_ys
            except ZeroDivisionError:
                rho_k[kp[k]] = 1.0e12

            if dot_ys < 0.0:
                # raise Exception('y_k^Ts_k is not positive!')
                parprint("y_k^Ts_k is not positive!")
                self.stable = False
                self.__init__(wfs, memory=self.m)
                return self.update_data(wfs, x_k1, g_k1, hess_1)

            # q = np.copy(g_k1)
            q = copy.deepcopy(g_k1)

            alpha = np.zeros(np.minimum(k + 1, m))
            j = np.maximum(-1, k - m)

            for i in range(k, j, -1):
                dot_sq = self.dot_all_k_and_b(s_k[kp[i]],
                                              q, wfs)
                alpha[kp[i]] = rho_k[kp[i]] * dot_sq
                q = self.calc_diff(q, y_k[kp[i]],
                                   wfs, const=alpha[kp[i]])

            try:
                t = k
                dot_yy = self.dot_all_k_and_b(y_k[kp[t]],
                                              y_k[kp[t]], wfs)

                self.beta_0 = 1.0 / (rho_k[kp[t]] * dot_yy)
                r = self.apply_prec(hess_1, q)

            except ZeroDivisionError:
                r = self.multiply(q, 1.0e12)

            for i in range(np.maximum(0, k - m + 1), k + 1):
                dot_yr = self.dot_all_k_and_b(y_k[kp[i]], r, wfs)
                beta = rho_k[kp[i]] * dot_yr
                r = self.calc_diff(r, s_k[kp[i]], wfs,
                                   const=(beta - alpha[kp[i]]))

            # save this step:
            del s_k
            del y_k
            del g_k
            del rho_k
            del q

            self.g_k = copy.deepcopy(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p

            return self.multiply(r, const=-1.0)


class LSR1P(SteepestDescent):

    def __init__(self, wfs, memory=10, method='LSR1', phi=None):
        """
        :param m: memory (amount of previous steps to use)
        """
        super(LSR1P, self).__init__(wfs)

        self.u_k = {i: None for i in range(memory)}
        self.j_k = {i: None for i in range(memory)}
        self.yj_k = np.zeros(shape=memory)
        self.method = method
        self.phi = phi

        self.phi_k = np.zeros(shape=memory)
        if self.phi is None:
            assert self.method in ['LSR1', 'LP',
                                   'LBofill', 'Linverse_Bofill'], \
                'Value Error'
            if self.method == 'LP':
                self.phi_k.fill(1.0)
        else:
            self.phi_k.fill(self.phi)

        self.kp = {}
        self.p = 0
        self.k = 0

        self.m = memory

    def __str__(self):

        return 'LSR1P'

    def update_data(self, wfs, x_k1, g_k1, precond=None):

        if precond is not None:
            bg_k1 = self.apply_prec(precond, g_k1, 1.0)
        else:
            bg_k1 = g_k1.copy()

        if self.k == 0:
            self.kp[self.k] = self.p
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)
            self.u_k[self.kp[self.k]] = self.zeros(g_k1)
            self.j_k[self.kp[self.k]] = self.zeros(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p
            p = self.minus(bg_k1)
            self.iters += 1

            return p

        else:
            if self.p == self.m:
                self.p = 0
                self.kp[self.k] = self.p

            x_k = self.x_k
            g_k = self.g_k
            u_k = self.u_k
            j_k = self.j_k
            yj_k = self.yj_k
            phi_k = self.phi_k

            x_k1 = copy.deepcopy(x_k1)

            kp = self.kp
            k = self.k
            m = self.m

            s_k = self.calc_diff(x_k1, x_k, wfs)
            y_k = self.calc_diff(g_k1, g_k, wfs)
            if precond is not None:
                by_k = self.apply_prec(precond, y_k, 1.0)
            else:
                by_k = y_k.copy()

            by_k = self.update_bv(wfs, by_k, y_k, u_k, j_k, yj_k, phi_k,
                                  np.maximum(1, k - m), k)

            j_k[kp[k]] = self.calc_diff(s_k, by_k, wfs)
            yj_k[kp[k]] = self.dot_all_k_and_b(y_k, j_k[kp[k]], wfs)

            if self.method == 'LSR1':
                if abs(yj_k[kp[k]]) < 1e-12:
                    yj_k[kp[k]] = 1e-12

            dot_yy = self.dot_all_k_and_b(y_k, y_k, wfs)
            if abs(dot_yy) > 1.0e-15:
                u_k[kp[k]] = self.multiply(y_k, 1.0 / dot_yy)
            else:
                u_k[kp[k]] = self.multiply(y_k, 1.0e15)

            if self.method == 'LBofill' and self.phi is None:
                jj_k = self.dot_all_k_and_b(j_k[kp[k]], j_k[kp[k]], wfs)
                phi_k[kp[k]] = 1 - yj_k[kp[k]]**2 / (dot_yy * jj_k)
            elif self.method == 'Linverse_Bofill' and self.phi is None:
                jj_k = self.dot_all_k_and_b(j_k[kp[k]], j_k[kp[k]], wfs)
                phi_k[kp[k]] = yj_k[kp[k]] ** 2 / (dot_yy * jj_k)

            bg_k1 = self.update_bv(wfs, bg_k1, g_k1, u_k, j_k, yj_k, phi_k,
                                   np.maximum(1, k - m + 1), k + 1)

            # save this step:
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p
            self.iters += 1

        return self.multiply(bg_k1, const=-1.0)

    def update_bv(self, wfs, bv, v, u_k, j_k, yj_k, phi_k, i_0, i_m):
        kp = self.kp

        for i in range(i_0, i_m):
            dot_uv = self.dot_all_k_and_b(u_k[kp[i]],
                                          v, wfs)
            dot_jv = self.dot_all_k_and_b(j_k[kp[i]],
                                          v, wfs)

            alpha = dot_jv - yj_k[kp[i]] * dot_uv
            beta_p = self.calc_diff(j_k[kp[i]], u_k[kp[i]],
                                    wfs, const_0=dot_uv,
                                    const=-alpha)

            beta_ms = self.multiply(j_k[kp[i]], dot_jv / yj_k[kp[i]])

            beta = self.calc_diff(beta_ms, beta_p, wfs,
                                  const_0=1 - phi_k[kp[i]],
                                  const=-phi_k[kp[i]])

            bv = self.calc_diff(bv, beta, wfs, const=-1.0)

        return bv
