import numpy as np
from ase.parallel import parprint
import copy

from gpaw.utilities.blas import dotc

class LBFGSdirection:

    def __init__(self, wfs, m=20):
        """
        :param m: memory (amount of previous steps to use)
        """

        self.n_kps = wfs.kd.nks // wfs.kd.nspins

        # self.x_k = np.zeros(shape=n_d)
        # self.g_k = np.zeros(shape=n_d)

        # self.s_k = np.zeros(shape=(m, n_d))
        # self.y_k = np.zeros(shape=(m, n_d))

        self.s_k = {i: None for i in range(m)}
        self.y_k = {i: None for i in range(m)}

        self.rho_k = np.zeros(shape=m)

        self.kp = {}
        self.p = 0
        self.k = 0

        self.m = m

        self.stable = True

    def __str__(self):

        return 'LBFGS'

    def update_data(self, wfs, a_k, g_k1):

        if self.k == 0:

            self.kp[self.k] = self.p
            self.x_k = self.get_x(a_k)
            self.g_k = g_k1

            self.s_k[self.kp[self.k]] = self.zeros(g_k1)
            self.y_k[self.kp[self.k]] = self.zeros(g_k1)

            self.k += 1
            self.p += 1

            self.kp[self.k] = self.p

            p = self.minus(g_k1)

            return p

        else:

            if self.p == self.m:
                self.p = 0
                self.kp[self.k] = self.p

            s_k = self.s_k
            x_k = self.x_k
            y_k = self.y_k
            g_k = self.g_k

            x_k1 = self.get_x(a_k)

            rho_k = self.rho_k

            kp = self.kp
            k = self.k
            m = self.m

            s_k[kp[k]] = self.calc_diff(x_k1, x_k, wfs)
            y_k[kp[k]] = self.calc_diff(g_k1, g_k, wfs)

            try:
                dot_ys = self.dot_all_k_and_b(y_k[kp[k]],
                                              s_k[kp[k]],
                                              wfs)
                rho_k[kp[k]] = 1.0 / dot_ys
            except ZeroDivisionError:
                rho_k[kp[k]] = 1.0e12

            if rho_k[kp[k]] < 0.0:
                # raise Exception('y_k^Ts_k is not positive!')
                parprint("y_k^Ts_k is not positive!")
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

            try:
                # t = np.maximum(1, k - m + 1)

                t = k

                dot_yy = self.dot_all_k_and_b(y_k[kp[t]],
                                              y_k[kp[t]], wfs)

                r = self.multiply(q, 1.0 / (rho_k[kp[t]] * dot_yy))

            except ZeroDivisionError:
                # r = 1.0e12 * q
                r = self.multiply(q, 1.0e12)

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

            return self.multiply(r, const=-1.0)

    def get_x(self, a_k):

        x = {}
        for k in a_k.keys():
            x[k] = a_k[k].copy()

        return x

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
            dot_pr_x1x2 += 2.0 * dotc(x1[k], x2[k]).real

        dot_pr_x1x2 = wfs.kd.comm.sum(dot_pr_x1x2)

        return dot_pr_x1x2

    def multiply(self, x, const=1.0):

        y = {}
        for k in x.keys():
            y[k] = const * x[k]

        return y


class QuickMin:

    def __init__(self, wfs):

        """
        :param m: memory (amount of previous steps to use)
        """

        self.n_kps = wfs.kd.nks // wfs.kd.nspins

        self.dt = 0.1
        self.m = 0.1
        self.iters = 0

    def __str__(self):

        return 'QuickMin'

    def update_data(self, wfs, a_k, g_k1):

        if self.iters == 0:

            dt = self.dt
            m = self.m

            x = self.get_x(a_k)
            self.v = self.zeros(x)

            dot_gv = self.dot_all_k_and_b(g_k1, self.v, wfs)
            dot_gg = self.dot_all_k_and_b(g_k1, g_k1, wfs)

            alpha = (-dot_gv / dot_gg + dt / m)

            v_new = self.multiply(g_k1, -alpha)

            p = self.multiply(v_new, dt)

            self.iters += 1

            self.v = copy.deepcopy(v_new)

            return p

        else:

            dt = self.dt
            m = self.m

            dot_gv = self.dot_all_k_and_b(g_k1, self.v, wfs)
            dot_gg = self.dot_all_k_and_b(g_k1, g_k1, wfs)

            if dot_gv > 0.0:
                dot_gv = 0.0

            alpha = (-dot_gv / dot_gg + dt / m)

            v_new = self.multiply(g_k1, -alpha)

            p = self.multiply(v_new, dt)

            self.iters += 1

            self.v = copy.deepcopy(v_new)

            return p

    def get_x(self, a_k):

        x = {}
        for k in a_k.keys():
            x[k] = a_k[k].copy()

        return x

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
            dot_pr_x1x2 += 2.0 * dotc(x1[k], x2[k]).real

        dot_pr_x1x2 = wfs.kd.comm.sum(dot_pr_x1x2)

        return dot_pr_x1x2

    def multiply(self, x, const=1.0):

        y = {}
        for k in x.keys():
            y[k] = const * x[k]

        return y


class HZcg:

    """
    conjugate gradient method from paper of
    William W. Hager and Hongchao Zhang
    SIAM J. optim., 16(1), 170-192. (23 pages)
    """

    def __init__(self, wfs):

        self.n_kps = wfs.kd.nks // wfs.kd.nspins
        self.eta = 0.01
        self.iters = 0

    def __str__(self):

        return 'HZcg'

    def update_data(self, wfs, a_k, g_k1):

        if self.iters == 0:
            self.p_k = self.minus(g_k1)

            #save the step
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
                                const=2.0*rho*norm2)

            beta_k = rho * \
                     self.dot_all_k_and_b(y1, g_k1, wfs)

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
            dot_pr_x1x2 += 2.0 * dotc(x1[k], x2[k]).real

        dot_pr_x1x2 = wfs.kd.comm.sum(dot_pr_x1x2)

        return dot_pr_x1x2


class LBFGSdirection_k_points:

    def __init__(self, wfs, m=20):

        """
        :param m: memory (amount of previous steps to use)
        """

        self.n_kps = wfs.kd.nks // wfs.kd.nspins
        self.k_LBFGSdirection = {}
        self.stable = True

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            self.k_LBFGSdirection[k] = LBFGSdirection_one_k_point(m)

    def __str__(self):

        return 'LBFGS'

    def update_data(self, wfs, a_k, g_k1):

        p = {}

        stability = []

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            p[k] = self.k_LBFGSdirection[k].update_data(a_k[k],
                                                        g_k1[k])
            stability.append(self.k_LBFGSdirection[k].stable)

        if False in stability:
            self.stable = False

        return p

    def get_x(self, a_k):

        x = {}
        for k in a_k.keys():
            x[k] = a_k[k].copy()

        return x

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
            dot_pr_x1x2 += 2.0 * dotc(x1[k], x2[k]).real

        dot_pr_x1x2 = wfs.kd.comm.sum(dot_pr_x1x2)

        return dot_pr_x1x2

    def multiply(self, x, const=1.0):

        y = {}
        for k in x.keys():
            y[k] = const * x[k]

        return y


class LBFGSdirection_one_k_point:

    def __init__(self, m=20):
        """
        :param m: memory (amount of previous steps to use)
        """

        self.s_k = {i: None for i in range(m)}
        self.y_k = {i: None for i in range(m)}

        self.rho_k = np.zeros(shape=m)

        self.kp = {}
        self.p = 0
        self.k = 0

        self.m = m

        self.stable = True

    def __str__(self):

        return 'LBFGS'

    def update_data(self, a_k, g_k1):

        if np.max(np.absolute(g_k1)) < 1.0e-10:
            return np.zeros_like(g_k1)

        if self.k == 0:

            self.kp[self.k] = self.p
            self.x_k = a_k.copy()
            self.g_k = g_k1.copy()

            self.s_k[self.kp[self.k]] = np.zeros_like(g_k1)
            self.y_k[self.kp[self.k]] = np.zeros_like(g_k1)

            self.k += 1
            self.p += 1

            self.kp[self.k] = self.p

            p = -g_k1.copy()

            return p

        else:

            if self.p == self.m:
                self.p = 0
                self.kp[self.k] = self.p

            s_k = self.s_k
            x_k = self.x_k
            y_k = self.y_k
            g_k = self.g_k

            x_k1 = a_k.copy()
            rho_k = self.rho_k

            kp = self.kp
            k = self.k
            m = self.m

            s_k[kp[k]] = x_k1 - x_k
            y_k[kp[k]] = g_k1 - g_k

            try:
                dot_ys = 2.0 * np.dot(y_k[kp[k]].conj(), s_k[kp[k]]).real
                rho_k[kp[k]] = 1.0 / dot_ys
            except ZeroDivisionError:
                rho_k[kp[k]] = 1.0e12

            if rho_k[kp[k]] < 0.0:
                # raise Exception('y_k^Ts_k is not positive!')
                parprint("y_k^Ts_k is not positive!")
                self.stable = False

            q = g_k1.copy()

            alpha = np.zeros(np.minimum(k + 1, m))
            j = np.maximum(-1, k - m)

            for i in range(k, j, -1):
                dot_sq = 2.0 * np.dot(s_k[kp[i]].conj(), q).real
                alpha[kp[i]] = rho_k[kp[i]] * dot_sq
                q -= alpha[kp[i]] * y_k[kp[i]]

            try:
                t = np.maximum(1, k - m + 1)
                dot_yy = 2.0 * np.dot(y_k[kp[t]].conj(), y_k[kp[t]]).real
                r = q / (rho_k[kp[t]] * dot_yy)

            except ZeroDivisionError:
                r = q * 1.0e7

            for i in range(np.maximum(0, k - m + 1), k + 1):
                dot_yr = 2.0 * np.dot(y_k[kp[i]].conj(), r).real
                beta = rho_k[kp[i]] * dot_yr
                r += s_k[kp[i]] * (alpha[kp[i]] - beta)

            # save this step:
            self.x_k = x_k1.copy()
            self.g_k = g_k1.copy()

            self.k += 1
            self.p += 1

            self.kp[self.k] = self.p

            return -1.0 * r


class FRcg:
    """
    The Fletcher-Reeves conj. grad. method
    See Jorge Nocedal and Stephen J. Wright 'Numerical
    Optimization' Second Edition, 2006 (p. 121)
    """

    def __init__(self,  wfs):
        """
        :param n_d: dimension of space
        """

        # self.x_k = np.zeros(shape=n_d)
        # self.g_k = np.zeros(shape=n_d)
        # self.p_k = np.zeros(shape=n_d)
        self.iters = 0
        self.n_kps = wfs.kd.nks // wfs.kd.nspins

    def __str__(self):
        return 'Fletcher-Reeves conjugate gradient method'

    def update_data(self, wfs, x_k1, g_k1):

        if self.iters == 0:
            self.p_k = self.minus(g_k1)

            #save the step
            self.g_k = self.get_x(g_k1)

            self.iters += 1

            return self.p_k
        else:

            dot_g_k1_g_k1 = self.dot_all_k_and_b(g_k1,
                                                 g_k1,
                                                 wfs)

            dot_g_g = self.dot_all_k_and_b(self.g_k,
                                           self.g_k,
                                           wfs)

            beta_k = dot_g_k1_g_k1 / dot_g_g

            self.p_k = self.calc_diff(self.p_k, g_k1, wfs, const_0=beta_k)

            # save this step
            self.g_k = self.get_x(g_k1)

            self.iters += 1

            return self.p_k

    def get_x(self, a_k):

        x = {}
        for k in a_k.keys():
            x[k] = a_k[k].copy()

        return x

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
            dot_pr_x1x2 += 2.0 * dotc(x1[k], x2[k]).real

        dot_pr_x1x2 = wfs.kd.comm.sum(dot_pr_x1x2)

        return dot_pr_x1x2

    def multiply(self, x, const=1.0):

        y = {}
        for k in x.keys():
            y[k] = const * x[k]

        return y


class LBFGSdirection_prec:

    def __init__(self, wfs, m=10, diag=False):
        """
        :param m: memory (amount of previous steps to use)
        """

        self.n_kps = wfs.kd.nks // wfs.kd.nspins

        # self.x_k = np.zeros(shape=n_d)
        # self.g_k = np.zeros(shape=n_d)

        # self.s_k = np.zeros(shape=(m, n_d))
        # self.y_k = np.zeros(shape=(m, n_d))

        self.s_k = {i: None for i in range(m)}
        self.y_k = {i: None for i in range(m)}

        self.rho_k = np.zeros(shape=m)

        self.kp = {}
        self.p = 0
        self.k = 0

        self.m = m

        self.stable = True

        self.beta_0 = 1.0

        self.diag = diag

    def __str__(self):

        return 'LBFGS_prec'

    def update_data(self, wfs, a_k, g_k1, heiss_1):

        if self.k == 0:

            self.kp[self.k] = self.p
            self.x_k = self.get_x(a_k)
            self.g_k = self.get_x(g_k1)

            self.s_k[self.kp[self.k]] = self.zeros(g_k1)
            self.y_k[self.kp[self.k]] = self.zeros(g_k1)

            self.k += 1
            self.p += 1

            self.kp[self.k] = self.p

            p = self.matrix_vector(heiss_1, g_k1, -1.0, diag=self.diag)

            self.beta_0 = 1.0

            # p = self.minus(g_k1)

            return p

        else:

            if self.p == self.m:
                self.p = 0
                self.kp[self.k] = self.p

            s_k = self.s_k
            x_k = self.x_k
            y_k = self.y_k
            g_k = self.g_k

            x_k1 = self.get_x(a_k)

            rho_k = self.rho_k

            kp = self.kp
            k = self.k
            m = self.m

            s_k[kp[k]] = self.calc_diff(x_k1, x_k, wfs)
            y_k[kp[k]] = self.calc_diff(g_k1, g_k, wfs)

            try:
                dot_ys = self.dot_all_k_and_b(y_k[kp[k]],
                                              s_k[kp[k]],
                                              wfs)
                rho_k[kp[k]] = 1.0 / dot_ys
            except ZeroDivisionError:
                rho_k[kp[k]] = 1.0e12

            if rho_k[kp[k]] < 0.0:
                # raise Exception('y_k^Ts_k is not positive!')
                parprint("y_k^Ts_k is not positive!")
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

            try:
                # t = np.maximum(1, k - m + 1)

                t = k

                dot_yy = self.dot_all_k_and_b(y_k[kp[t]],
                                              y_k[kp[t]], wfs)

                # r = self.multiply(q, 1.0 / (rho_k[kp[t]] * dot_yy))

                self.beta_0 = 1.0 / (rho_k[kp[t]] * dot_yy)

                r = self.matrix_vector(heiss_1, q, diag=self.diag)

            except ZeroDivisionError:
                # r = 1.0e12 * q
                r = self.multiply(q, 1.0e12)

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

    def get_x(self, a_k):

        x = {}
        for k in a_k.keys():
            x[k] = a_k[k].copy()

        return x

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
            dot_pr_x1x2 += dotc(x1[k], x2[k]).real

        dot_pr_x1x2 = wfs.kd.comm.sum(dot_pr_x1x2)

        return dot_pr_x1x2

    def multiply(self, x, const=1.0):

        y = {}
        for k in x.keys():
            y[k] = const * x[k]

        return y

    def matrix_vector(self, L, x, const=1.0, diag=False):

        y = {}
        for k in x.keys():
            if diag:
                if L[k].dtype == complex:
                    y[k] = const * (L[k].real * x[k].real +
                                    1.0j*L[k].imag * x[k].imag)
                else:
                     y[k] = const * L[k] * x[k]
            else:
                y[k] = const * L[k].dot(x[k])

        return y



