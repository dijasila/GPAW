"""
Optimization methods for calculating
search directions in space of skew-hermitian matrices
Examples are Steepest Descent, Conjugate gradients, L-BFGS
"""


import numpy as np
import copy
# from gpaw.utilities.blas import dotc


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
        """
        update search direction

        :param wfs:
        :param x_k1:
        :param g_k1:
        :param precond:
        :return:
        """

        if precond is None:
            p_k = self.minus(g_k1)
        else:
            p_k = self.apply_prec(precond, g_k1, -1.0)
        self.iters += 1
        return p_k

    def zeros(self, x):
        """
        return zero vector
        :param x:
        :return: 0
        """

        y = {}
        for k in x.keys():
            y[k] = np.zeros_like(x[k])

        return y

    def minus(self, x):
        """
        :param x:
        :return: -x
        """

        p = {}

        for k in x.keys():
            p[k] = - x[k].copy()

        return p

    def calc_diff(self, x1, x2, wfs, const_0=1.0, const=1.0):
        """
        calculate difference between x1 and x2

        :param x1:
        :param x2:
        :param wfs:
        :param const_0:
        :param const:
        :return:
        """
        y_k = {}
        for kpt in wfs.kpt_u:
            y_k[self.n_kps * kpt.s + kpt.q] = \
                const_0 * x1[self.n_kps * kpt.s + kpt.q] - \
                const * x2[self.n_kps * kpt.s + kpt.q]

        return y_k

    def dot_all_k_and_b(self, x1, x2, wfs):
        """
       dot product between x1 and x2 over all k-points and bands

        :param x1:
        :param x2:
        :param wfs:
        :return:
        """

        dot_pr_x1x2 = 0.0

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            # dot_pr_x1x2 += 2.0 * dotc(x1[k], x2[k]).real
            dot_pr_x1x2 += np.dot(x1[k].conj(), x2[k]).real

        dot_pr_x1x2 = wfs.kd.comm.sum(dot_pr_x1x2)

        return dot_pr_x1x2

    def multiply(self, x, const=1.0):
        """

        :param x:
        :param const:
        :return: const * x
        """

        y = {}
        for k in x.keys():
            y[k] = const * x[k]

        return y

    def apply_prec(self, prec, x, const=1.0):
        """
        apply preconditioning to the gradient

        :param prec:
        :param x:
        :param const:
        :return:
        """

        y = {}
        for k in x.keys():
            if prec[k].ndim == 1:
                if prec[k].dtype == complex:
                    y[k] = const * (prec[k].real * x[k].real +
                                    1.0j * prec[k].imag * x[k].imag)
                else:
                    y[k] = const * prec[k] * x[k]
            else:
                y[k] = const * prec[k] @ x[k]

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
        """
        update search direction

        :param wfs:
        :param x_k1:
        :param g_k1:
        :param precond:
        :return:
        """

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


class LBFGS(SteepestDescent):
    """
    The limited-memory BFGS.
    See Jorge Nocedal and Stephen J. Wright 'Numerical
    Optimization' Second Edition, 2006 (p. 177)
    """

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
        """
        update search direction

        :param wfs:
        :param x_k1:
        :param g_k1:
        :param precond:
        :return:
        """

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
                # print("y_k^Ts_k is not positive!")
                self.stable = False
                self.__init__(wfs, memory=self.m)
                return self.update_data(wfs, x_k1, g_k1)

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

            t = k
            dot_yy = self.dot_all_k_and_b(y_k[kp[t]],
                                          y_k[kp[t]], wfs)

            rhoyy = rho_k[kp[t]] * dot_yy

            if abs(rhoyy) > 1.0e-20:
                r = self.multiply(q, 1.0 / (rhoyy))
            else:
                r = self.multiply(q, 1.0 * 1.0e20)

            for i in range(np.maximum(0, k - m + 1), k + 1):
                dot_yr = self.dot_all_k_and_b(y_k[kp[i]], r, wfs)

                beta = rho_k[kp[i]] * dot_yr

                r = self.calc_diff(r, s_k[kp[i]], wfs,
                                   const=(beta - alpha[kp[i]]))

            # save this step:
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p
            self.iters += 1

            return self.multiply(r, const=-1.0)


class LBFGS_P(SteepestDescent):
    """
       The limited-memory BFGS.
       See Jorge Nocedal and Stephen J. Wright 'Numerical
       Optimization' Second Edition, 2006 (p. 177)

       used with preconditioning
       """

    def __init__(self, wfs, memory=3):
        """
        :param m: memory (amount of previous steps to use)
        """
        super(LBFGS_P, self).__init__(wfs)

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

        """
        update search direction

        :param wfs:
        :param x_k1:
        :param g_k1:
        :param precond:
        :return:
        """

        if self.k == 0:

            self.kp[self.k] = self.p
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = g_k1

            self.s_k[self.kp[self.k]] = self.zeros(g_k1)
            self.y_k[self.kp[self.k]] = self.zeros(g_k1)

            self.k += 1
            self.p += 1

            self.kp[self.k] = self.p

            if precond is None:
                p = self.minus(g_k1)
            else:
                p = self.apply_prec(precond, g_k1, -1.0)

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
                # print("y_k^Ts_k is not positive!")
                self.stable = False
                self.__init__(wfs, memory=self.m)
                return self.update_data(wfs, x_k1, g_k1)

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

            t = k
            dot_yy = self.dot_all_k_and_b(y_k[kp[t]],
                                          y_k[kp[t]], wfs)

            rhoyy = rho_k[kp[t]] * dot_yy

            if precond is not None:
                r = self.apply_prec(precond, q)
            else:
                if abs(rhoyy) > 1.0e-20:
                    r = self.multiply(q, 1.0 / (rhoyy))
                else:
                    r = self.multiply(q, 1.0 * 1.0e20)

            for i in range(np.maximum(0, k - m + 1), k + 1):
                dot_yr = self.dot_all_k_and_b(y_k[kp[i]], r, wfs)

                beta = rho_k[kp[i]] * dot_yr

                r = self.calc_diff(r, s_k[kp[i]], wfs,
                                   const=(beta - alpha[kp[i]]))

            # save this step:
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p
            self.iters += 1

            return self.multiply(r, const=-1.0)
