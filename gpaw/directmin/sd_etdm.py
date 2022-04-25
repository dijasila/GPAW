"""
Search directions in space of skew-hermitian matrices

LSR1 algorithm and application to excited states:
arXiv:2006.15922 [physics.chem-ph]
J. Chem. Theory Comput. 16, 6968 (2020).
https://pubs.acs.org/doi/10.1021/acs.jctc.0c00597
"""

import numpy as np
import copy


class SearchDirectionBase(object):
    def __init__(self):
        self.iters = 0
        self.kp = None
        self.p = None
        self.k = None
        super(SearchDirectionBase, self).__init__()

    def __str__(self):
        raise NotImplementedError('Search direction class needs string '
                                  'representation')

    def update_data(self, wfs, x_k1, g_k1, precond=None):
        raise NotImplementedError('Search direction class needs '
                                  '\'update_data\' method')

    def reset(self):
        self.iters = 0
        self.kp = {}
        self.p = 0
        self.k = 0


class SteepestDescent(SearchDirectionBase):
    """
    Steepest descent algorithm
    """

    def __init__(self):
        super(SteepestDescent, self).__init__()

        self.name = 'sd'
        self.type = 'steepest-descent'

    def __str__(self):
        return 'Steepest Descent algorithm'

    def todict(self):
        return {'name': self.name}

    def update_data(self, wfs, x_k1, g_k1, precond=None):

        if precond is None:
            p_k = minus(g_k1)
        else:
            p_k = apply_prec(precond, g_k1, -1.0)
        self.iters += 1
        return p_k


class FRcg(SteepestDescent):
    """
    The Fletcher-Reeves conj. grad. method
    See Jorge Nocedal and Stephen J. Wright 'Numerical
    Optimization' Second Edition, 2006 (p. 121)
    """

    def __init__(self):
        super(FRcg, self).__init__()
        self.name = 'fr-cg'
        self.type = 'conjugate-gradients'

    def __str__(self):
        return 'Fletcher-Reeves conjugate gradient method'

    def todict(self):
        return {'name': self.name}

    def update_data(self, wfs, x_k1, g_k1, precond=None):

        if precond is not None:
            g_k1 = apply_prec(precond, g_k1, 1.0)

        if self.iters == 0:
            self.p_k = minus(g_k1)
        else:
            dot_g_k1_g_k1 = dot_all_k_and_b(g_k1, g_k1, wfs)
            dot_g_g = dot_all_k_and_b(self.g_k, self.g_k, wfs)
            beta_k = dot_g_k1_g_k1 / dot_g_g
            self.p_k = calc_diff(self.p_k, g_k1, beta_k)

        # save this step
        self.g_k = copy.deepcopy(g_k1)

        self.iters += 1
        return self.p_k


class QuickMin(SearchDirectionBase):

    def __init__(self, dt=0.01, mass=0.01):
        super(QuickMin, self).__init__()
        self.dt = dt
        self.mass = mass
        self.name = 'quick-min'
        self.type = 'equation-of-motion'

    def __str__(self):

        return 'QuickMin'

    def todict(self):
        return {'name': self.name,
                'dt': self.dt,
                'mass': self.mass}

    def update_data(self, wfs, x_k1, g_k1, precond=None):

        if precond is not None:
            g_k1 = apply_prec(precond, g_k1, 1.0)

        dt = self.dt
        m = self.mass

        if self.iters == 0:
            self.v = multiply(g_k1, -dt / m)
            p = multiply(self.v, dt)
        else:
            dot_gv = dot_all_k_and_b(g_k1, self.v, wfs)
            dot_gg = dot_all_k_and_b(g_k1, g_k1, wfs)
            if dot_gv > 0.0:
                dot_gv = 0.0
            gamma = (dt / m - dot_gv / dot_gg)
            self.v = multiply(g_k1, -gamma)
            p = multiply(self.v, dt)

        self.iters += 1
        return p


class LBFGS(SearchDirectionBase):

    def __init__(self, memory=3):
        """
        :param memory: amount of previous steps to use
        """
        super(LBFGS, self).__init__()

        self.s_k = {i: None for i in range(memory)}
        self.y_k = {i: None for i in range(memory)}

        self.rho_k = np.zeros(shape=memory)

        self.kp = {}
        self.p = 0
        self.k = 0

        self.memory = memory
        self.stable = True
        self.name = 'l-bfgs'
        self.type = 'quasi-newton'

    def __str__(self):

        return 'L-BFGS'

    def todict(self):
        return {'name': self.name,
                'memory': self.memory}

    def update_data(self, wfs, x_k1, g_k1, precond=None):

        self.iters += 1

        if precond is not None:
            g_k1 = apply_prec(precond, g_k1, 1.0)

        if self.k == 0:

            self.kp[self.k] = self.p
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = g_k1
            self.s_k[self.kp[self.k]] = zeros(g_k1)
            self.y_k[self.kp[self.k]] = zeros(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p

            return minus(g_k1)

        else:

            if self.p == self.memory:
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
            m = self.memory

            s_k[kp[k]] = calc_diff(x_k1, x_k)
            y_k[kp[k]] = calc_diff(g_k1, g_k)
            dot_ys = dot_all_k_and_b(y_k[kp[k]], s_k[kp[k]], wfs)

            if abs(dot_ys) > 1.0e-15:
                rho_k[kp[k]] = 1.0 / dot_ys
            else:
                rho_k[kp[k]] = 1.0e15

            if dot_ys < 0.0:
                self.stable = False

            q = copy.deepcopy(g_k1)

            alpha = np.zeros(np.minimum(k + 1, m))
            j = np.maximum(-1, k - m)

            for i in range(k, j, -1):
                dot_sq = dot_all_k_and_b(s_k[kp[i]], q, wfs)

                alpha[kp[i]] = rho_k[kp[i]] * dot_sq

                q = calc_diff(q, y_k[kp[i]], const=alpha[kp[i]])

            t = k
            dot_yy = dot_all_k_and_b(y_k[kp[t]], y_k[kp[t]], wfs)

            if abs(dot_yy) > 1.0e-15:
                r = multiply(q, 1.0 / (rho_k[kp[t]] * dot_yy))
            else:
                r = multiply(q, 1.0e15)

            for i in range(np.maximum(0, k - m + 1), k + 1):
                dot_yr = dot_all_k_and_b(y_k[kp[i]], r, wfs)

                beta = rho_k[kp[i]] * dot_yr

                r = calc_diff(r, s_k[kp[i]], const=(beta - alpha[kp[i]]))

            # save this step:
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p

            return multiply(r, -1.0)


class LBFGS_P(SearchDirectionBase):

    def __init__(self, memory=3, beta_0=1.0):
        """
        :param memory: amount of previous steps to use
        """
        super(LBFGS_P, self).__init__()
        self.s_k = {i: None for i in range(memory)}
        self.y_k = {i: None for i in range(memory)}
        self.rho_k = np.zeros(shape=memory)
        self.kp = {}
        self.p = 0
        self.k = 0
        self.memory = memory
        self.stable = True
        self.beta_0 = beta_0
        self.name = 'l-bfgs-p'
        self.type = 'quasi-newton'

    def __str__(self):

        return 'L-BFGS-P'

    def todict(self):

        return {'name': self.name,
                'memory': self.memory,
                'beta_0': self.beta_0}

    def update_data(self, wfs, x_k1, g_k1, hess_1=None):
        self.iters += 1
        if self.k == 0:
            self.kp[self.k] = self.p
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)
            self.s_k[self.kp[self.k]] = zeros(g_k1)
            self.y_k[self.kp[self.k]] = zeros(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p
            if hess_1 is None:
                p = minus(g_k1)
            else:
                p = apply_prec(hess_1, g_k1, -1.0)
            self.beta_0 = 1.0
            return p

        else:
            if self.p == self.memory:
                self.p = 0
                self.kp[self.k] = self.p

            s_k = self.s_k
            x_k = self.x_k
            y_k = self.y_k
            g_k = self.g_k
            rho_k = self.rho_k
            kp = self.kp
            k = self.k
            m = self.memory

            s_k[kp[k]] = calc_diff(x_k1, x_k)
            y_k[kp[k]] = calc_diff(g_k1, g_k)

            dot_ys = dot_all_k_and_b(y_k[kp[k]], s_k[kp[k]], wfs)

            if abs(dot_ys) > 1.0e-20:
                rho_k[kp[k]] = 1.0 / dot_ys
            else:
                rho_k[kp[k]] = 1.0e20

            if rho_k[kp[k]] < 0.0:
                self.stable = False
                self.__init__(memory=self.memory)
                return self.update_data(wfs, x_k1, g_k1, hess_1)

            q = copy.deepcopy(g_k1)

            alpha = np.zeros(np.minimum(k + 1, m))
            j = np.maximum(-1, k - m)

            for i in range(k, j, -1):
                dot_sq = dot_all_k_and_b(s_k[kp[i]], q, wfs)
                alpha[kp[i]] = rho_k[kp[i]] * dot_sq
                q = calc_diff(q, y_k[kp[i]], const=alpha[kp[i]])

            t = k
            dot_yy = dot_all_k_and_b(y_k[kp[t]], y_k[kp[t]], wfs)
            rhoyy = rho_k[kp[t]] * dot_yy
            if abs(rhoyy) > 1.0e-20:
                self.beta_0 = 1.0 / rhoyy
            else:
                self.beta_0 = 1.0e20
            
            if hess_1 is not None:
                r = apply_prec(hess_1, q)
            else:
                r = multiply(q, self.beta_0)

            for i in range(np.maximum(0, k - m + 1), k + 1):
                dot_yr = dot_all_k_and_b(y_k[kp[i]], r, wfs)
                beta = rho_k[kp[i]] * dot_yr
                r = calc_diff(r, s_k[kp[i]], const=(beta - alpha[kp[i]]))

            # save this step:
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)

            self.k += 1
            self.p += 1

            self.kp[self.k] = self.p

            return multiply(r, -1.0)


class LSR1P(SearchDirectionBase):
    """
    This class describes limited memory versions of
    SR-1, Powell and their combintaions (such as Bofill).
    """

    def __init__(self, memory=20, method='LSR1', phi=None):
        """
        :param memory: amount of previous steps to use
        """
        super(LSR1P, self).__init__()

        self.u_k = {i: None for i in range(memory)}
        self.j_k = {i: None for i in range(memory)}
        self.yj_k = np.zeros(shape=memory)
        self.method = method
        self.phi = phi

        self.phi_k = np.zeros(shape=memory)
        if self.phi is None:
            assert self.method in ['LSR1', 'LP', 'LBofill',
                                   'Linverse_Bofill'], 'Value Error'
            if self.method == 'LP':
                self.phi_k.fill(1.0)
        else:
            self.phi_k.fill(self.phi)

        self.kp = {}
        self.p = 0
        self.k = 0

        self.memory = memory
        self.name = 'l-sr1p'
        self.type = 'quasi-newton'

    def __str__(self):

        return 'LSR1P'

    def todict(self):

        return {'name': self.name,
                'memory': self.memory,
                'method': self.method}

    def update_data(self, wfs, x_k1, g_k1, precond=None):

        if precond is not None:
            bg_k1 = apply_prec(precond, g_k1, 1.0)
        else:
            bg_k1 = g_k1.copy()

        if self.k == 0:
            self.kp[self.k] = self.p
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)
            self.u_k[self.kp[self.k]] = zeros(g_k1)
            self.j_k[self.kp[self.k]] = zeros(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p
        else:
            if self.p == self.memory:
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
            m = self.memory

            s_k = calc_diff(x_k1, x_k)
            y_k = calc_diff(g_k1, g_k)
            if precond is not None:
                by_k = apply_prec(precond, y_k, 1.0)
            else:
                by_k = y_k.copy()

            by_k = self.update_bv(wfs, by_k, y_k, u_k, j_k, yj_k, phi_k,
                                  np.maximum(1, k - m), k)

            j_k[kp[k]] = calc_diff(s_k, by_k)
            yj_k[kp[k]] = dot_all_k_and_b(y_k, j_k[kp[k]], wfs)

            if self.method == 'LSR1':
                if abs(yj_k[kp[k]]) < 1e-12:
                    yj_k[kp[k]] = 1e-12

            dot_yy = dot_all_k_and_b(y_k, y_k, wfs)
            if abs(dot_yy) > 1.0e-15:
                u_k[kp[k]] = multiply(y_k, 1.0 / dot_yy)
            else:
                u_k[kp[k]] = multiply(y_k, 1.0e15)

            if self.method == 'LBofill' and self.phi is None:
                jj_k = dot_all_k_and_b(j_k[kp[k]], j_k[kp[k]], wfs)
                phi_k[kp[k]] = 1 - yj_k[kp[k]]**2 / (dot_yy * jj_k)
            elif self.method == 'Linverse_Bofill' and self.phi is None:
                jj_k = dot_all_k_and_b(j_k[kp[k]], j_k[kp[k]], wfs)
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
        return multiply(bg_k1, -1.0)

    def update_bv(self, wfs, bv, v, u_k, j_k, yj_k, phi_k, i_0, i_m):
        kp = self.kp

        for i in range(i_0, i_m):
            dot_uv = dot_all_k_and_b(u_k[kp[i]], v, wfs)
            dot_jv = dot_all_k_and_b(j_k[kp[i]], v, wfs)

            alpha = dot_jv - yj_k[kp[i]] * dot_uv
            beta_p = calc_diff(j_k[kp[i]], u_k[kp[i]], dot_uv, -alpha)

            beta_ms = multiply(j_k[kp[i]], dot_jv / yj_k[kp[i]])

            beta = calc_diff(beta_ms, beta_p, 1 - phi_k[kp[i]], -phi_k[kp[i]])

            bv = calc_diff(bv, beta, const=-1.0)

        return bv


def multiply(x, const=1.0):
    """
    it must not change x!
    :param x:
    :param const:
    :return: new dictionary y = cons*x
    """
    y = {}
    for k in x:
        y[k] = const * x[k]
    return y


def zeros(x):
    y = {}
    for k in x.keys():
        y[k] = np.zeros_like(x[k])
    return y


def minus(x):
    return multiply(x, -1.0)


def calc_diff(x1, x2, const_0=1.0, const=1.0):
    y_k = {}
    for k in x1.keys():
        y_k[k] = const_0 * x1[k] - const * x2[k]
    return y_k


def dot_all_k_and_b(x1, x2, wfs):
    dot_pr_x1x2 = 0.0
    for k in x1.keys():
        dot_pr_x1x2 += np.dot(x1[k].conj(), x2[k]).real
    dot_pr_x1x2 = wfs.kd.comm.sum(dot_pr_x1x2)
    return dot_pr_x1x2


def apply_prec(prec, x, const=1.0):
    y = {}
    for k in x.keys():
        if prec[k].dtype == complex:
            y[k] = const * (prec[k].real * x[k].real +
                            1.0j * prec[k].imag * x[k].imag)
        else:
            y[k] = const * prec[k] * x[k]
    return y
