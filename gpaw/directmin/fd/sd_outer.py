import numpy as np
import copy


class SteepestDescent:

    def __init__(self, wfs, dimensions):
        """
        """
        self.iters = 0
        self.n_kps = wfs.kd.nks // wfs.kd.nspins
        self.dimensions = dimensions

    def __str__(self):

        return 'Steepest Descent'

    def update_data(self, psi, g, wfs, prec):

        self.apply_prec(wfs, g, prec, 1.0)

        return self.minus(wfs, g)

    def dot(self, psi_1, psi_2, kpt, wfs):

        def S(psit_G):
            return psit_G

        def dS(a, P_ni):
            return np.dot(P_ni, wfs.setups[a].dO_ii)

        P1_ai = wfs.pt.dict(shape=1)
        P2_ai = wfs.pt.dict(shape=1)

        wfs.pt.integrate(psi_1, P1_ai, kpt.q)
        wfs.pt.integrate(psi_2, P2_ai, kpt.q)

        dot_prod = wfs.gd.integrate(psi_1, S(psi_2), False)

        paw_dot_prod = 0.0

        for a in P1_ai.keys():
            paw_dot_prod += \
                np.dot(dS(a, P2_ai[a]), P1_ai[a].T.conj())[0][0]

        sum_dot = dot_prod + paw_dot_prod
        sum_dot = wfs.gd.comm.sum(sum_dot)

        return sum_dot

    def dot_2(self, psi_1, psi_2, wfs):

        dot_prod = wfs.gd.integrate(psi_1, psi_2, False)
        dot_prod = wfs.gd.comm.sum(dot_prod)

        return dot_prod

    def dot_all_k_and_b(self, x1, x2, wfs):

        dot_pr_x1x2 = 0.0

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            for i in range(self.dimensions[k]):
                dot_pr_x1x2 += self.dot(x1[k][i],
                                        x2[k][i],
                                        kpt, wfs)

        dot_pr_x1x2 = wfs.kd.comm.sum(dot_pr_x1x2)

        return 2.0 * dot_pr_x1x2.real

    def calc_diff(self, x1, x2, wfs, const_0=1.0, const=1.0):
        y_k = {}
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            y_k[k] = \
                const_0 * x1[k] - \
                const * x2[k]

        return y_k

    def minus(self, wfs, x):

        p = {}
        for kpt in wfs.kpt_u:
            p[self.n_kps * kpt.s + kpt.q] = \
                - x[self.n_kps * kpt.s + kpt.q].copy()

        return p

    def multiply(self, x, const=1.0):

        y = {}
        for k in x.keys():
            y[k] = const * x[k]

        return y

    def zeros(self, x):

        y = {}
        for k in x.keys():
            y[k] = np.zeros_like(x[k])

        return y

    def apply_prec(self, wfs, x, prec, const=1.0):
        deg = (3.0 - wfs.kd.nspins)
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            for i, y in enumerate(x[k]):
                x[k][i] = - const * prec(y, kpt, None) / deg


class HZcg(SteepestDescent):

    """
    conjugate gradient method from paper of
    William W. Hager and Hongchao Zhang
    SIAM J. optim., 16(1), 170-192. (23 pages)
    """

    def __init__(self, wfs, dimensions):

        """
        """
        super().__init__(wfs, dimensions)
        self.eta = 0.01

    def __str__(self):
        return 'Hager-Zhang conjugate gradient method'

    def update_data(self, psi, g_k1, wfs, prec):

        if prec is not None:
            self.apply_prec(wfs, g_k1, prec, 1.0)

        if self.iters == 0:

            # first step is just minus gradients:
            p = {}
            for kpt in wfs.kpt_u:
                p[self.n_kps * kpt.s + kpt.q] = \
                    - g_k1[self.n_kps * kpt.s + kpt.q].copy()

            self.p_k = p
            # save the step
            self.g_k = g_k1
            self.iters += 1
            return self.p_k
        else:

            # calculate difference between current gradients
            # and previous one
            y_k = self.calc_diff(g_k1, self.g_k, wfs)

            # calculate dot product of
            # np.dot(y_k, p_k) with over all k-points
            dot_pr_yp = self.dot_all_k_and_b(y_k, self.p_k, wfs)

            try:
                rho = 1.0 / dot_pr_yp
            except ZeroDivisionError:
                rho = 1.0e10

            # calculate ||y_k||^2
            norm2 = self.dot_all_k_and_b(y_k, y_k, wfs)
            z = self.calc_diff(y_k, self.p_k,
                               wfs, const=2.0*rho*norm2)
            beta_k = rho * self.dot_all_k_and_b(z, g_k1, wfs)

            try:

                norm_p = np.sqrt(self.dot_all_k_and_b(self.p_k,
                                                      self.p_k, wfs))
                norm_g = np.sqrt(self.dot_all_k_and_b(self.g_k,
                                                      self.g_k, wfs))
                eta_k = - 1.0 / (norm_p * min(self.eta, norm_g))

            except ZeroDivisionError:
                eta_k = 1.0e10

            beta_k = max(beta_k, eta_k)
            self.p_k = self.calc_diff(g_k1, self.p_k, wfs,
                                      const_0=-1.0,
                                      const=-beta_k)
            # save this step
            self.g_k = g_k1
            self.iters += 1
            # if self.iters > 3:
            #     self.iters = 0

            return self.p_k


class FRcg(SteepestDescent):
    """
    The Fletcher-Reeves conj. grad. method
    See Jorge Nocedal and Stephen J. Wright 'Numerical
    Optimization' Second Edition, 2006 (p. 121)
    """

    def __init__(self, wfs, dimensions):

        """
        """
        super().__init__(wfs, dimensions)

    def __str__(self):
        return 'Fletcher-Reeves conjugate gradient method'

    def update_data(self, psi, g_k1, wfs, prec):

        if prec is not None:
            self.apply_prec(wfs, g_k1, prec, 1.0)

        if self.iters == 0:
            self.p_k = self.minus(wfs, g_k1)
            # save the step
            self.g_k = g_k1
            self.iters += 1
            return self.p_k
        else:
            dot_gg_k1 = self.dot_all_k_and_b(g_k1, g_k1, wfs)
            dot_gg_k = self.dot_all_k_and_b(self.g_k, self.g_k, wfs)
            beta_k = dot_gg_k1 / dot_gg_k
            self.p_k = self.calc_diff(g_k1, self.p_k, wfs,
                                      const_0=-1.0,
                                      const=-beta_k)
            # self.p_k = -g_k1 + beta_k * self.p_k
            # save this step
            self.g_k = g_k1
            self.iters += 1

            if self.iters > 10:
                self.iters = 0

            return self.p_k


class PRcg(SteepestDescent):

    """
    The Polak Ribiere c.gr.method
    Jorge Nocedal and Stephen J. Wright 'Numerical
    Optimization' Second Edition, 2006 (p. 122)
    """

    def __init__(self, wfs, dimensions):

        """
        """
        super().__init__(wfs, dimensions)

    def __str__(self):
        return 'Polak-Ribiere conjugate gradient method'

    def update_data(self, psi, g_k1, wfs, prec):

        if prec is not None:
            self.apply_prec(wfs, g_k1, prec, 1.0)

        if self.iters == 0:
            self.p_k = self.minus(wfs, g_k1)
            # save the step
            self.g_k = g_k1
            self.iters += 1
            return self.p_k
        else:
            dg = self.calc_diff(g_k1, self.g_k, wfs)
            dot_gdg = self.dot_all_k_and_b(g_k1, dg, wfs)
            dot_gg_k = self.dot_all_k_and_b(self.g_k, self.g_k, wfs)
            beta_k = dot_gdg / dot_gg_k
            self.p_k = self.calc_diff(g_k1, self.p_k, wfs,
                                      const_0=-1.0,
                                      const=-beta_k)
            # save this step
            self.g_k = g_k1
            self.iters += 1
            if self.iters > 3:
                self.iters = 0
            return self.p_k


class PRpcg(SteepestDescent):

    """
    The modified Polak-Ribiere c.gr.method
    Jorge Nocedal and Stephen J. Wright 'Numerical
    Optimization' Second Edition, 2006 (p. 122)
    """

    def __init__(self, wfs, dimensions):

        """
        """
        super().__init__(wfs, dimensions)

    def __str__(self):
        return 'Modified Polak-Ribiere conjugate gradient method'

    def update_data(self, psi, g_k1, wfs, prec):

        if prec is not None:
            self.apply_prec(wfs, g_k1, prec, 1.0)

        if self.iters == 0:
            self.p_k = self.minus(wfs, g_k1)
            # save the step
            self.g_k = g_k1
            self.iters += 1
            return self.p_k
        else:

            dg = self.calc_diff(g_k1, self.g_k, wfs)
            dot_gdg = self.dot_all_k_and_b(g_k1, dg, wfs)
            dot_gg_k = self.dot_all_k_and_b(self.g_k, self.g_k, wfs)
            beta_k = dot_gdg / dot_gg_k
            beta_k = max(beta_k, 0.0)
            self.p_k = self.calc_diff(g_k1, self.p_k, wfs,
                                      const_0=-1.0,
                                      const=-beta_k)
            # save this step
            self.g_k = g_k1
            self.iters += 1
            if self.iters > 3:
                self.iters = 0
            return self.p_k


class QuickMin(SteepestDescent):

    def __init__(self, wfs, dimensions):

        """
        """
        super().__init__(wfs, dimensions)

        self.dt = 0.01
        self.m = 0.01

    def __str__(self):

        return 'QuickMin'

    def update_data(self, psi, g_k1, wfs, prec):

        if prec is not None:
            self.apply_prec(wfs, g_k1, prec, 1.0)

        dt = self.dt
        m = self.m
        if self.iters == 0:
            self.v = self.multiply(g_k1, -dt/m)
            p = self.multiply(self.v, dt)
            self.iters += 1
            return p
        else:
            dot_gv = self.dot_all_k_and_b(g_k1, self.v, wfs)
            dot_gg = self.dot_all_k_and_b(g_k1, g_k1, wfs)
            if dot_gv > 0.0:
                dot_gv = 0.0
            alpha = (-dot_gv / dot_gg + dt / m)
            self.v = self.multiply(g_k1, -alpha)
            p = self.multiply(self.v, dt)
            self.iters += 1
            return p


class LBFGS(SteepestDescent):

    def __init__(self, wfs, dimensions, memory=1):

        """
        """
        super().__init__(wfs, dimensions)

        self.s_k = {i: None for i in range(memory)}
        self.y_k = {i: None for i in range(memory)}
        self.rho_k = np.zeros(shape=memory)

        self.kp = {}
        self.p = 0
        self.k = 0  # number of calls

        self.m = memory
        self.stable = True

    def __str__(self):

        return 'LBFGS'

    def update_data(self, psi, g_k1, wfs, prec):

        if prec is not None:
            self.apply_prec(wfs, g_k1, prec, 1.0)

        if self.k == 0:

            self.kp[self.k] = self.p
            self.x_k = psi
            self.g_k = g_k1
            self.s_k[self.kp[self.k]] = self.zeros(g_k1)
            self.y_k[self.kp[self.k]] = self.zeros(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p
            p = self.minus(wfs, g_k1)

            return p

        else:
            if self.p == self.m:
                self.p = 0
                self.kp[self.k] = self.p

            s_k = self.s_k
            x_k = self.x_k
            y_k = self.y_k
            g_k = self.g_k

            x_k1 = psi

            rho_k = self.rho_k

            kp = self.kp
            k = self.k
            m = self.m

            s_k[kp[k]] = self.calc_diff(x_k1, x_k, wfs)
            y_k[kp[k]] = self.calc_diff(g_k1, g_k, wfs)

            dot_ys = self.dot_all_k_and_b(y_k[kp[k]],
                                          s_k[kp[k]], wfs)
            if abs(dot_ys) > 0.0:
                rho_k[kp[k]] = 1.0 / dot_ys
            else:
                rho_k[kp[k]] = 1.0e16 * np.sign(dot_ys)

            if rho_k[kp[k]] < 0.0:
                # raise Exception('y_k^Ts_k is not positive!')
                self.stable = False
                self.__init__(wfs, self.dimensions, self.m)
                # we could call self.update,
                # but we already applied prec to g
                self.kp[self.k] = self.p
                self.x_k = x_k1
                self.g_k = g_k1
                self.s_k[self.kp[self.k]] = self.zeros(g_k1)
                self.y_k[self.kp[self.k]] = self.zeros(g_k1)
                self.k += 1
                self.p += 1
                self.kp[self.k] = self.p
                p = self.multiply(g_k1, -1.0)

                return p

            # q = np.copy(g_k1)
            q = copy.deepcopy(g_k1)

            alpha = np.zeros(np.minimum(k + 1, m))
            j = np.maximum(-1, k - m)

            for i in range(k, j, -1):

                dot_sq = self.dot_all_k_and_b(s_k[kp[i]], q, wfs)

                alpha[kp[i]] = rho_k[kp[i]] * dot_sq

                q = self.calc_diff(q, y_k[kp[i]],
                                   wfs, const=alpha[kp[i]])

                # q -= alpha[kp[i]] * y_k[kp[i]]

            try:
                t = np.maximum(1, k - m + 1)

                dot_yy = self.dot_all_k_and_b(y_k[kp[t]],
                                              y_k[kp[t]], wfs)

                r = self.multiply(q, 1.0 / (rho_k[kp[t]] * dot_yy))

                # r = q / (
                #       rho_k[kp[t]] * np.dot(y_k[kp[t]], y_k[kp[t]]))
            except ZeroDivisionError:
                # r = 1.0e12 * q
                r = self.multiply(q, 1.0e16)

            for i in range(np.maximum(0, k - m + 1), k + 1):

                dot_yr = self.dot_all_k_and_b(y_k[kp[i]], r, wfs)

                beta = rho_k[kp[i]] * dot_yr

                r = self.calc_diff(r, s_k[kp[i]], wfs,
                                   const=(beta-alpha[kp[i]]))

                # r += s_k[kp[i]] * (alpha[kp[i]] - beta)

            # save this step:
            self.x_k = x_k1
            self.g_k = g_k1

            self.k += 1
            self.p += 1

            self.kp[self.k] = self.p

            return self.multiply(r, const=-1.0)

#
# class LBFGSdirection_prec:
#
#     def __init__(self, wfs, precond, m=3):
#
#         """
#         :param m: memory (number of previous steps to use)
#         """
#
#         self.n_kps = wfs.kd.nks // wfs.kd.nspins
#         self.n_bands = wfs.bd.nbands
#
#         self.s_k = {i: None for i in range(m)}
#         self.y_k = {i: None for i in range(m)}
#         self.rho_k = np.zeros(shape=m)
#
#         self.kp = {}
#         self.p = 0
#         self.k = 0  # number of calls
#
#         self.m = m
#         self.stable = True
#         self.precond = precond
#
#         self.beta_0 = 1.0
#         self.alpha = 0.0
#
#     def __str__(self):
#
#         return 'L-BFGS'
#
#     def update_data(self, wfs, g_k1):
#
#         if self.k == 0:
#
#             self.kp[self.k] = self.p
#             self.x_k = self.get_wf(wfs)
#             self.g_k = copy.deepcopy(g_k1)
#
#             self.s_k[self.kp[self.k]] = self.zeros(g_k1)
#             self.y_k[self.kp[self.k]] = self.zeros(g_k1)
#
#             self.k += 1
#             self.p += 1
#
#             self.kp[self.k] = self.p
#
#             # p = self.minus(wfs, g_k1)
#             p = self.apply_prec(wfs, g_k1, -1.0)
#             return p
#
#         else:
#
#             if self.p == self.m:
#                 self.p = 0
#                 self.kp[self.k] = self.p
#
#             s_k = self.s_k
#             x_k = self.x_k
#             y_k = self.y_k
#             g_k = self.g_k
#
#             x_k1 = self.get_wf(wfs)
#
#             rho_k = self.rho_k
#
#             kp = self.kp
#             k = self.k
#             m = self.m
#
#             s_k[kp[k]] = self.calc_diff(x_k1, x_k, wfs)
#             y_k[kp[k]] = self.calc_diff(g_k1, g_k, wfs)
#
#             dot_ys = self.dot_all_k_and_b(y_k[kp[k]],
#                                           s_k[kp[k]], wfs)
#             if abs(dot_ys) > 0.0:
#                 rho_k[kp[k]] = 1.0 / dot_ys
#             else:
#                 rho_k[kp[k]] = 1.0e12
#
#             if rho_k[kp[k]] < 0.0:
#                 # raise Exception('y_k^Ts_k is not positive!')
#                 parprint("y_k^Ts_k is not positive!")
#                 self.stable = False
#
#             # q = np.copy(g_k1)
#             q = copy.deepcopy(g_k1)
#
#             alpha = np.zeros(np.minimum(k + 1, m))
#             j = np.maximum(-1, k - m)
#
#             for i in range(k, j, -1):
#
#                 dot_sq = self.dot_all_k_and_b(s_k[kp[i]], q, wfs)
#
#                 alpha[kp[i]] = rho_k[kp[i]] * dot_sq
#
#                 q = self.calc_diff(q, y_k[kp[i]],
#                                    wfs, const=alpha[kp[i]])
#
#                 # q -= alpha[kp[i]] * y_k[kp[i]]
#
#             try:
#                 t = np.maximum(1, k - m + 1)
#
#                 dot_yy = self.dot_all_k_and_b(y_k[kp[t]],
#                                               y_k[kp[t]], wfs)
#
#                 self.beta_0 = 1.0 / (rho_k[kp[t]] * dot_yy)
#                 r = self.apply_prec(wfs, q)
#                 # r = q / (
#                 #       rho_k[kp[t]] * np.dot(y_k[kp[t]], y_k[kp[t]]))
#                 # r = self.multiply(q, 1.0 / (rho_k[kp[t]] * dot_yy))
#
#             except ZeroDivisionError:
#                 # r = 1.0e12 * q
#                 r = self.multiply(q, 1.0e12)
#
#             for i in range(np.maximum(0, k - m + 1), k + 1):
#
#                 dot_yr = self.dot_all_k_and_b(y_k[kp[i]], r, wfs)
#
#                 beta = rho_k[kp[i]] * dot_yr
#
#                 r = self.calc_diff(r, s_k[kp[i]], wfs,
#                                    const=(beta-alpha[kp[i]]))
#
#                 # r += s_k[kp[i]] * (alpha[kp[i]] - beta)
#
#             # save this step:
#             self.x_k = copy.deepcopy(x_k1)
#             self.g_k = copy.deepcopy(g_k1)
#
#             self.k += 1
#             self.p += 1
#
#             self.kp[self.k] = self.p
#
#             return self.multiply(r, const=-1.0)
#
#     def dot(self, psi_1, psi_2, kpt, wfs):
#
#         def S(psit_G):
#             return psit_G
#
#         def dS(a, P_ni):
#             return np.dot(P_ni, wfs.setups[a].dO_ii)
#
#         P1_ai = wfs.pt.dict(shape=1)
#         P2_ai = wfs.pt.dict(shape=1)
#
#         wfs.pt.integrate(psi_1, P1_ai, kpt.q)
#         wfs.pt.integrate(psi_2, P2_ai, kpt.q)
#
#         dot_prod = wfs.gd.integrate(psi_1, S(psi_2), False)
#
#         paw_dot_prod = 0.0
#
#         for a in P1_ai.keys():
#             paw_dot_prod += np.dot(dS(a, P2_ai[a]), P1_ai[a].T.conj())[0][0]
#
#         sum_dot = dot_prod + paw_dot_prod
#         sum_dot = wfs.gd.comm.sum(sum_dot)
#
#         return sum_dot
#
#     def dot_2(self, psi_1, psi_2, kpt, wfs):
#
#         def S(psit_G):
#             return psit_G
#
#         dot_prod = wfs.gd.integrate(S(psi_1), psi_2, False)
#         dot_prod = wfs.gd.comm.sum(dot_prod)
#
#         return dot_prod
#
#     def dot_all_k_and_b(self, x1, x2, wfs):
#
#         dot_pr_x1x2 = 0.0
#
#         for kpt in wfs.kpt_u:
#             n_occ = 0
#             for f in kpt.f_n:
#                 if f > 1.0e-10:
#                     n_occ += 1
#
#             for i in range(n_occ):
#                 dot_pr_x1x2 += self.dot(x1[self.n_kps *
#                                              kpt.s +
#                                              kpt.q][i],
#                                         x2[self.n_kps *
#                                              kpt.s +
#                                              kpt.q][i],
#                                         kpt, wfs)
#
#         dot_pr_x1x2 = wfs.kd.comm.sum(dot_pr_x1x2)
#
#         return 2.0 * dot_pr_x1x2.real
#
#     def calc_diff(self, x1, x2, wfs, const_0=1.0, const=1.0):
#         y_k = {}
#         for kpt in wfs.kpt_u:
#             k = self.n_kps * kpt.s + kpt.q
#             y_k[k] = \
#                 const_0 * x1[k] - \
#                 const * x2[k]
#
#         return y_k
#
#     def minus(self, wfs, x):
#
#         p = {}
#         for kpt in wfs.kpt_u:
#             p[self.n_kps * kpt.s + kpt.q] = \
#                 - x[self.n_kps * kpt.s + kpt.q].copy()
#
#         return p
#
#     def get_wf(self, wfs):
#
#         x = {}
#         for kpt in wfs.kpt_u:
#             x[self.n_kps * kpt.s + kpt.q] = kpt.psit_nG.copy()
#
#         return x
#
#     def multiply(self, x, const=1.0):
#
#         y = {}
#         for k in x.keys():
#             y[k] = const * x[k]
#
#         return y
#
#     def zeros(self, x):
#
#         y = {}
#         for k in x.keys():
#             y[k] = np.zeros_like(x[k])
#
#         return y
#
#     def apply_prec(self, wfs, x, const=1.0):
#         z = {}
#         a = self.alpha
#         for kpt in wfs.kpt_u:
#             k = self.n_kps * kpt.s + kpt.q
#             z[k] = np.zeros_like(x[k])
#             for i, y in enumerate(x[k]):
#                 z[k][i] = (-1.0 + a) * self.precond(y, kpt, None) + \
#                           a * self.beta_0 * x[k][i]
#                 z[k][i] *= const
#
#         return z
