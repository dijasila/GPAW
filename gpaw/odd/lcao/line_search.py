import numpy as np
from gpaw.odd.lcao.tools import cubic_interpolation, \
    parabola_interpolation


def descent(phi_0, phi_j, eps=1.0e-2):
    if phi_j <= phi_0 + eps * abs(phi_0):
        return True
    else:
        return False


def appr_wc(der_phi_0, phi_0, der_phi_j, phi_j):

    eps = 1.0e-6
    delta = 0.1
    sigma = 0.9

    if (phi_j <= phi_0 + eps * abs(phi_0)) and \
            ((2.0*delta - 1.0) * der_phi_0 >= der_phi_j >= sigma *
             der_phi_0):
        return True
    else:
        return False


class StrongWolfeConditions:
    """
    From a book of Jorge Nocedal and Stephen J. Wright 'Numerical
    Optimization' Second Edition, 2006 (p. 56)

    This call should return a_star, phi_star, der_phi_star, g_star,
    where a_star is step length satisfied the strong Wolfe condts:

    f(x_k + a_k p_k) <= f(x_k) + c_1 a_k \grad f_k \cdot p_k,

    |\grad f(x_k + a_k p_k) \cdot p_k | <= c_2 |\grad f_k \cdot p_k|,

    phi = f (x_k + a_k*p_k)
    der_phi = \grad f(x_k + a_k p_k) \cdot p_k
    g = \grad f(x_k + a_k p_k)
    """

    def __init__(self, evaluate_phi_and_der_phi, log,
                 c1=1.0e-4, c2=0.9,
                 method=None, max_iter=3, eps_dx=1.0e-10,
                 eps_df=1.0e-10, awc=True):
        """
        :param evaluate_phi_and_der_phi: function which calculate
        phi, der_phi and g for given A_s, P_s, n_dim and alpha
        A_s[s] is skew-hermitian matrix, P_s[s] is matrix direction
        :param log: for output messages
        :param method: used only in initial guess for alpha
        :param max_iter: maximum number of iterations
        :param eps_dx: length of minimal interval where alpha can
        be found
        :param eps_df: minimal change of function
        :param c1: see above
        :param c2: see above
        """

        self.\
            evaluate_phi_and_der_phi = evaluate_phi_and_der_phi
        self.max_iter = max_iter
        self.log = log
        self.method = method
        self.eps_dx = eps_dx
        self.eps_df = eps_df
        self.c1 = c1
        self.c2 = c2
        self.awc = awc

    def step_length_update(self, A_s, P_s, n_dim,
                           phi_0=None, der_phi_0=None,
                           phi_old=None, der_phi_old=None,
                           alpha_max=5.0, alpha_old=1.0):
        if self.method in ['HZcg', 'SD', 'FRcg', 'PRcg', 'PRpcg']:
            c1 = self.c1 = 1.0e-4
            c2 = self.c2 = 0.9
        else:
            c1 = self.c1
            c2 = self.c2

        alpha_1 = self.init_guess(phi_0=phi_0, der_phi_0=der_phi_0,
                                  phi_old=phi_old,
                                  der_phi_old=der_phi_old,
                                  alpha_old=alpha_old)

        i = 1
        if phi_0 is None or der_phi_0 is None:
            phi_0, der_phi_0, g_0 = \
                self.evaluate_phi_and_der_phi(A_s, P_s, n_dim)

        alpha = [0.0, alpha_1]

        phi_i_1 = phi_0
        der_phi_i_1 = der_phi_0

        max_iter = self.max_iter
        phi_max = None
        der_phi_max = None

        # 'Get_step_length:'
        while True:

            phi_i, der_phi_i, g_i = \
                self.evaluate_phi_and_der_phi(A_s, P_s, n_dim,
                                              alpha[i])

            if self.awc is True:
                if appr_wc(der_phi_0, phi_0, der_phi_i, phi_i):

                    a_star = alpha[i]
                    phi_star = phi_i
                    der_phi_star = der_phi_i
                    g_star = g_i

                    break
            if phi_i > phi_0 + c1 * alpha[i] * der_phi_0 or \
                    (phi_i >= phi_i_1 and i > 1):
                a_star, phi_star, der_phi_star, g_star = \
                    self.zoom(alpha[i - 1], alpha[i],
                              phi_i_1, der_phi_i_1,
                              phi_i, der_phi_i, A_s, P_s, n_dim,
                              phi_0, der_phi_0, c1, c2)
                break

            if np.fabs(der_phi_i) <= -c2 * der_phi_0:
                a_star = alpha[i]
                phi_star = phi_i
                der_phi_star = der_phi_i
                g_star = g_i
                break

            if der_phi_i >= 0.0:
                a_star, phi_star, der_phi_star, g_star = \
                    self.zoom(alpha[i], alpha[i - 1],
                              phi_i, der_phi_i,
                              phi_i_1, der_phi_i_1,
                              A_s, P_s, n_dim,
                              phi_0, der_phi_0, c1, c2)
                break

            if alpha_max < alpha[i]:
                alpha_max = 2.0 * alpha[i]

            if phi_max is None or der_phi_max is None:
                phi_max, der_phi_max, g_max = \
                    self.evaluate_phi_and_der_phi(A_s, P_s, n_dim,
                                                  alpha_max)

                if self.awc is True:
                    if appr_wc(der_phi_0, phi_0, der_phi_max, phi_max):
                        a_star = alpha_max
                        phi_star = phi_max
                        der_phi_star = der_phi_max
                        g_star = g_max
                        break

            a_new = cubic_interpolation(alpha[i], alpha_max,
                                        phi_i, phi_max,
                                        der_phi_i, der_phi_max)

            if a_new > alpha_max or np.fabs(
                    a_new - alpha_max) < 1.0e-1:
                # Jump over boundary
                # Or close to the upper boundary
                # Increase interval

                alpha_max = 1.5 * a_new
                phi_max = None
                der_phi_max = None

            alpha.append(a_new)

            phi_i_1 = phi_i
            der_phi_i_1 = der_phi_i

            i += 1
            if abs(alpha[-1] - alpha[-2]) < 1.0e-5 or i == max_iter + 1:
                # self.log('Cannot satisfy strong Wolfe condition')
                # if i == max_iter:
                #     self.log('Too many iterations')

                a_star = alpha[i]
                phi_star = phi_i
                der_phi_star = der_phi_i
                g_star = g_i

                break

        return a_star, phi_star, der_phi_star, g_star

    def zoom(self, a_lo, a_hi,
             f_lo, df_lo, f_hi, df_hi, A_s,
             P_s, n_dim, phi_0, der_phi_0,
             c1, c2):

        max_iter = self.max_iter
        i = 0

        while True:

            a_j = cubic_interpolation(a_lo, a_hi,
                                      f_lo, f_hi,
                                      df_lo, df_hi)

            phi_j, der_phi_j, g_j = \
                self.evaluate_phi_and_der_phi(A_s, P_s, n_dim, a_j)

            if self.awc is True:
                if appr_wc(der_phi_0, phi_0, der_phi_j, phi_j):

                    a_star = a_j
                    phi_star = phi_j
                    der_phi_star = der_phi_j
                    g_star = g_j

                    break

            if phi_j > phi_0 + c1 * a_j * der_phi_0 or phi_j >= f_lo:
                a_hi = a_j
                f_hi = phi_j
                df_hi = der_phi_j

            else:
                if abs(der_phi_j) <= -c2 * der_phi_0:
                    a_star = a_j
                    phi_star = phi_j
                    der_phi_star = der_phi_j
                    g_star = g_j

                    break
                if der_phi_j * (a_hi - a_lo) >= 0.0:
                    a_hi = a_lo
                    f_hi = f_lo
                    df_hi = df_lo

                a_lo = a_j
                f_lo = phi_j
                df_lo = der_phi_j

            i += 1

            if np.fabs(a_lo - a_hi) < self.eps_dx and a_lo < \
                    self.eps_dx:

                self.log('Cannot satisfy strong Wolfe condition')
                self.log('Interval is close to zero')
                self.log('Lower boundary is close to zero.')

                a_star = a_lo

                phi_star, der_phi_star, g_star = \
                    self.evaluate_phi_and_der_phi(A_s, P_s, n_dim,
                                                  a_star)

                break

            elif np.fabs(a_lo - a_hi) < self.eps_dx:
                # self.log('Cannot satisfy strong Wolfe condition,')
                # self.log('Only sufficient descent')
                # self.log('Search interval is less than'
                #          '%.2e' % self.eps_dx)

                a_star = a_lo

                phi_star, der_phi_star, g_star = \
                    self.evaluate_phi_and_der_phi(A_s, P_s, n_dim,
                                                  a_star)
                break

            if i == max_iter:
                # self.log('Cannot satisfy strong Wolfe condition,')
                # self.log('Made too many iterations')
                if a_lo > self.eps_dx:
                    a_star = a_lo

                    phi_star, der_phi_star, g_star = \
                        self.evaluate_phi_and_der_phi(A_s, P_s, n_dim,
                                                      a_star)

                else:
                    a_star = a_hi

                    phi_star, der_phi_star, g_star = \
                        self.evaluate_phi_and_der_phi(A_s, P_s, n_dim,
                                                      a_star)

                break

        return a_star, phi_star, der_phi_star, g_star

    def init_guess(self, phi_0, der_phi_0, phi_old, der_phi_old,
                   alpha_old = 1.0):

        # chose initial alpha
        if self.method in ['HZcg', 'SD', 'FRcg', 'PRcg', 'PRpcg']:
            if phi_old is not None and der_phi_old is not None:
                try:
                    alpha_1 = 2.0 * (phi_0 - phi_old) / der_phi_old
                    # alpha_1 = alpha_old * der_phi_old / der_phi_0
                    if alpha_1 < 0.1:
                        if alpha_old < 0.1:
                            alpha_1 = 10.0
                        else:
                            alpha_1 = alpha_old

                except ZeroDivisionError:
                    alpha_1 = 1.0
            else:
                alpha_1 = 1.0
        elif self.method is 'BFGS' or 'LBFGS':
            alpha_1 = 1.0
        else:
            alpha_1 = 1.0

        return alpha_1


class UnitStepLength:

    def __init__(self, evaluate_phi_and_der_phi, log):
        """

        :param evaluate_phi_and_der_phi: function which calculate
        phi, der_phi and g for given A_s, P_s, n_dim and alpha
        A_s[s] is skew-hermitian matrix, P_s[s] is matrix direction
        :param log: for output messages
        """

        self.evaluate_phi_and_der_phi = evaluate_phi_and_der_phi
        self.log = log

    def step_length_update(self, A_s, P_s, n_dim,
                           *args, **kwargs):

        a_star = 1.0
        phi_star, der_phi_star, g_star = \
                self.evaluate_phi_and_der_phi(A_s, P_s, n_dim,
                                              a_star)

        return a_star, phi_star, der_phi_star, g_star


class Parabola:

    """
    phi = f (x_k + a_k*p_k)
    der_phi = \grad f(x_k + a_k p_k) \cdot p_k
    g = \grad f(x_k + a_k p_k)
    """

    def __init__(self, evaluate_phi_and_der_phi, log):
        """
        :param evaluate_phi_and_der_phi: function which calculate
        phi, der_phi and g for given A_s, P_s, n_dim and alpha
        A_s[s] is skew-hermitian matrix, P_s[s] is matrix direction
        :param log: for output messages
        """
        self.evaluate_phi_and_der_phi = evaluate_phi_and_der_phi
        self.log = log

    def step_length_update(self, A_s, P_s, n_dim,
                           phi_0, der_phi_0,
                           *args, **kwargs):

        phi_i, der_phi_i, g_i = \
            self.evaluate_phi_and_der_phi(A_s, P_s, n_dim,
                                          alpha=1.0)

        # if appr_wc(der_phi_0, phi_0, der_phi_i, phi_i):
        if descent(phi_0, phi_i, eps=1.0e-2):
            return 1.0, phi_i, der_phi_i, g_i
        else:
            a_star = parabola_interpolation(0.0, 1.0,
                                            phi_0, phi_i,
                                            der_phi_0)
            if a_star < 0.8e-1:
                a_star = 0.5
        phi_star, der_phi_star, g_star = \
            self.evaluate_phi_and_der_phi(A_s, P_s, n_dim,
                                          a_star)

        return a_star, phi_star, der_phi_star, g_star

