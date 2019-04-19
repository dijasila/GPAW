import numpy as np
from gpaw.utilities.lapack import diagonalize


def cubic_interpolation(x_0, x_1, f_0, f_1, df_0, df_1):
    """
    f(x) = a x^3 + b x^2 + c x + d
    :param x_0:
    :param x_1:
    :param f_0:
    :param f_1:
    :param df_0:
    :param df_1:
    :return:
    """

    if x_0 > x_1:
        x_0, x_1 = x_1, x_0
        f_0, f_1 = f_1, f_0
        df_0, df_1 = df_1, df_0

    r = x_1 - x_0
    a = - 2.0 * (f_1 - f_0) / r ** 3.0 + (df_1 + df_0) / r ** 2.0
    b = 3.0 * (f_1 - f_0) / r ** 2.0 - (df_1 + 2.0 * df_0) / r
    c = df_0
    d = f_0
    D = b ** 2.0 - 3.0 * a * c

    if D < 0.0:
        if f_0 < f_1:
            alpha = x_0
        else:
            alpha = x_1
    else:
        r0 = (-b + np.sqrt(D)) / (3.0 * a) + x_0
        if x_0 < r0 < x_1:
            f_r0 = cubic_function(a, b, c, d, r0 - x_0)

            if f_0 > f_r0 and f_1 > f_r0:
                alpha = r0
            else:
                if f_0 < f_1:
                    alpha = x_0
                else:
                    alpha = x_1
        else:
            if f_0 < f_1:
                alpha = x_0
            else:
                alpha = x_1

    return alpha


def cubic_interpolation_2(x_0, x_1, x_2, f_0, df_0, f_1, f_2):
    """
    f(x) = a x^3 + b x^2 + c x + d
    :param x_0:
    :param x_1:
    :param x_2:
    :param f_0:
    :param df_0:
    :param f_1:
    :param df_2:
    :return:
    """
    assert x_0 <= x_1 <= x_2

    # shift to the center of origin
    x_c = x_0
    x_0 -= x_c
    x_1 -= x_c
    x_2 -= x_c

    d = f_0
    c = df_0

    y1 = f_1 - c * x_1 - d
    y2 = f_2 - c * x_2 - d

    # a = (y1 / x_1**2 - y2 / x_2**2) / (x_1 - x_2)
    # b = x_1 * x_2 * (y1 / x_1**3 - y2 / x_2**3) / (x_1 - x_2)

    a = (x_1**2.0 * y2 - x_2**2.0 * y1) / (x_1**2 * x_2**2 * (x_2 - x_1))
    b = (-x_1**3.0 * y2 + x_2**3.0 * y1) / (x_1**2 * x_2**2 * (x_2 - x_1))
    D = b**2.0 - 3.0 * a * c
    if D >= 0.0:
        x_min = (-b + np.sqrt(D)) / (3.0 * a)
    else:
        if f_1 <= f_2:
            x_min = x_1
        else:
            x_min = x_2

    f_min = cubic_function(a, b, c, d, x_min)

    assert f_min < f_1
    assert f_min < f_2
    assert f_min < f_0

    return x_min + x_c


def cubic_function(a, b, c, d, x):
    return a * x ** 3 + b * x ** 2 + c * x + d


def parabola_interpolation(x_0, x_1, f_0, f_1, df_0):
    """
    f(x) = a x^2 + b x + c
    :param x_0:
    :param x_1:
    :param f_0:
    :param f_1:
    :param df_0:
    :return:
    """
    assert x_0 <= x_1

    r = x_1 - x_0
    a = (f_1 - f_0 - r * df_0) / r**2
    b = df_0
    c = f_0
    a_min = - b / (2.0 * a)
    f_min = a * a_min**2 + b * a_min + c

    if f_min > f_1:
        a_min = x_1 - x_0
        if f_0 < f_1:
            a_min = 0

    return a_min + x_0


def d_matrix(omega):

    m = omega.shape[0]
    u_m = np.ones(shape=(m, m))
    u_m = omega[:, np.newaxis] * u_m - omega * u_m

    with np.errstate(divide='ignore', invalid='ignore'):
        u_m = 1.0j * np.divide(np.exp(-1.0j * u_m) - 1.0, u_m)

    u_m[np.isnan(u_m)] = 1.0
    u_m[np.isinf(u_m)] = 1.0

    return u_m


def expm_ed(A, evalevec=False):

    """
    calcualte matrix exponential
    through eigendecomposition of matrix A

    :param A: to be exponented
    :param evalevec: if True then return eigenvalues
                     and eigenvectors of A
    :return:
    """

    evec = 1.0j * A
    eval = np.empty(A.shape[0])
    diagonalize(evec, eval)

    if evalevec:
        if A.dtype == float:
            return np.dot(evec.T.conj() * np.exp(-1.0j*eval),
                          evec).real, evec.T.conj(), eval
        else:
            return np.dot(evec.T.conj() * np.exp(-1.0j * eval),
                          evec), evec.T.conj(), eval

    else:
        if A.dtype == float:
            return np.dot(evec.T.conj() * np.exp(-1.0j * eval),
                          evec).real
        else:
            return np.dot(evec.T.conj() * np.exp(-1.0j * eval),
                          evec)
