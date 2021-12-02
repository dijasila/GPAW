"""
Tools for direcmin
"""

import numpy as np
import scipy.linalg as lalg


def expm_ed(a_mat, evalevec=False):
    """
    calculate matrix exponential
    using eigendecomposition of matrix a_mat

    :param a_mat: matrix to be exponented
    :param evalevec: if True then returns eigenvalues
                     and eigenvectors of A

    :return:
    """

    eigval, evec = np.linalg.eigh(1.0j * a_mat)

    product = (evec * np.exp(-1.0j * eigval)) @ evec.T.conj()

    if a_mat.dtype == float:
        product = product.real
    if evalevec:
        return np.ascontiguousarray(product), evec, eigval

    return np.ascontiguousarray(product)


def expm_ed_unit_inv(a_upp_r, oo_vo_blockonly=False):
    """
    calculate matrix exponential using
    Eq. (6) from
    J. Hutter, M. Parrinello, and S. Vogel,
    J. Chem. Phys., 101, 3862 (1994)
    :param a_upp_r: X (see eq in paper)
    :return: unitary matrix
    """
    if np.allclose(a_upp_r, np.zeros_like(a_upp_r)):
        dim_v = a_upp_r.shape[1]
        dim_o = a_upp_r.shape[0]
        if not oo_vo_blockonly:
            dim_v = a_upp_r.shape[1]
            dim_o = a_upp_r.shape[0]

            return np.eye(dim_o + dim_v, dtype=a_upp_r.dtype)
        else:
            return np.vstack([np.eye(dim_o, dtype=a_upp_r.dtype),
                              np.zeros(shape=(dim_v, dim_o),
                                       dtype=a_upp_r.dtype)])

    p_nn = a_upp_r @ a_upp_r.T.conj()
    eigval, evec = np.linalg.eigh(p_nn)
    # Eigenvalues cannot be negative
    eigval[eigval.real < 1.0e-13] = 1.0e-13
    sqrt_eval = np.sqrt(eigval)

    sin_sqrt_p = matrix_function(sqrt_eval, evec, np.sin)
    cos_sqrt_p = matrix_function(sqrt_eval, evec, np.cos)
    sqrt_inv_p = matrix_function(1.0 / sqrt_eval, evec)

    psin = sqrt_inv_p @ sin_sqrt_p
    u_oo = cos_sqrt_p
    u_vo = - a_upp_r.T.conj() @ psin

    if not oo_vo_blockonly:
        inv_p = matrix_function(1.0 / eigval, evec)
        u_ov = psin @ a_upp_r
        dim_v = a_upp_r.shape[1]
        dim_o = a_upp_r.shape[0]
        u_vv = np.eye(dim_v) + \
            a_upp_r.T.conj() @ (cos_sqrt_p - np.eye(dim_o)) @ inv_p @ a_upp_r
        u = np.vstack([
            np.hstack([u_oo, u_ov]),
            np.hstack([u_vo, u_vv])])
    else:
        u = np.vstack([u_oo, u_vo])

    return np.ascontiguousarray(u)


def d_matrix(omega):
    """
    Helper function for calculation of gradient
    w.r.t. skew-hermitian matrix
    see eq. 40 from
    A. V. Ivanov, E. Jónsson, T. Vegge, and H. Jónsso
    Comput. Phys. Commun., 267, 108047 (2021).
    arXiv:2101.12597 [physics.comp-ph]
    """

    m = omega.shape[0]
    u_m = np.ones(shape=(m, m))

    u_m = omega[:, np.newaxis] * u_m - omega * u_m

    with np.errstate(divide='ignore', invalid='ignore'):
        u_m = 1.0j * np.divide(np.exp(-1.0j * u_m) - 1.0, u_m)

    u_m[np.isnan(u_m)] = 1.0
    u_m[np.isinf(u_m)] = 1.0

    return u_m


def minimum_cubic_interpol(x_0, x_1, f_0, f_1, df_0, df_1):
    """
    given f, f' at boundaries of interval [x0, x1]
    calc. x_min where cubic interpolation is minimal
    :return: x_min
    """

    def cubic_function(a, b, c, d, x):
        """
        f(x) = a x^3 + b x^2 + c x + d
        :return: f(x)
        """
        return a * x ** 3 + b * x ** 2 + c * x + d

    if x_0 > x_1:
        x_0, x_1 = x_1, x_0
        f_0, f_1 = f_1, f_0
        df_0, df_1 = df_1, df_0

    r = x_1 - x_0
    a = - 2.0 * (f_1 - f_0) / r ** 3.0 + \
        (df_1 + df_0) / r ** 2.0
    b = 3.0 * (f_1 - f_0) / r ** 2.0 - \
        (df_1 + 2.0 * df_0) / r
    c = df_0
    d = f_0
    D = b ** 2.0 - 3.0 * a * c

    if D < 0.0:
        if f_0 < f_1:
            x_min = x_0
        else:
            x_min = x_1
    else:
        r0 = (-b + np.sqrt(D)) / (3.0 * a) + x_0
        if x_0 < r0 < x_1:
            f_r0 = cubic_function(a, b, c, d, r0 - x_0)
            if f_0 > f_r0 and f_1 > f_r0:
                x_min = r0
            else:
                if f_0 < f_1:
                    x_min = x_0
                else:
                    x_min = x_1
        else:
            if f_0 < f_1:
                x_min = x_0
            else:
                x_min = x_1

    return x_min


def minimum_parabola_interpol(x_0, x_1, f_0, f_1, df_0):
    """
    given f(x0), f'(x0) and f(x1)
    calc. x_min where parabola interpolation
    f(x) = a x^2 + b x + c is minimal
    :return: x_min
    """
    assert x_0 <= x_1

    r = x_1 - x_0
    a = (f_1 - f_0 - r * df_0) / r ** 2
    b = df_0
    c = f_0

    a_min = - b / (2.0 * a)
    f_min = a * a_min ** 2 + b * a_min + c
    if f_min > f_1:
        a_min = x_1 - x_0
        if f_0 < f_1:
            a_min = 0
    x_min = a_min + x_0
    return x_min


def matrix_function(evals, evecs, func=lambda x: x):
    """
    calculate matrix function func(A)
    you need to provide
    :param evals: eigenvalues of A
    :param evecs: eigenvectors of A
    :return: func(A)
    """
    return (evecs * func(evals)) @ evecs.T.conj()


def loewdin_lcao(C_nM, S_MM):
    """
    Loewdin based orthonormalization
    for LCAO mode

    C_nM <- sum_m C_nM[m] [1/sqrt(S)]_mn
    S_mn = (C_nM[m].conj(), S_MM C_nM[n])

    :param C_nM: LCAO coefficients
    :param S_MM: Overlap matrix between basis functions
    :return: Orthonormalized coefficients so that new S_mn = delta_mn
    """

    ev, S_overlapp = np.linalg.eigh(C_nM.conj() @ S_MM @ C_nM.T)
    ev_sqrt = np.diag(1.0 / np.sqrt(ev))

    S = S_overlapp @ ev_sqrt @ S_overlapp.T.conj()

    return S.T @ C_nM


def gramschmidt_lcao(C_nM, S_MM):
    """
    Gram-Schmidt orthonormalization using cholesky decomposition
    for LCAO mode

    :param C_nM: LCAO coefficients
    :param S_MM: Overlap matrix between basis functions
    :return: Orthonormalized coefficients so that new S_mn = delta_mn
    """

    S_nn = C_nM @ S_MM.conj() @ C_nM.T.conj()
    L_nn = lalg.cholesky(S_nn, lower=True,
                         overwrite_a=True, check_finite=False)
    return lalg.solve(L_nn, C_nM)


def excite(calc, i, a, spin=(0, 0)):
    """
    remove an electron from spin[0], homo + i
    and add an electron to spin[1], lumo + a
    occupation number will be taken from
    calc.get_occupation_numbers() for each spin

    :return: new occupation numbers
    """

    f_sn = []
    for s in range(calc.wfs.nspins):
        f_sn.append(
            calc.get_occupation_numbers(spin=s))

    k = 0
    for _ in reversed(f_sn[spin[0]]):
        k += 1
        if _ > 0:
            break

    homo = len(f_sn[spin[0]]) - k

    lumo = 0
    for _ in f_sn[spin[1]]:
        if _ == 0:
            break
        lumo += 1

    f_sn[spin[0]][homo + i] -= 1
    f_sn[spin[1]][lumo + a] += 1

    return f_sn
