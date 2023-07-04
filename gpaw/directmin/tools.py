"""
Tools for directmin
"""

import numpy as np
import scipy.linalg as lalg
from copy import deepcopy


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
    Gram-Schmidt orthonormalization using Cholesky decomposition
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

    f_sn = [calc.get_occupation_numbers(spin=s)
            for s in range(calc.wfs.nspins)]

    f_n0 = np.asarray(f_sn[spin[0]])
    lumo = len(f_n0[f_n0 > 0])
    homo = lumo - 1

    f_sn[spin[0]][homo + i] -= 1.0
    f_sn[spin[1]][lumo + a] += 1.0

    return f_sn


def get_occupations(wfs):
    f_sn = {}
    for kpt in wfs.kpt_u:
        n_kps = wfs.kd.nibzkpts
        u = n_kps * kpt.s + kpt.q
        f_sn[u] = kpt.f_n.copy()
    if wfs.nspins == 2 and wfs.kd.comm.size > 1:
        if wfs.kd.comm.rank == 0:
            # occupation numbers
            size = np.array([0])
            wfs.kd.comm.receive(size, 1)
            f_2n = np.zeros(shape=(int(size[0])))
            wfs.kd.comm.receive(f_2n, 1)
            f_sn[1] = f_2n
            size = np.array([f_sn[0].shape[0]])
            wfs.kd.comm.send(size, 1)
            wfs.kd.comm.send(f_sn[0], 1)
        else:
            # occupations
            size = np.array([f_sn[1].shape[0]])
            wfs.kd.comm.send(size, 0)
            wfs.kd.comm.send(f_sn[1], 0)
            size = np.array([0])
            wfs.kd.comm.receive(size, 0)
            f_2n = np.zeros(shape=(int(size[0])))
            wfs.kd.comm.receive(f_2n, 0)
            f_sn[0] = f_2n

    return f_sn


def excite_and_sort(wfs, i, a, spin=(0, 0), mode='fdpw'):

    assert wfs.nspins == 2
    if spin == (0, 0) or spin == (1, 1):
        exctype = 'singlet'
    else:
        exctype = 'triplet'

    if exctype == 'singlet':
        for kpt in wfs.kpt_u:
            n_kps = wfs.kd.nibzkpts
            u = n_kps * kpt.s + kpt.q
            s = spin[0]
            if kpt.s == s:
                occ_ex_up = kpt.f_n.copy()
                lumo = len(occ_ex_up[occ_ex_up > 0])
                homo = lumo - 1
                indx = [homo + i, lumo + a]
                swindx = [lumo + a, homo + i]
                if mode == 'fdpw':
                    kpt.psit_nG[indx] = kpt.psit_nG[swindx]
                elif mode == 'lcao':
                    wfs.eigensolver.c_nm_ref[u][indx] = \
                        wfs.eigensolver.c_nm_ref[u][swindx]
                    kpt.C_nM[:] = wfs.eigensolver.c_nm_ref[u].copy()
                else:
                    raise KeyError
                kpt.eps_n[indx] = kpt.eps_n[swindx]
    elif exctype == 'triplet':
        f_sn = get_occupations(wfs)
        lumo = len(f_sn[spin[1]][f_sn[spin[1]] > 0])
        homo = len(f_sn[spin[0]][f_sn[spin[1]] > 0]) - 1
        f_sn[spin[0]][homo + i] -= 1
        f_sn[spin[1]][lumo + a] += 1
        # sort wfs
        for kpt in wfs.kpt_u:
            n_kps = wfs.kd.nibzkpts
            u = n_kps * kpt.s + kpt.q
            kpt.f_n = f_sn[u].copy()
            occupied = kpt.f_n > 1.0e-10
            n_occ = len(kpt.f_n[occupied])
            if n_occ == 0.0:
                continue
            if np.min(kpt.f_n[:n_occ]) == 0:
                ind_occ = np.argwhere(occupied)
                ind_unocc = np.argwhere(~occupied)
                ind = np.vstack((ind_occ, ind_unocc))
                # Sort coefficients, occupation numbers, eigenvalues
                if mode == 'fdpw':
                    kpt.psit_nG[:] = np.squeeze(kpt.psit_nG[ind])
                elif mode == 'lcao':
                    wfs.eigensolver.c_nm_ref[u] = np.squeeze(
                        wfs.eigensolver.c_nm_ref[u][ind])
                    kpt.C_nM[:] = wfs.eigensolver.c_nm_ref[u].copy()
                else:
                    raise KeyError
                kpt.f_n = np.squeeze(kpt.f_n[ind])
                kpt.eps_n = np.squeeze(kpt.eps_n[ind])
    else:
        raise KeyError


def dict_to_array(x):
    """
    Converts dictionaries with integer keys to one long array by appending.

    :param x: Dictionary
    :return: Long array, dimensions of original dictionary parts, total
             dimensions
    """
    y = []
    dim = []
    dimtot = 0
    for k in x.keys():
        assert type(k) == int, 'Cannot convert dict to array if keys are not '
        'integer.'
        y += list(x[k])
        dim.append(len(x[k]))
        dimtot += len(x[k])
    return np.asarray(y), dim, dimtot


def array_to_dict(x, dim):
    """
    Converts long array to dictionary with integer keys with values of
    dimensionality specified in dim.

    :param x: Array
    :param dim: List with dimensionalities of parts of the dictionary
    :return: Dictionary
    """
    y = {}
    start = 0
    stop = 0
    for i in range(len(dim)):
        stop += dim[i]
        y[i] = x[start: stop]
        start += dim[i]
    return y


def rotate_orbitals(etdm, wfs, indices, angles, channels):
    """
    Applies rotations between pairs of orbitals.

    :param etdm:       ETDM object for a converged or at least initialized
                       calculation
    :param indices:    List of indices. Each element must be a list of an
                       orbital pair corresponding to the orbital rotation.
                       For occupied-virtual rotations (unitary invariant or
                       sparse representations), the first index represents the
                       occupied, the second the virtual orbital.
                       For occupied-occupied rotations (sparse representation
                       only), the first index must always be smaller than the
                       second.
    :param angles:     List of angles in radians.
    :param channels:   List of spin channels.
    """

    angles = - np.array(angles) * np.pi / 180.0
    a_vec_u = get_a_vec_u(etdm, wfs, indices, angles, channels)
    c = {}
    for kpt in wfs.kpt_u:
        k = etdm.kpointval(kpt)
        c[k] = wfs.kpt_u[k].C_nM.copy()
    etdm.rotate_wavefunctions(wfs, a_vec_u, c)


def get_a_vec_u(etdm, wfs, indices, angles, channels, occ=None):
    """
    Creates an orbital rotation vector based on given indices, angles and
    corresponding spin channels.

    :param etdm:       ETDM object for a converged or at least initialized
                       calculation
    :param indices:    List of indices. Each element must be a list of an
                       orbital pair corresponding to the orbital rotation.
                       For occupied-virtual rotations (unitary invariant or
                       sparse representations), the first index represents the
                       occupied, the second the virtual orbital.
                       For occupied-occupied rotations (sparse representation
                       only), the first index must always be smaller than the
                       second.
    :param angles:     List of angles in radians.
    :param channels:   List of spin channels.
    :param occ:        Occupation numbers for each k-point. Must be specified
                       if the orbitals in the ETDM object are not ordered
                       canonically, as the user orbital indexation is different
                       from the one in the ETDM object then.

    :return new_vec_u: Orbital rotation coordinate vector containing the
                       specified values.
    """

    etdm.sort_orbitals_mom(wfs)

    new_vec_u = {}
    ind_up = etdm.ind_up
    a_vec_u = deepcopy(etdm.a_vec_u)
    conversion = []
    for k in a_vec_u.keys():
        new_vec_u[k] = np.zeros_like(a_vec_u[k])
        if occ is not None:
            f_n = occ[k]
            occupied = f_n > 1.0e-10
            n_occ = len(f_n[occupied])
            if n_occ == 0.0:
                continue
            if np.min(f_n[:n_occ]) == 0:
                ind_occ = np.argwhere(occupied)
                ind_unocc = np.argwhere(~occupied)
                ind = np.vstack((ind_occ, ind_unocc))
                ind = np.squeeze(ind)
                conversion.append(list(ind))
            else:
                conversion.append(None)

    for ind, ang, s in zip(indices, angles, channels):
        if occ is not None:
            if conversion[s] is not None:
                ind[0] = conversion[s].index(ind[0])
                ind[1] = conversion[s].index(ind[1])
        m = np.where(ind_up[s][0] == ind[0])[0]
        n = np.where(ind_up[s][1] == ind[1])[0]
        res = None
        for i in m:
            for j in n:
                if i == j:
                    res = i
        if res is None:
            raise ValueError('Orbital rotation does not exist.')
        new_vec_u[s][res] = ang

    return new_vec_u


def get_n_occ(kpt):

    nbands = len(kpt.f_n)
    n_occ = 0
    while n_occ < nbands and kpt.f_n[n_occ] > 1e-10:
        n_occ += 1
    return n_occ


def get_indices(dimens, dtype):

    if dtype == complex:
        il1 = np.tril_indices(dimens)
    else:
        il1 = np.tril_indices(dimens, -1)

    return il1


def get_random_um(dim, dtype):

    a = 0.01 * np.random.rand(dim, dim)
    if dtype is complex:
        a = a + 1.0j * 0.01 * np.random.rand(dim, dim)
    a = a - a.T.conj()
    return expm_ed(a)
