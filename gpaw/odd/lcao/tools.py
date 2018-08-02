from scipy.linalg import expm
import numpy as np
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.blas import mmm


def construct_matrix_for_exp(A, L):
    C = np.zeros_like(A)
    return np.array(np.bmat([[A, L], [C, A]]))


def get_grad_from_matrix_exponential(A, L):
    B = construct_matrix_for_exp(A, L)
    me = expm(B)
    U = me[0:A.shape[0], 0:A.shape[1]].T.conj()
    D = me[0:A.shape[0], A.shape[1]:]

    return np.dot(D, U).T


def steepest_descent(A, G, alpha = 1.0):

    for i in range(A.shape[0]):
        for j in range(i, A.shape[1]):
            A[i][j] += -alpha*G[i][j]
            if j != i:
                A[j][i] = -np.conjugate(A[i][j])


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

        a = - 2.0 * (f_1 - f_0) / r ** 3.0 + \
            (df_1 + df_0) / r ** 2.0

        b = 3.0 * (f_1 - f_0) / r ** 2.0 - \
            (df_1 + 2.0 * df_0) / r

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


def cubic_function(a, b, c, d, x):
    return a * x ** 3 + b * x ** 2 + c * x + d


def random_skew_herm_matrix(n_d, a=-1.0, b=1.0, dtype=complex):
    A = np.zeros(shape=(n_d, n_d),
                 dtype=dtype)

    if dtype == complex:
        for i in range(n_d):
            for j in range(i, n_d):
                if j != i:
                    A[i][j] = np.random.uniform(a, b) + \
                              1.0j * np.random.uniform(a, b)
                    A[j][i] = - np.conjugate(A[i][j])
                else:
                    A[i][j] = 1.0j * np.random.uniform(a, b)
    else:
        for i in range(n_d):
            for j in range(i + 1, n_d):
                A[i][j] = np.random.uniform(a, b)
                A[j][i] = -A[i][j]

    return A


def transform_antisymmetric_X_s_to_x(X_s, dtype=complex):
    nspins = X_s.keys()

    if len(nspins) == 2:
        n1 = X_s[0].shape[0]
        n2 = X_s[1].shape[0]
    else:
        n1 = X_s[0].shape[0]
        n2 = 0
    n_dim = (n1, n2)

    if dtype == complex:
        x = np.zeros(shape=(n1 * n1 +
                            n2 * n2),
                     dtype=float)

        for s in nspins:
            for i in range(n_dim[s]):
                for j in range(i + 1, n_dim[s]):
                    x[i * n_dim[s] + j - i * (i + 1) // 2 - i - 1 +
                      s * n_dim[s - 1] ** 2] = \
                        X_s[s][i][j].real

                    x[i * n_dim[s] + j - i * (i + 1) // 2 +
                      s * n_dim[s - 1] ** 2 + n_dim[s] * (
                          n_dim[s] - 1) // 2] = \
                        X_s[s][i][j].imag

                x[i * n_dim[s] + i - i * (i + 1) // 2 +
                  s * n_dim[s - 1] ** 2 +
                  n_dim[s] * (n_dim[s] - 1) // 2] = \
                    X_s[s][i][i].imag

    else:
        x = np.zeros(shape=(n1 * (n1 - 1) // 2 +
                            n2 * (n2 - 1) // 2),
                     dtype=float)

        for s in nspins:
            for i in range(n_dim[s]):
                for j in range(i + 1, n_dim[s]):
                    x[i * n_dim[s] + j - i * (i + 1) // 2 - i - 1 +
                      s * n_dim[s - 1] * (n_dim[s - 1] - 1) // 2] = \
                        X_s[s][i][j]

    return x


def transform_x_to_antisymmetric_X_s(x, n_dim, dtype=complex):
    if n_dim[1] == 0:
        nspins = 1
    else:
        nspins = 2

    X_s = {}
    for s in range(nspins):
        X_s[s] = np.zeros(shape=(n_dim[s], n_dim[s]),
                          dtype=dtype)

    if dtype == complex:

        for s in range(nspins):
            for i in range(n_dim[s]):
                for j in range(i + 1, n_dim[s]):
                    X_s[s][i][j] = \
                        x[i * n_dim[s] + j - i * (i + 1) // 2 -
                          i - 1 +
                          s * n_dim[s - 1] ** 2] + \
                        1.0j * x[i * n_dim[s] + j - i * (i + 1) // 2 +
                                 s * n_dim[s - 1] ** 2 +
                                 n_dim[s] * (n_dim[s] - 1) // 2]

                    X_s[s][j][i] = -np.conjugate(X_s[s][i][j])

                X_s[s][i][i] = \
                    1.0j * x[i * n_dim[s] + i - i * (i + 1) // 2 +
                             s * n_dim[s - 1] ** 2 +
                             n_dim[s] * (n_dim[s] - 1) // 2]

    else:

        for s in range(nspins):
            for i in range(n_dim[s]):
                for j in range(i + 1, n_dim[s]):
                    X_s[s][i][j] = \
                        x[i * n_dim[s] + j - i * (i + 1) // 2 -
                          i - 1 +
                          s * n_dim[s - 1] * (n_dim[s - 1] - 1) // 2]

                    X_s[s][j][i] = -X_s[s][i][j]

    return X_s


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

        # print(x_0, x_1)

        # if x_0 > x_1:
        #     x_0, x_1 = x_1, x_0
        #     f_0, f_1 = f_1, f_0
        #     df_1 = df_0

        r = x_1 - x_0

        a = (f_1 - f_0 - r * df_0) / r**2
        b = df_0
        c = f_0

        a_min = - b / (2.0*a)

        f_min = a * a_min**2 + b * a_min + c

        if f_min > f_1:
            a_min = x_1 - x_0
            if f_0 < f_1:
                a_min = 0

        return a_min + x_0


def expm_ed2(A, evalevec=False):

    """
    calcualte matrix exponential
    through eigendecomposition of matrix A

    :param A: to be exponented
    :param evalevec: if True then return eigenvalues
                     and eigenvectors of A
    :return:
    """

    eval, evec = np.linalg.eigh(1.0j*A)

    if evalevec:
        if A.dtype == float:
            return np.dot(evec * np.exp(-1.0j*eval),
                          evec.T.conj()).real,\
                   evec, eval
        else:
            return np.dot(evec * np.exp(-1.0j * eval),
                          evec.T.conj()), evec, eval
    else:
        if A.dtype == float:
            return np.dot(evec * np.exp(-1.0j*eval),
                          evec.T.conj()).real
        else:
            return np.dot(evec * np.exp(-1.0j * eval),
                          evec.T.conj())


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


def D_matrix(omega):

    m = omega.shape[0]
    u_m = np.ones(shape=(m, m))

    u_m = omega[:, np.newaxis] * u_m - omega * u_m

    with np.errstate(divide='ignore', invalid='ignore'):
        u_m = 1.0j * np.divide(np.exp(-1.0j * u_m) - 1.0, u_m)

    u_m[np.isnan(u_m)] = 1.0
    u_m[np.isinf(u_m)] = 1.0

    return u_m


def get_grad(L, evec, eval):

    if L.dtype == complex:
        G = np.zeros_like(L)

        mmm(1.0, L, 'n', evec, 'n', 0.0, G)
        A = np.zeros_like(L)
        mmm(1.0, evec, 'c', G, 'n', 0.0, A)

        G = A * D_matrix(eval)

        mmm(1.0, G, 'n', evec, 'c', 0.0, A)
        mmm(1.0, evec, 'n', A, 'n', 0.0, G)

    else:
        G = np.zeros_like(L)

        mmm(1.0, L, 't', evec, 'n', 0.0, G)
        A = np.zeros_like(L)
        mmm(1.0, evec, 't', G, 'n', 0.0, A)

        G = A * D_matrix(eval)

        mmm(1.0, G, 'n', evec, 't', 0.0, A)
        mmm(1.0, evec, 'n', A, 'n', 0.0, G)

    return G


def calculate_hamiltonian_matrix(hamiltonian, wfs, kpt,
                                 Vt_xMM=None,
                                 root=-1, add_kinetic=True,
                                 calc_on_finegd=False,
                                 bfs_finegd=None,
                                 interpolator=None):
    # XXX document parallel stuff, particularly root parameter
    if calc_on_finegd is True:
        if bfs_finegd is None:
            raise Exception('You forgot to pass bfs_finegd!')
        if interpolator is None:
            raise Exception('You forgot to pass interpolator')
        bfs = bfs_finegd
    else:
        bfs = wfs.basis_functions

    # distributed_atomic_correction works with ScaLAPACK/BLACS in general.
    # If SL is not enabled, it will not work with band parallelization.
    # But no one would want that for a practical calculation anyway.
    # dH_asp = wfs.atomic_correction.redistribute(wfs, hamiltonian.dH_asp)
    # XXXXX fix atomic corrections
    dH_asp = hamiltonian.dH_asp

    if Vt_xMM is None:
        wfs.timer.start('Potential matrix')
        if calc_on_finegd is True:
            # vt_G = hamiltonian.vt_sG[kpt.s]
            # vt_g = bfs_finegd.gd.zeros()
            # interpolator.apply(vt_G, vt_g)
            vt_g = hamiltonian.vt_sg[kpt.s]
            Vt_xMM = bfs.calculate_potential_matrices(vt_g)
        else:
            vt_G = hamiltonian.vt_sG[kpt.s]
            Vt_xMM = bfs.calculate_potential_matrices(vt_G)

        wfs.timer.stop('Potential matrix')

    if bfs.gamma and wfs.dtype == float:
        yy = 1.0
        H_MM = Vt_xMM[0]

        # make matrix hermitian
        ind_l = np.tril_indices(H_MM.shape[0], -1)
        H_MM[(ind_l[1], ind_l[0])] = H_MM[ind_l]

        # for i in range(H_MM.shape[0]):
        #     for j in range(i + 1, H_MM.shape[1]):
        #         H_MM[i][j] = H_MM[j][i]
    else:
        wfs.timer.start('Sum over cells')
        yy = 0.5
        # yy = 1.0
        k_c = wfs.kd.ibzk_qc[kpt.q]
        H_MM = (0.5 + 0.0j) * Vt_xMM[0]
        # TODO: do we need to do this for potential matrix as well?
        for sdisp_c, Vt_MM in zip(bfs.sdisp_xc[1:], Vt_xMM[1:]):
            H_MM += np.exp(2j * np.pi * np.dot(sdisp_c, k_c)) * Vt_MM
        wfs.timer.stop('Sum over cells')

    # Add atomic contribution
    #
    #           --   a     a  a*
    # H      += >   P    dH  P
    #  mu nu    --   mu i  ij nu j
    #           aij
    #
    name = wfs.atomic_correction.__class__.__name__
    wfs.timer.start(name)
    wfs.atomic_correction.calculate_projections(wfs, kpt)
    wfs.atomic_correction.calculate_hamiltonian(wfs, kpt, dH_asp,
                                                H_MM, yy)
    wfs.timer.stop(name)

    # FIXME: Why we do this?
    wfs.timer.start('Distribute overlap matrix')
    H_MM = wfs.ksl.distribute_overlap_matrix(
        H_MM, root, add_hermitian_conjugate=(yy == 0.5))
    wfs.timer.stop('Distribute overlap matrix')

    if add_kinetic:
        H_MM += wfs.T_qMM[kpt.q]
    return H_MM

