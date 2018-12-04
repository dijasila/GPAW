import numpy as np
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.blas import mmm

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
                          evec).real, evec, eval
        else:
            return np.dot(evec.T.conj() * np.exp(-1.0j * eval),
                          evec), evec, eval

    else:
        if A.dtype == float:
            return np.dot(evec.T.conj() * np.exp(-1.0j * eval),
                          evec).real
        else:
            return np.dot(evec.T.conj() * np.exp(-1.0j * eval),
                          evec)


def expm_ed_numpy(A, evalevec=False):
    eval, evec = np.linalg.eigh(1.0j * A)
    if evalevec:
        if A.dtype == float:
            return np.dot(evec * np.exp(-1.0j*eval),
                          evec.T.conj()).real, evec.T.conj(), eval
        else:
            return np.dot(evec * np.exp(-1.0j * eval),
                          evec.T.conj()), evec.T.conj(), eval

    else:
        if A.dtype == float:
            return np.dot(evec * np.exp(-1.0j * eval),
                          evec.T.conj()).real
        else:
            return np.dot(evec * np.exp(-1.0j * eval),
                          evec.T.conj())


def expm_ed2(A, evalevec=False):

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

    x = np.ascontiguousarray(evec.T.conj() * np.exp(-1.0j * eval))
    exp_mat = np.empty_like(evec)
    mmm(1.0, x, 'N', evec, 'N', 0.0, exp_mat)

    if evalevec:
        if A.dtype == float:
            return exp_mat.real, evec, eval
        else:
            return exp_mat, evec, eval

    else:
        if A.dtype == float:
            return exp_mat.real
        else:
            return exp_mat


def D_matrix(omega):

    m = omega.shape[0]
    u_m = np.ones(shape=(m, m))

    u_m = omega[:, np.newaxis] * u_m - omega * u_m

    with np.errstate(divide='ignore', invalid='ignore'):
        u_m = 1.0j * np.divide(np.exp(-1.0j * u_m) - 1.0, u_m)

    u_m[np.isnan(u_m)] = 1.0
    u_m[np.isinf(u_m)] = 1.0

    return u_m

