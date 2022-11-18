import numpy as np
import scipy.linalg as sla


def cholesky(a, lower, overwrite_a, check_finite):
    from gpaw.cpupy import CuPyArray
    return CuPyArray(sla.cholesky(a._data,
                                  lower=lower,
                                  overwrite_a=overwrite_a,
                                  check_finite=check_finite))


def inv(a, overwrite_a, check_finite):
    from gpaw.cpupy import CuPyArray
    return CuPyArray(sla.inv(a._data,
                             overwrite_a=overwrite_a,
                             check_finite=check_finite))


def eigh(a, UPLO):
    from gpaw.cpupy import CuPyArray
    eigvals, eigvecs = np.linalg.eigh(a._data, UPLO)
    return CuPyArray(eigvals), CuPyArray(eigvecs)
