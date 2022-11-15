import scipy.linalg as sla


def cholesky(a, lower, overwrite_a, check_finite):
    from gpaw.cpupy import CuPyArray
    return CuPyArray(sla.cholesky(a.data,
                                  lower=lower,
                                  overwrite_a=overwrite_a,
                                  check_finite=check_finite))


def inv(a, overwrite_a, check_finite):
    from gpaw.cpupy import CuPyArray
    return CuPyArray(sla.inv(a.data,
                             overwrite_a=overwrite_a,
                             check_finite=check_finite))
