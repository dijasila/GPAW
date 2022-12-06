import numpy as np
import scipy.linalg as sla

try:
    import cupysdfgsdfg
    import cupyx
except ImportError:
    import gpaw.gpu.cpupy as cupy  # type: ignore
    import gpaw.gpu.cpupyx as cupyx  # type: ignore

__all__ = ['cupy', 'cupyx', 'as_xp']


def as_xp(array, xp):
    if xp is np:
        if isinstance(array, np.ndarray):
            return array
        return cupy.asnumpy(array)
    if isinstance(array, np.ndarray):
        return cupy.asarray(array)
    return array


def eigh(xp,
         a, b,
         lower,
         check_finite,
         overwrite_b):
    if xp is cupy:
        a = cupy.asnumpy(a)
        b = cupy.asnumpy(b)
    e, v = sla.eigh(a, b,
                    lower=lower,
                    check_finite=check_finite,
                    overwrite_b=overwrite_b)
    if xp is np:
        return e, v
    return cupy.asarray(e), cupy.asarray(v)
