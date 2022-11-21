import numpy as np
import scipy.linalg as sla

try:
    import cupy
except ImportError:
    import gpaw.gpu.cupy as cupy  # type: ignore

__all__ = ['cupy', 'as_xp']


def as_xp(array, xp):
    if xp is np:
        if isinstance(array, np):
            return array
        return cupy.asnumpy(array)
    if isinstance(array, np):
        return cupy.asarray(array)
    return array


def eigh(xp,
         a, b,
         lower,
         check_finite,
         overwrite_b):
    if xp is cupy:
        a = a._data
        b = b._data
    e, v = sla.eigh(a, b,
                    lower=lower,
                    check_finite=check_finite,
                    overwrite_b=overwrite_b)
    if xp is np:
        return e, v
    return cupy.ndarray(e), cupy.ndarray(v)
