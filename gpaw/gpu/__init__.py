from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as sla

cupy_is_fake = True
if TYPE_CHECKING:
    import gpaw.gpu.cpupy as cupy
    import gpaw.gpu.cpupyx as cupyx
else:
    try:
        import cupy
        import cupyx
        cupy_is_fake = False
    except ImportError:
        import gpaw.gpu.cpupy as cupy
        import gpaw.gpu.cpupyx as cupyx

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
