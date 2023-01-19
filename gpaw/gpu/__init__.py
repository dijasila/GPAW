from typing import TYPE_CHECKING

import numpy as np

cupy_is_fake = True
is_hip = False

if TYPE_CHECKING:
    import gpaw.gpu.cpupy as cupy
    import gpaw.gpu.cpupyx as cupyx
else:
    try:
        import cupy
        import cupyx
        from cupy.cuda import runtime
        is_hip = runtime.is_hip
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


def cupy_eigh(a, UPLO):
    """HIP version of eigh() is too slow for now."""
    from scipy.linalg import eigh
    if not is_hip:
        return cupy.linalg.eigh(a, UPLO)
    eigs, evals = eigh(cupy.asnumpy(a), UPLO)
    return cupy.asarray(eigs), cupy.asarray(evals)
