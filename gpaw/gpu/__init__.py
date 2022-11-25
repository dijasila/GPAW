import numpy as np

try:
    import cupy
    import cupyx
except ImportError:
    import gpaw.gpu.cupy as cupy  # type: ignore
    import gpaw.gpu.cupyx as cupyx  # type: ignore

__all__ = ['cupy', 'cupyx', 'as_xp']


def as_xp(array, xp):
    if xp is np:
        if isinstance(array, np.ndarray):
            return array
        return cupy.asnumpy(array)
    if isinstance(array, np.ndarray):
        return cupy.asarray(array)
    return array
