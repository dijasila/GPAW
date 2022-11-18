import numpy as np

try:
    import cupy
except ImportError:
    import gpaw.cpupy as cupy  # type: ignore


def as_xp(array, xp):
    if xp is np:
        if isinstance(array, np):
            return array
        return cupy.asnumpy(array)
    if isinstance(array, np):
        return cupy.asarray(array)
    return array
