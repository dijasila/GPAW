from typing import TYPE_CHECKING

import numpy as np

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


def setup():
    # select GPU device (round-robin based on MPI rank)
    # if not set, all MPI ranks will use the same default device
    if not cupy_is_fake:
        from gpaw.mpi import rank
        device_id = rank % cupy.cuda.runtime.getDeviceCount()
        cupy.cuda.runtime.setDevice(device_id)


def as_xp(array, xp):
    if xp is np:
        if isinstance(array, np.ndarray):
            return array
        return cupy.asnumpy(array)
    if isinstance(array, np.ndarray):
        return cupy.asarray(array)
    return array
