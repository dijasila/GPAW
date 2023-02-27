from __future__ import annotations
import contextlib
from time import time
from typing import TYPE_CHECKING

import numpy as np

cupy_is_fake = True
"""True if :mod:`cupy` has been replaced by ``gpaw.gpu.cpupy``"""

is_hip = False
"""True if we are using HIP"""

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

__all__ = ['cupy', 'cupyx', 'as_xp', 'synchronize']


def synchronize():
    if not cupy_is_fake:
        cupy.cuda.runtime.deviceSynchronize()


def setup():
    # select GPU device (round-robin based on MPI rank)
    # if not set, all MPI ranks will use the same default device
    if not cupy_is_fake:
        from gpaw.mpi import rank
        device_id = rank % cupy.cuda.runtime.getDeviceCount()
        cupy.cuda.runtime.setDevice(device_id)


def as_xp(array, xp):
    """Transfer array to CPU or GPU (if not already there).

    Parameters
    ==========
    array:
        Numpy or CuPy array.
    xp:
        :mod:`numpy` or :mod:`cupy`.
    """
    if xp is np:
        if isinstance(array, np.ndarray):
            return array
        return cupy.asnumpy(array)
    if isinstance(array, np.ndarray):
        return cupy.asarray(array)
    return array


def cupy_eigh(a: cupy.ndarray, UPLO: str) -> tuple[cupy.ndarray, cupy.ndarray]:
    """Wrapper for ``eigh()``.

    HIP-GPU version is too slow for now so we do it on the CPU.
    """
    from scipy.linalg import eigh
    if not is_hip:
        return cupy.linalg.eigh(a, UPLO=UPLO)
    eigs, evals = eigh(cupy.asnumpy(a), lower=(UPLO == 'L'))
    return cupy.asarray(eigs), cupy.asarray(evals)


@contextlib.contextmanager
def T():
    t1 = time()
    yield
    synchronize()
    t2 = time()
    print(f'{(t2 - t1) * 1e9:_.3f} ns')


class CuPyMPI:
    """Quick'n'dirty wrapper to make things work without a GPU-aware MPI."""
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size

    def sum(self, array):
        if isinstance(array, float):
            return self.comm.sum(array)
        if isinstance(array, np.ndarray):
            self.comm.sum(array)
            return
        a = array.get()
        self.comm.sum(a)
        array[:] = cupy.asarray(a)

    def max(self, array):
        self.comm.max(array)

    def all_gather(self, a, b):
        self.comm.all_gather(a, b)

    def gather(self, a, rank, b):
        if isinstance(a, np.ndarray):
            self.comm.gather(a, rank, b)
        else:
            if rank == self.rank:
                c = np.empty(b.shape, b.dtype)
            else:
                c = None
            self.comm.gather(a.get(), rank, c)
            if rank == self.rank:
                b[:] = cupy.asarray(c)

    def receive(self, a, rank, tag):
        b = np.empty(a.shape, a.dtype)
        self.comm.receive(b, rank, tag)
        a[:] = cupy.asarray(b)

    def send(self, a, rank, tag, block):
        1 / 0
