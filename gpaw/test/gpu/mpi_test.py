from gpaw.gpu import cupy as cp
from gpaw.mpi import world


def test_mpi():
    a = cp.ones(1)
    world.sum(a)
    assert a[0].get() == world.size
