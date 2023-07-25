from gpaw.core import PlaneWaves, UniformGrid
from gpaw.mpi import world
import pytest


def comms():
    for s in [1, 2, 4, 8]:
        if s > world.size:
            return
        domain_comm = world.new_communicator(
            range(world.rank // s * s, world.rank // s * s + s))
        band_comm = world.new_communicator(
            range(world.rank % s, world.size, s))
        yield domain_comm, band_comm


def func(f):
    g = f.copy()
    g.data *= 2.0
    return g


@pytest.mark.parametrize('domain_comm, band_comm', list(comms()))
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('nbands', [1, 7, 21])
@pytest.mark.parametrize('function', [None, func])
def test_me(domain_comm, band_comm, dtype, nbands, function):
    a = 2.5
    n = 20
    grid = UniformGrid(cell=[a, a, a], size=(n, n, n))
    desc = PlaneWaves(ecut=50, cell=grid.cell)
    desc = desc.new(comm=domain_comm, dtype=dtype)
    f = desc.empty(nbands, comm=band_comm)
    f.randomize()
    M = f.matrix_elements(f, function=function)

    f1 = f.gathergather()
    M2 = M.gather()
    if f1 is not None:
        M1 = f1.matrix_elements(f1, function=function)
        M1.tril2full()
        M2.tril2full()
        dM = M1.data - M2.data
        assert abs(dM).max() < 1e-11

    if function is None:
        g = f.new()
        g.randomize()
        M = f.matrix_elements(g)

        f1 = f.gathergather()
        g1 = g.gathergather()
        M2 = M.gather()
        if f1 is not None:
            M1 = f1.matrix_elements(g1)
            M1.tril2full()
            M2.tril2full()
            dM = M1.data - M2.data
            assert abs(dM).max() < 1e-11


if __name__ == '__main__':
    from gpaw.mpi import serial_comm
    # test_me(world, serial_comm, float, 7, None)
    test_me(serial_comm, world, float, 7, None)
