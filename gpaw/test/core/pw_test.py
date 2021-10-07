import numpy as np
import pytest
from gpaw.core import UniformGrid, PlaneWaves
from gpaw.mpi import world


@pytest.mark.ci
def test_redist():
    a = 2.5
    n = 20

    # comm = world.new_communicator([world.rank])
    grid = UniformGrid(cell=[a, a, a], size=(n, n, n), comm=world)
    pw1 = PlaneWaves(ecut=10, grid=grid)
    f1 = pw1.empty()
    f1.data[:] = 1.0
    f2 = f1.collect()
    assert (f2.data == 1.0).all()
    assert f2.layout.grid.comm.size == 1


def test_pw_integrate():
    a = 1.0
    grid = UniformGrid(cell=[a, a, a], size=(4, 4, 4), comm=world)
    gridc = grid.new(dtype=complex)

    g1 = grid.empty()
    g1.data[:] = 1.0

    g2 = grid.empty()
    g2.data[:] = 1.0
    g2.data += [0, 1, 0, -1]

    g3 = gridc.empty()
    g3.data[:] = 1.0

    g4 = gridc.empty()
    g4.data[:] = 1.0
    g4.data += [0, 1, 0, -1]

    g5 = gridc.empty()
    g5.data[:] = 1.0 + 1.0j
    g5.data += [0, 1, 0, -1]

    ecut = 0.5 * (2 * np.pi / a)**2 * 1.01
    for g in [g1, g2, g3, g4, g5]:
        pw = PlaneWaves(grid=g.grid, ecut=ecut)
        f = g.fft(pw=pw)

        i1 = g.integrate()
        i2 = f.integrate()
        assert i1 == i2
        assert type(i1) == g.grid.dtype

        i1 = g.integrate(g)
        i2 = f.integrate(f)
        assert i1 == i2
        assert type(i1) == g.grid.dtype
