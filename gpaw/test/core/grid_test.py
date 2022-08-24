from math import pi

import pytest

from gpaw.core import UniformGrid
from gpaw.fd_operators import Laplace
from gpaw.mpi import world


@pytest.mark.ci
def test_redist():
    a = 2.5
    n = 2

    # comm = world.new_communicator([world.rank])
    grid1 = UniformGrid(cell=[a, a, a], size=(n, n, n), comm=world)
    f1 = grid1.empty()
    f1.data[:] = world.rank + 1
    f2 = f1.gather()
    f3 = f1.gather(broadcast=True)
    if world.rank == 0:
        assert (f2.data == f3.data).all()
    print(f2)
    f4 = f1.new()
    f4.scatter_from(f2)
    assert (f4.data == f1.data).all()


def test_complex_laplace():
    a = 2.5
    grid = UniformGrid(cell=[a, a, a],
                       size=(24, 8, 8),
                       kpt=[1 / 3, 0, 0],
                       comm=world)
    f = grid.empty()
    f.data[:] = 1.0
    f.multiply_by_eikr()
    lap = Laplace(grid._gd, n=2, dtype=complex)
    g = grid.empty()
    lap.apply(f.data, g.data, grid.phase_factors_cd)
    k = 2 * pi / a / 3
    error = g.data.conj() * f.data + k**2
    assert abs(error).max() == pytest.approx(0.0, abs=1e-6)
