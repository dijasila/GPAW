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
    assert f2.pw.grid.comm.size == 1
