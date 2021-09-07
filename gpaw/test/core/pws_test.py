import pytest
from gpaw.core import UniformGrid, PlaneWaves
from gpaw.mpi import world, serial_comm


@pytest.mark.ci
def test_redist():
    a = 2.5
    n = 20

    # comm = world.new_communicator([world.rank])
    grid = UniformGrid(cell=[a, a, a], size=(n, n, n), comm=world)
    pw1 = PlaneWaves(ecut=10, grid=grid)
    pw2 = pw1.new(comm=serial_comm)
    f1 = pw1.empty()
    f1.data[:] = 1.0
    f2 = f1.redistribute(pw2)
