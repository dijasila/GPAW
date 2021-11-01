import pytest
from gpaw.core import UniformGrid
from gpaw.mpi import world, serial_comm


@pytest.mark.ci
def test_redist():
    a = 2.5
    n = 2

    # comm = world.new_communicator([world.rank])
    grid1 = UniformGrid(cell=[a, a, a], size=(n, n, n), comm=world)
    grid2 = grid1.new(comm=serial_comm)
    f1 = grid1.empty()
    f1.data[:] = world.rank + 1
    f2 = f1.collect()
    f3 = f1.collect(grid2)
    if world.rank == 0:
        assert (f2.data == f3.data).all()
    print(f2.data)
    f4 = f2.distribute(f1.desc)
    assert (f4.data == f1.data).all()
