import pytest
from gpaw.core import UniformGrid
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
