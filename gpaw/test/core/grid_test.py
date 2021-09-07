from gpaw.core import UniformGrid
from gpaw.mpi import world, serial_comm


def test_redist():
    a = 2.5
    n = 20

    # comm = world.new_communicator([world.rank])
    grid1 = UniformGrid(cell=[a, a, a], size=(n, n, n), comm=world)
    grid2 = grid1.new(comm=serial_comm)
    f1 = grid1.empty()
    f1.data[:] = 1.0
    f2 = f1.redistribute(grid2)
