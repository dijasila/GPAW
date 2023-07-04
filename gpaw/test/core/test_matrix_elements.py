from gpaw.core import PlaneWaves, UniformGrid
from gpaw.mpi import world


def test_me():
    a = 2.5
    n = 20
    grid = UniformGrid(cell=[a, a, a], size=(n, n, n))
    pw = PlaneWaves(ecut=50, cell=grid.cell)
    f1 = pw.zeros(8, comm=world)
    if world.rank == 0:
        f1.data[:, 0] = 1
    M = f1.matrix_elements(f1)
    print(M)
    print(M.data)
