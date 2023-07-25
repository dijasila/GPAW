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


@pytest.mark.parametrize('domain_comm, band_comm', list(comms()))
def test_me(domain_comm, band_comm):
    a = 2.5
    n = 20
    grid = UniformGrid(cell=[a, a, a], size=(n, n, n))
    desc = PlaneWaves(ecut=50, cell=grid.cell)
    desc = desc.new(comm=domain_comm)
    f = desc.empty(8, comm=band_comm)
    f.randomize()
    M = f.matrix_elements(f)

    f1 = f.gathergather()
    M2 = M.gather()
    if f1 is not None:
        M1 = f1.matrix_elements(f1)
        M1.tril2full()
        M2.tril2full()
        dM = M1.data - M2.data
        assert abs(dM).max() < 1e-12
