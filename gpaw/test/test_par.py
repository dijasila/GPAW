import pytest
from gpaw.mpi import world


@pytest.mark.parallel
def test_MPI(in_tmp_dir):
    assert world.size == 2
    if world.rank == 0:
        with open('mpi.txt', 'w') as fd:
            fd.write('hello\n')
