import pytest
from gpaw.test import parallel
from gpaw.mpi import world


@pytest.mark.mpi
@parallel(1, 2)
def test_MPI(in_tmp_dir):
    assert world.size == 2
    if world.rank == 0:
        with open('mpi.txt', 'w') as fd:
            fd.write('hello\n')
