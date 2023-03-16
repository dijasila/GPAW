import pytest

from gpaw.mpi import world

pytestmark = pytest.mark.skipif(world.size == 1, reason='world.size == 1')


def test_parallel_extract_kptdata():
    pass
