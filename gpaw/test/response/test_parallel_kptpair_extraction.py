import pytest

from gpaw.mpi import world
from gpaw.test.response.test_chiks import generate_system_s

pytestmark = pytest.mark.skipif(world.size == 1, reason='world.size == 1')


# ---------- Actual tests ---------- #


@pytest.mark.response
@pytest.mark.parametrize('system', generate_system_s())
def test_parallel_extract_kptdata(in_tmp_dir, gpw_files, system):
    pass
