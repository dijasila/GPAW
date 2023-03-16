import pytest

from gpaw import GPAW
from gpaw.mpi import world
from gpaw.response import ResponseContext, ResponseGroundStateAdapter
from gpaw.test.response.test_chiks import generate_system_s

pytestmark = pytest.mark.skipif(world.size == 1, reason='world.size == 1')


# ---------- Actual tests ---------- #


@pytest.mark.response
@pytest.mark.parametrize('system', generate_system_s())
def test_parallel_extract_kptdata(in_tmp_dir, gpw_files, system):
    """Test that the KohnShamKPointPair data extracted from a serial and a
    parallel calculator object is identical."""

    # ---------- Inputs ---------- #

    wfs, spincomponent, _, _, _ = system
    # nblocks XXX

    # ---------- Script ---------- #

    # Initialize serial ground state adapter
    context = ResponseContext()
    serial_gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files[wfs], context)
    
    # Initialize parallel ground state adapter
    calc = GPAW(gpw_files[wfs], parallel=dict(domain=1))
    nbands = calc.parameters.convergence['bands']
    parallel_gs = ResponseGroundStateAdapter(calc)
