import pytest

import numpy as np

from gpaw import GPAW
from gpaw.mpi import world
from gpaw.response import ResponseContext, ResponseGroundStateAdapter
from gpaw.response.pw_parallelization import block_partition
from gpaw.response.kspair import KohnShamKPointPairExtractor
from gpaw.response.pair_functions import SingleQPWDescriptor
from gpaw.response.pair_integrator import KPointPairPointIntegral
from gpaw.response.symmetry import PWSymmetryAnalyzer

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
    q_c = np.array([0., 0., 0.])  # Introduce parametrization XXX
    nblocks = 1  # Introduce parametrization XXX

    # ---------- Script ---------- #

    # Initialize serial ground state adapter
    context = ResponseContext()
    serial_gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files[wfs], context)
    
    # Initialize parallel ground state adapter
    calc = GPAW(gpw_files[wfs], parallel=dict(domain=1))
    nbands = calc.parameters.convergence['bands']
    parallel_gs = ResponseGroundStateAdapter(calc)

    # Set up extractors and integrals
    tcomm, kcomm = block_partition(context.comm, nblocks)
    serial_extractor = initialize_extractor(serial_gs, context, tcomm, kcomm)
    parallel_extractor = initialize_extractor(parallel_gs, context, tcomm, kcomm)
    serial_integral = initialize_integral(serial_extractor, context, q_c)
    parallel_integral = initialize_integral(parallel_extractor, context, q_c)


# ---------- Test functionality ---------- #


def initialize_extractor(gs, context, tcomm, kcomm):
    return KohnShamKPointPairExtractor(gs, context,
                                       transitions_blockcomm=tcomm,
                                       kpts_blockcomm=kcomm)


def initialize_integral(extractor, context, q_c):
    # Initialize symmetry analyzer
    gs = extractor.gs
    qpd = SingleQPWDescriptor.from_q(q_c, 1e-3, gs.gd)
    analyzer = PWSymmetryAnalyzer(gs.kd, qpd, context)

    return KPointPairPointIntegral(extractor, analyzer)

