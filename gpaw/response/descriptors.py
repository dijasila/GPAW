"""Descriptor short-cuts for the GPAW response code."""

from gpaw.response.pair_functions import SingleQPWDescriptor
from gpaw.response.pw_parallelization import PlaneWaveBlockDistributor


def as_qpd(plane_waves):
    # Construct qpd
    if isinstance(plane_waves, SingleQPWDescriptor):
        qpd = plane_waves
    else:
        assert isinstance(plane_waves, tuple)
        assert len(plane_waves) == 3
        qpd = SingleQPWDescriptor.from_q(*plane_waves)
    return qpd


def as_blockdist(parallelization):
    # Construct blockdist
    if isinstance(parallelization, PlaneWaveBlockDistributor):
        blockdist = parallelization
    else:
        assert isinstance(parallelization, tuple)
        assert len(parallelization) == 3
        blockdist = PlaneWaveBlockDistributor(*parallelization)
    return blockdist
