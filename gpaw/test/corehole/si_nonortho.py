import pytest
import numpy as np

from gpaw import GPAW
from gpaw.test import gen
from gpaw.xas import XAS
import gpaw.mpi as mpi


@pytest.mark.skip(reason='TODO')
def test_si_nonortho(gpw_files):
    # Generate setup for oxygen with half a core-hole:
    gen('Si', name='hch1s', corehole=(1, 0, 0.5))

    # restart from file
    # maybe these fixtures work. But they were added when this test was
    # skipped so \o/ code moved to fixtures: si_corehole_sym,
    # si_corehole_nosym_pw, si_corehole_sym_pw
    calc1 = GPAW(gpw_files['si_corehole_sym_pw'])
    calc2 = GPAW(gpw_files['si_corehole_nosym_pw'])
    if mpi.size == 1:
        xas1 = XAS(calc1)
        x, y1 = xas1.get_spectra()
        xas2 = XAS(calc2)
        x2, y2 = xas2.get_spectra(E_in=x)

        assert (np.sum(abs(y1 - y2)[0, :500]**2) < 5e-9)
        assert (np.sum(abs(y1 - y2)[1, :500]**2) < 5e-9)
        assert (np.sum(abs(y1 - y2)[2, :500]**2) < 5e-9)
