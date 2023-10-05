import pytest
from gpaw import GPAW
from gpaw.xc.rpa import RPACorrelation
from gpaw.test import equal


@pytest.mark.rpa
@pytest.mark.response
def test_rpa_rpa_energy_Na(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['na_pw'])
    ecut = 120
    rpa = RPACorrelation(calc, txt=f'rpa_{ecut}s.txt', ecut=[ecut])
    E = rpa.calculate()
    equal(E, -1.106, 0.005)
