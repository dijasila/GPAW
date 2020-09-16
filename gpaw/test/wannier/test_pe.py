import pytest
from gpaw import GPAW
from gpaw.wannier import calculate_overlaps


@pytest.mark.serial
def test_pe(gpw_files):
    calc = GPAW(gpw_files['c2h4_pw_nosym_wfs'])
    o = calculate_overlaps(calc, n2=6)
    calc = GPAW(gpw_files['c6h12_pw_wfs'])
    o = calculate_overlaps(calc, n2=3 * 6)
    w = o.localize()
    print(w)
