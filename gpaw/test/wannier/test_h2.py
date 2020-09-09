import pytest
from gpaw import GPAW
from gpaw.wannier import calculate_overlaps


@pytest.mark.parametrize('mode', ['pw', 'fd', 'lcao'])
def test_h2(gpw_files, mode):
    calc = GPAW(gpw_files[f'h2_{mode}_wfs'])
    overlaps = calculate_overlaps(calc, n1=0, n2=1)
    wan = overlaps.localize(verbose=True)
    print(wan.centers)
