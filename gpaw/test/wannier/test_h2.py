import pytest
from gpaw import GPAW
from gpaw.wannier import calculate_overlaps


@pytest.mark.parametrize('mode', ['pw', 'fd', 'lcao'])
def test_h2(gpw_files, mode):
    calc = GPAW(gpw_files[f'h2_{mode}_wfs'])
    overlaps = calculate_overlaps(calc, n1=0, n2=1, nwannier=1)
    wan = overlaps.localize_er(verbose=True)
    print(wan.centers)
    x = calc.atoms.positions[:, 0].mean()
    assert wan.centers[0, 0] == pytest.approx(x, abs=1e-7)
    overlaps.localize_w90()
