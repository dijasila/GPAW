import pytest
from gpaw import GPAW
from gpaw.wannier import calculate_overlaps


@pytest.mark.serial
def test_pe(gpw_files):
    calc = GPAW(gpw_files['c2h4_pw_nosym_wfs'])
    o = calculate_overlaps(calc, n2=6, nwannier=6,
                           projections={'C': 's', 'H': 's'})
    o.localize_w90('pe', num_iter=100, guiding_centres=True)
    return




    calc = GPAW(gpw_files['c6h12_pw_wfs'])
    o = calculate_overlaps(calc, n2=3 * 6, nwannier=18)
    w = o.localize_er()
    print(w)
