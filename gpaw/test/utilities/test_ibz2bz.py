import pytest
from gpaw import GPAW
from gpaw.utilities.ibz2bz import ibz2bz


@pytest.mark.serial
def test_ibz2bz(bcc_li_gpw, in_tmp_dir):
    """Test ibz.gpw -> bz.gpw utility."""
    path = 'bz.gpw'
    ibz2bz(bcc_li_gpw, path)
    ef1 = GPAW(bcc_li_gpw).get_fermi_level()
    ef2 = GPAW(path).get_fermi_level()
    assert ef1 == ef2
