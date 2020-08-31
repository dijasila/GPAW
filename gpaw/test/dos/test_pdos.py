"""PDOS tests."""
from gpaw import GPAW


def test_li_pdos_pxyz(bcc_li_gpw):
    """Test pdos method with and without m."""
    dos = GPAW(bcc_li_gpw).dos()
    p1 = dos.pdos(a=0, l=1).get_weights()
    p2 = sum(dos.pdos(a=0, l=1, m=m).get_weights() for m in range(3))
    assert abs(p1 - p2).max() < 1e-7
