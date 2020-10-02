"""PDOS tests."""
from gpaw import GPAW


def test_li_pdos_pxyz(gpw_files):
    """Test pdos method with and without m."""
    dos = GPAW(gpw_files['bcc_li_pw']).dos()
    energies = dos.get_energies(npoints=100)
    p1 = dos.pdos(energies, a=0, l=1).get_weights()
    p2 = sum(dos.pdos(energies, a=0, l=1, m=m).get_weights() for m in range(3))
    assert abs(p1 - p2).max() < 1e-7
