"""check if an error is raised if the laplacian is needed (mgga)"""
import pytest
from gpaw.xc import LibXC


@pytest.mark.mgga
@pytest.mark.libxc
def test_mgga_lxc_laplacian():
    """Check for raised error."""
    laplacian_test = False
    try:
        LibXC('MGGA_X_BR89+MGGA_C_TPSS')
    except ValueError:
        laplacian_test = True
    assert laplacian_test


if __name__ == '__main__':
    test_mgga_lxc_laplacian()
