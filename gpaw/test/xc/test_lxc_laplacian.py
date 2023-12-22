"""check if an error is raised if the laplacian is needed (mgga)"""
import pytest
from ase.build import molecule
from gpaw import GPAW
from gpaw.cluster import Cluster

vacuum = 4.0
h = 0.3

@pytest.mark.mgga
@pytest.mark.libxc
def test_mgga_lxc_laplacian():
    cluster = Cluster(molecule('CO'))
    cluster.minimal_box(border=vacuum, h=h)
    calc = GPAW(xc='MGGA_X_BR89+MGGA_C_TPSS', mode='fd',
                h=h, maxiter=14, txt='try.log',
                convergence={
                    'energy': 0.5,
                    'density': 1.0e-1,
                    'eigenstates': 4.0e-1})
    cluster.calc = calc
    laplacian_test = False
    try:
        cluster.get_potential_energy()
    except ValueError:
        laplacian_test = True
    assert laplacian_test == True

if __name__ == '__main__':
    test_mgga_lxc_laplacian()
