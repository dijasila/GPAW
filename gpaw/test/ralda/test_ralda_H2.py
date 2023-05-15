import pytest
from ase import Atoms

from gpaw import GPAW, PW
from gpaw.mpi import world
from gpaw.xc.fxc import FXCCorrelation




@pytest.mark.rpa
@pytest.mark.response
def test_ralda_ralda_energy_H2(in_tmp_dir, gpw_files, scalapack):
    gpw = gpw_files['h2_pw210_rmmdiis_wfs']

    ralda = FXCCorrelation(gpw, xc='rALDA', nblocks=min(4, world.size),
                           ecut=[200])
    E_ralda_H2 = ralda.calculate()

    rapbe = FXCCorrelation(gpw, xc='rAPBE', ecut=[200])
    E_rapbe_H2 = rapbe.calculate()

    assert E_ralda_H2 == pytest.approx(-0.8411, abs=0.001)
    assert E_rapbe_H2 == pytest.approx(-0.7233, abs=0.001)


def test_more_stuff(in_tmp_dir, gpw_files, scalapack):
    gpw = gpw_files['h_pw210_rmmdiis_wfs']
    ralda = FXCCorrelation(gpw, xc='rALDA', ecut=[200])
    E_ralda_H = ralda.calculate()

    rapbe = FXCCorrelation(gpw, xc='rAPBE', nblocks=min(4, world.size),
                           ecut=[200])
    E_rapbe_H = rapbe.calculate()

    assert E_ralda_H == pytest.approx(0.0029, abs=0.0001)
    assert E_rapbe_H == pytest.approx(0.0161, abs=0.0001)
