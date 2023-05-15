import pytest
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.xc.fxc import FXCCorrelation
# from gpaw.mpi import world,


@pytest.fixture
def ni_gpw(gpw_files, scalapack):
    return gpw_files['ni_pw_kpts333_wfs']


@pytest.mark.rpa
@pytest.mark.response
def test_ralda_ralda_energy_Ni(in_tmp_dir, ni_gpw):
    #if world.rank == 0:

    #world.barrier()
    # gpw_files
    #calc = 
    #print(calc)

    rpa = FXCCorrelation(ni_gpw, xc='RPA',
                         nfrequencies=8, skip_gamma=True,
                         ecut=[50])
    E_rpa = rpa.calculate()

    ralda = FXCCorrelation(ni_gpw, xc='rALDA', unit_cells=[2, 1, 1],
                           nfrequencies=8, skip_gamma=True,
                           ecut=[50])
    E_ralda = ralda.calculate()

    rapbe = FXCCorrelation(ni_gpw, xc='rAPBE', unit_cells=[2, 1, 1],
                           nfrequencies=8, skip_gamma=True,
                           ecut=[50])
    E_rapbe = rapbe.calculate()

    assert E_rpa == pytest.approx(-7.827, abs=0.01)
    assert E_ralda == pytest.approx(-7.501, abs=0.01)
    assert E_rapbe == pytest.approx(-7.444, abs=0.01)
