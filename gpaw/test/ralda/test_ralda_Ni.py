import pytest
from ase.build import bulk
from ase.dft.kpoints import monkhorst_pack
from gpaw import GPAW, FermiDirac
from gpaw.xc.fxc import FXCCorrelation
from gpaw.mpi import world, serial_comm


@pytest.mark.rpa
@pytest.mark.response
def test_ralda_ralda_energy_Ni(in_tmp_dir, scalapack):
    if world.rank == 0:
        Ni = bulk('Ni', 'fcc')
        Ni.set_initial_magnetic_moments([0.7])

        kpts = monkhorst_pack((3, 3, 3))

        calc = GPAW(mode='pw',
                    kpts=kpts,
                    occupations=FermiDirac(0.001),
                    setups={'Ni': '10'},
                    communicator=serial_comm)

        Ni.calc = calc
        Ni.get_potential_energy()
        calc.diagonalize_full_hamiltonian()
        calc.write('Ni.gpw', mode='all')

    world.barrier()

    rpa = FXCCorrelation('Ni.gpw', xc='RPA',
                         nfrequencies=8, skip_gamma=True,
                         ecut=[50])
    E_rpa = rpa.calculate()

    ralda = FXCCorrelation('Ni.gpw', xc='rALDA', unit_cells=[2, 1, 1],
                           nfrequencies=8, skip_gamma=True,
                           ecut=[50])
    E_ralda = ralda.calculate()

    rapbe = FXCCorrelation('Ni.gpw', xc='rAPBE', unit_cells=[2, 1, 1],
                           nfrequencies=8, skip_gamma=True,
                           ecut=[50])
    E_rapbe = rapbe.calculate()

    assert E_rpa == pytest.approx(-7.827, abs=0.01)
    assert E_ralda == pytest.approx(-7.501, abs=0.01)
    assert E_rapbe == pytest.approx(-7.444, abs=0.01)
