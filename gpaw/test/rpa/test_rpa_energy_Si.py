import pytest
from gpaw.mpi import serial_comm
from gpaw import GPAW
from gpaw.xc.rpa import RPACorrelation

@pytest.mark.rpa
@pytest.mark.response
def test_qpt_rounding_error(in_tmp_dir):
    from ase.build import bulk
    atoms = bulk('Si')
    calc = GPAW(mode='pw', kpts={'gamma':True, 'size':(6,1,1)})
    atoms.calc = calc
    atoms.get_potential_energy()

    calc.diagonalize_full_hamiltonian(nbands=50)

    ecut = 50
    rpa = RPACorrelation(calc, qsym=False, nfrequencies=8, ecut=[ecut])
    E_rpa_noqsym = rpa.calculate()

    rpa = RPACorrelation(calc, qsym=True, nfrequencies=8, ecut=[ecut])
    E_rpa_qsym = rpa.calculate()

    assert E_rpa_qsym == pytest.approx(-9.753, abs=0.01)
    assert E_rpa_qsym == pytest.approx(E_rpa_noqsym)

@pytest.mark.rpa
@pytest.mark.response
def test_rpa_rpa_energy_Si(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['si_pw'], communicator=serial_comm)
    calc.diagonalize_full_hamiltonian(nbands=50)

    ecut = 50
    rpa = RPACorrelation(calc, qsym=False, nfrequencies=8, ecut=[ecut])
    E_rpa_noqsym = rpa.calculate()

    rpa = RPACorrelation(calc, qsym=True, nfrequencies=8, ecut=[ecut])
    E_rpa_qsym = rpa.calculate()

    assert E_rpa_qsym == pytest.approx(-12.61, abs=0.01)
    assert E_rpa_qsym == pytest.approx(E_rpa_noqsym)
