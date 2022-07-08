import pytest
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.mpi import serial_comm
from gpaw.xc.rpa import RPACorrelation


@pytest.mark.response
def test_rpa_rpa_energy_Si(in_tmp_dir):
    a0 = 5.43
    Si = bulk('Si', a=a0)

    calc = GPAW(mode='pw',
                kpts={'size': (2, 2, 2), 'gamma': True},
                occupations=FermiDirac(0.001),
                communicator=serial_comm)
    Si.calc = calc
    Si.get_potential_energy()
    calc.diagonalize_full_hamiltonian(nbands=50)

    ecut = 50
    rpa = RPACorrelation(calc, qsym=False, nfrequencies=8)
    E_rpa_noqsym = rpa.calculate(ecut=[ecut])

    rpa = RPACorrelation(calc, qsym=True, nfrequencies=8)
    E_rpa_qsym = rpa.calculate(ecut=[ecut])

    assert E_rpa_qsym == pytest.approx(-12.61, abs=0.01)
    assert E_rpa_qsym == pytest.approx(E_rpa_noqsym)
