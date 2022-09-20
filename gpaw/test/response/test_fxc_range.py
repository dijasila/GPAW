import pytest
from gpaw import GPAW
from gpaw.xc.fxc import FXCCorrelation
from ase.units import Hartree

@pytest.mark.response
def test_xc_short_range(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['si_pw'])
    calc.diagonalize_full_hamiltonian()

    fxc = FXCCorrelation(calc,
                         xc='range_RPA',
                         range_rc=2.0)
    E_i = fxc.calculate(ecut=[2.25 * Hartree], nbands=100)

    assert E_i[0] == pytest.approx(-16.0465, abs=0.01)

