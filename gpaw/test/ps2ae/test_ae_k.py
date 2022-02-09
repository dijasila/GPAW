import pytest
from ase.units import Bohr

from gpaw import GPAW
from gpaw.utilities.ps2ae import PS2AE


def test_ae_k(gpw_files):
    """Test normalization of non gamma-point wave functions."""
    calc = GPAW(gpw_files['bcc_li_pw_wfs'])

    if 1:
        converter = PS2AE(calc)
        ae = converter.get_wave_function(n=0, k=1, ae=True)
        norm = converter.gd.integrate((ae * ae.conj()).real) * Bohr**3
        assert norm == pytest.approx(1.0, abs=3e-5)

    ae = calc.calculation.state.ibzwfs.get_all_electron_wave_function(0, kpt=1)
    assert ae.norm2() == pytest.approx(1.0, abs=3e-5)
