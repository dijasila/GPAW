import os

import pytest
from ase.units import Bohr

from gpaw.calculator import GPAW as GPAW1
from gpaw.mpi import size
from gpaw.new.ase_interface import GPAW as GPAW2
from gpaw.utilities.ps2ae import PS2AE


@pytest.mark.parametrize('name, tol',
                         [('bcc_li_pw_wfs', 3e-5),
                          ('bcc_li_fd_wfs', 4e-4)])
def test_ae_k(gpw_files, name, tol):
    """Test normalization of non gamma-point wave functions."""

    if os.environ.get('GPAW_NEW'):
        # New API:
        if size > 1:
            return
        calc = GPAW2(gpw_files[name])
        ae = calc.calculation.state.ibzwfs.get_all_electron_wave_function(
            0, kpt=1, grid_spacing=0.1)
        assert ae.norm2() == pytest.approx(1.0, abs=tol)
        return

    # Old API:
    calc = GPAW1(gpw_files[name])
    converter = PS2AE(calc)
    ae = converter.get_wave_function(n=0, k=1, ae=True) * Bohr**1.5
    norm = converter.gd.integrate((ae * ae.conj()).real)
    assert norm == pytest.approx(1.0, abs=tol)
