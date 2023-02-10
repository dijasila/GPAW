import os
import subprocess

import pytest
from ase.io import write
from ase.units import Bohr

from gpaw import GPAW
from gpaw.utilities.bader import read_bader_charges
from gpaw.utilities.ps2ae import PS2AE


@pytest.mark.serial
def test_bader(gpw_files, in_tmp_dir):
    """Test bader analysis on interpolated density."""
    calc = GPAW(gpw_files['c2h4_pw_nosym'])
    if os.environ.get('GPAW_NEW'):
        nt_sR = calc.calculation.densities().pseudo_densities()
        ne = nt_sR.integrate().sum()
        density = nt_sR.data.sum(0)
    else:
        converter = PS2AE(calc)
        density = converter.get_pseudo_density()
        ne = density.sum() * converter.dv

    assert ne == pytest.approx(12, abs=1e-5)

    write('density.cube', calc.atoms, data=density * Bohr**3)
    try:
        subprocess.run('bader -p all_atom density.cube'.split())
    except FileNotFoundError:
        return

    charges = read_bader_charges()
    assert charges[:2].ptp() == 0.0
    assert charges[2:].ptp() < 0.001
    assert charges.sum() == pytest.approx(12.0, abs=0.0001)
