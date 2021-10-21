from types import SimpleNamespace

import pytest
import numpy as np
from ase.units import Bohr as bohr

from gpaw import GPAW
from gpaw.utilities.ps2ae import PS2AE
from gpaw.zero_field_splitting import WaveFunctions, zfs1, zfs


def test_zfs_o2(gpw_files):
    calc = GPAW(gpw_files['o2_pw_wfs'])
    D1 = zfs(calc) * 1e6  # ueV
    print(D1)
    assert D1 == pytest.approx(np.diag([117, -59, -59]), abs=1)

    # Poor man's PAW correction:
    converter = PS2AE(calc)
    psit_nR = np.array([converter.get_wave_function(n, ae=True) * bohr**1.5
                        for n in [5, 6]])
    wf2 = WaveFunctions(psit_nR, {}, 0, calc.setups, converter.gd)
    cc = SimpleNamespace(add=lambda a, b: None)
    D2 = zfs1(wf2, wf2, cc) * 1e6  # uev
    print(D2 - D1)
    assert D2 - D1 == pytest.approx(np.diag([14, -7, -7]), abs=1)
