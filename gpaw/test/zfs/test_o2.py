from types import SimpleNamespace

import numpy as np
import pytest
from ase.units import Bohr as bohr

from gpaw import GPAW
from gpaw.utilities.ps2ae import PS2AE
from gpaw.zero_field_splitting.zfs import (WaveFunctions,
                                           create_compensation_charge, zfs1)


@pytest.mark.serial
def test_zfs_o2(gpw_files):
    calc = GPAW(gpw_files['o2_pw_wfs'])
    wf1 = WaveFunctions.from_calc(calc, 0)
    wf1 = wf1.view(5, 7)
    cc = create_compensation_charge(wf1, calc.spos_ac)
    D1 = zfs1(wf1, wf1, cc, False)
    print(D1)

    converter = PS2AE(calc)
    psit_nR = np.array([converter.get_wave_function(n, ae=False) * bohr**1.5
                        for n in [5, 6]])
    wf2 = WaveFunctions(psit_nR, wf1.projections, 0, calc.setups, converter.gd)
    cc = create_compensation_charge(wf2, calc.spos_ac)
    D2 = zfs1(wf2, wf2, cc, False)
    print(D2)
    for gs in [0.1, 0.05, 0.025]:
        converter = PS2AE(calc, grid_spacing=gs)
        psit_nR = np.array([converter.get_wave_function(n, ae=True) * bohr**1.5
                            for n in [5, 6]])
        wf3 = WaveFunctions(psit_nR, {}, 0, calc.setups, converter.gd)
        cc = SimpleNamespace(add=lambda a, b: None)
        D3 = zfs1(wf3, wf3, cc, False)
        print(D3)
