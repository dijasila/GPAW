from types import SimpleNamespace
from ase.units import Bohr as bohr
import numpy as np

from gpaw import GPAW
from gpaw.zero_field_splitting import (zfs1, WaveFunctions,
                                       create_compensation_charge)
from gpaw.utilities.ps2ae import PS2AE
from gpaw.wavefunctions.pw import PWDescriptor


def test_zfs_o2(gpw_files):
    calc = GPAW(gpw_files['o2_pw_wfs'])
    wf1 = WaveFunctions.from_kpt(calc.wfs.kpt_u[0], calc.setups)
    wf1 = wf1.view(5, 7)
    cc = create_compensation_charge(wf1, calc.spos_ac)
    D1 = zfs1(wf1, wf1, cc)
    print(D1)

    converter = PS2AE(calc)
    psit_nR = np.array([converter.get_wave_function(n, ae=True) * bohr**1.5
                        for n in [5, 6]])
    pd = PWDescriptor(ecut=None, gd=converter.gd)
    wf2 = WaveFunctions(pd,
                        psit_nR,
                        {}, 0, calc.setups)
    cc = SimpleNamespace(add=lambda a, b: None)
    D2 = zfs1(wf2, wf2, cc)
    print(D2)
