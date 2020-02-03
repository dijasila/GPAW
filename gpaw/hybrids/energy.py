import json
from pathlib import Path
from typing import List, Union

import numpy as np
from ase.units import Ha

from gpaw import GPAW
from gpaw.xc.tools import vxc
from .hybrid import HybridXC


def non_self_consistent_energy(calc: Union[GPAW, str, Path],
                               xcname: str,
                               ftol=1e-9) -> float:
    """Calculate non self-consistent energy for Hybrid functional.

    Based on a self-consistent DFT calculation (calc).  EXX integrals involving
    states with occupation numbers less than ftol are skipped.

    >>> eig_dft, vxc_dft, vxc_hyb = non_self_consistent_eigenvalues(...)
    >>> eig_hyb = eig_dft - vxc_dft + vxc_hyb
    """

    if not isinstance(calc, GPAW):
        calc = GPAW(Path(calc), txt=None, parallel={'band': 1, 'kpt': 1})

    wfs = calc.wfs
    nocc = max(((kpt.f_n / kpt.weight) > ftol).sum()
               for kpt in wfs.mykpts)
    nspins = wfs.nspins

    xc = HybridXC(xcname)
    xc.initialize_from_calculator(calc)
    xc.set_positions(calc.spos_ac)
    xc._initialize()

    evc = 0.0
    evv = 0.0
    for spin in range(nspins):
        e1, e2 = xc.calculate_energy(spin, nocc)
        evc += e1
        evv += e2

    return evc, evv
