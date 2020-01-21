from pathlib import Path
from typing import List

import numpy as np
from ase.units import Ha
from ase.utils import convert_string_to_fd

from gpaw.xc.tools import vxc
from .hybrid import HybridXC


def non_self_consistent_eigenvalues(calc,
                                    xcname: str,
                                    n1=0,
                                    n2=-1,
                                    kpts: List[int] = None,
                                    restart=None,
                                    txt='-',
                                    ftol=1e-6) -> np.ndarray:
    fd = convert_string_to_fd(txt)
    wfs = calc.wfs

    if n2 == -1:
        n2 = wfs.bd.nbands

    nocc = max(((kpt.f_n / kpt.weight) > ftol).sum()
               for kpt in wfs.mykpts)
    nspins = wfs.nspins
    nkpts = len(kpts)

    e_isn = np.empty((nkpts, nspins, n2 - n1))

    i0 = 0
    if restart:
        path = Path(restart)
        if path.is_file():
            data = np.loadtxt(path)
            i0 = len(data)
            print(f'Restarting from {path} (k-points: {i0})',
                  file=fd)
            e_isn[:i0].reshape((i0, -1))[:] = data

    if i0 == nkpts:
        return e_isn.reshape((1, 0, 2)) * Ha

    xc = HybridXC(xcname)
    xc.initialize_from_calculator(calc)
    xc.set_positions(calc.spos_ac)
    xc._initialize()

    if xc.xc:
        v_skn = vxc(xc.xc, calc) / Ha
        e_isn[i0:] = v_skn[:, kpts[i0:], n1:n2]

    for i, k in enumerate(kpts[i0:], i0):
        for spin in range(nspins):
            e_n = xc.calculate_eigenvalue_contribution(spin, k, n1, n2, nocc)
            e_isn[i, spin] += e_n
            e_isn[i, spin] -= calc.get_eigenvalues(spin, k) / Ha
        if restart:
            np.savetxt(path, e_isn[:i + 1].reshape((i + 1, -1)))

    return e_isn.reshape((1, 0, 2)) * Ha
