import json
from pathlib import Path
from typing import List, Union, Tuple, Generator

import numpy as np
from ase.units import Ha

from gpaw import GPAW
from gpaw.xc import XC
from gpaw.xc.tools import vxc


def non_self_consistent_eigenvalues(calc: Union[GPAW, str, Path],
                                    xcname: str,
                                    n1=0,
                                    n2=0,
                                    kpts: List[int] = None,
                                    restart=None,
                                    ftol=1e-9) -> np.ndarray:
    """Calculate non self-consistent eigenvalues for Hybrid functional.

    Based on a self-consistent DFT calculation (calc).  Only eigenvalues n1 to
    n2 - 1 for the IBZ indices in kpts are calculated (default is all bands
    and all k-points). EXX integrals involving
    states with occupation numbers less than ftol are skipped.  Use
    restart=name.json to get snapshots for each k-point finished.

    returns three (nspins, nkpts, n2 - n1) shaped ndarrays with contributuons
    to the eigenvalues in eV:

    >>> eig_dft, vxc_dft, vxc_hyb = non_self_consistent_eigenvalues(...)
    >>> eig_hyb = eig_dft - vxc_dft + vxc_hyb
    """

    if not isinstance(calc, GPAW):
        calc = GPAW(Path(calc), txt=None, parallel={'band': 1, 'kpt': 1})

    wfs = calc.wfs

    if n2 <= 0:
        n2 += wfs.bd.nbands

    if kpts is None:
        kpts = np.arange(wfs.kd.nibzkpts)

    nkpts = len(kpts)

    dct = {}
    if restart:
        path = Path(restart)
        if path.is_file():
            dct = json.loads(path.read_text())

    xcname, exx_fraction, omega = parse_name(xcname)
    xc = XC(xcname)

    if 'v_dft_sin' in dct:
        v_dft_sin = np.array(dct['v_dft_sin'])
        e_dft_sin = np.array(dct['e_dft_sin'])
        v_hyb_sl_sin = np.array(dct['v_hyb_sl_sin'])  # semi-local part
    else:
        e_dft_sin, v_dft_sin, v_hyb_sl_sin = _semi_local(xc, calc)
        if wfs.world.rank == 0 and restart:
            dct = {'v_dft_sin': v_dft_sin.tolist(),
                   'e_dft_sin': e_dft_sin.tolist(),
                   'v_hyb_sl_sin': v_hyb_sl_sin.tolist()}
            path.write_text(json.dumps(dct))

    # Non-local hybrid contribution:
    v_hyb_nl_sin = np.empty_like(v_hyb_sl_sin)
    if 'v_hyb_nl_sin' in dct:
        v_sin = np.array(dct['v_hyb_nl_sin'])
        i0 = v_sin.shape[1]
        v_hyb_nl_sin[:, :i0] = v_sin
    else:
        i0 = 0

    if i0 < nkpts:
        i = i0
        for v_hyb_nl_sn in _non_local(calc, xc, n1, n2, kpts[i0:], ftol):
            v_hyb_nl_sin[:, i] = v_hyb_nl_sn
            i += 1
            if wfs.world.rank == 0 and restart:
                dct['v_hyb_nl_sin'] = v_hyb_nl_sin[:, :i].tolist()
                path.write_text(json.dumps(dct))

    return e_dft_sin, v_dft_sin, v_hyb_sl_sin + v_hyb_nl_sin


def _semi_local(calc: GPAW,
                xc,
                n1: int,
                n2: int,
                kpts: List[int]
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wfs = calc.wfs
    nspins = wfs.nspins

    e_dft_sin = np.array([[calc.get_eigenvalues(k, spin)[n1:n2]
                           for k in kpts]
                          for spin in range(nspins)])
    v_dft_sin = vxc(calc)[:, kpts, n1:n2]
    v_hyb_sl_sin = vxc(calc, xc)[:, kpts, n1:n2]
    return e_dft_sin, v_dft_sin, v_hyb_sl_sin


def _non_local(calc: GPAW,
               xc,
               n1: int,
               n2: int,
               kpts: List[int],
               ftol: float
               ) -> Generator[np.ndarray, None, None]:
    wfs = calc.wfs
    nocc = max(((kpt.f_n / kpt.weight) > ftol).sum()
               for kpt in wfs.mykpts)
    nspins = wfs.nspins

    coulomb = coulomb_inteaction(omega, wfs.gd, kd)
    sym = Symmetry(kd)

    VV_saii, VC_aii, Delta_aiiL = calculate_paw_stuff(dens, setups)

    for k in kpts:
        for spin in range(nspins):
            v_n = xc.calculate_eigenvalue_contribution(
                spin, k, n1, n2, nocc, VV_saii[spin]) * Ha
            yield v_n
