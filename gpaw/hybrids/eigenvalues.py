import json
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
from ase.units import Ha

from gpaw import GPAW
from gpaw.xc.tools import vxc
from .hybrid import HybridXC


def non_self_consistent_eigenvalues(calc: Union[GPAW, str, Path],
                                    xcname: str,
                                    n1=0,
                                    n2=-1,
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

    if n2 == -1:
        n2 = wfs.bd.nbands

    if kpts is None:
        kpts = np.arange(wfs.kd.nibzkpts)

    nocc = max(((kpt.f_n / kpt.weight) > ftol).sum()
               for kpt in wfs.mykpts)
    nspins = wfs.nspins
    nkpts = len(kpts)

    dct = {}
    if restart:
        path = Path(restart)
        if path.is_file():
            dct = json.loads(path.read_text())

    xc = HybridXC(xcname)

    if 'v_dft_sin' in dct:
        v_dft_sin = np.array(dct['v_dft_sin'])
        e_dft_sin = np.array(dct['e_dft_sin'])
        v_hyb_sl_sin = np.array(dct['v_hyb_sl_sin'])  # semi-local part
    else:
        v_dft_sin = vxc(calc)[:, kpts, n1:n2]
        e_dft_sin = np.array([[calc.get_eigenvalues(k, spin)[n1:n2]
                               for k in kpts]
                              for spin in range(nspins)])
        if xc.xc:
            v_hyb_sl_sin = vxc(calc, xc.xc)[:, kpts, n1:n2]
        else:
            v_hyb_sl_sin = np.zeros_like(v_dft_sin)
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
        xc.initialize_from_calculator(calc)
        xc.set_positions(calc.spos_ac)
        xc._initialize()
        VV_saii = [xc.calculate_valence_valence_paw_corrections(spin)
                   for spin in range(nspins)]

        for i, k in enumerate(kpts[i0:], i0):
            for spin in range(nspins):
                v_n = xc.calculate_eigenvalue_contribution(
                    spin, k, n1, n2, nocc, VV_saii[spin]) * Ha
                v_hyb_nl_sin[spin, i] = v_n
            if wfs.world.rank == 0 and restart:
                dct = {'v_dft_sin': v_dft_sin.tolist(),
                       'e_dft_sin': e_dft_sin.tolist(),
                       'v_hyb_sl_sin': v_hyb_sl_sin.tolist(),
                       'v_hyb_nl_sin': v_hyb_nl_sin[:, :i + 1].tolist()}
                path.write_text(json.dumps(dct))

    return e_dft_sin, v_dft_sin, v_hyb_sl_sin + v_hyb_nl_sin


def _semi_local_eigenvalues(calc: GPAW,
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
    if xc.xc:
        v_hyb_sl_sin = vxc(calc, xc.xc)[:, kpts, n1:n2]
    else:
        v_hyb_sl_sin = np.zeros_like(v_dft_sin)
    return e_dft_sin, v_dft_sin, v_hyb_sl_sin


def _non_local_eigenvalues(calc: GPAW,
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

    xc.initialize_from_calculator(calc)
    xc.set_positions(calc.spos_ac)
    xc._initialize()
    VV_saii = [xc.calculate_valence_valence_paw_corrections(spin)
               for spin in range(nspins)]

    for k in kpts:
        for spin in range(nspins):
            v_n = xc.calculate_eigenvalue_contribution(
                spin, k, n1, n2, nocc, VV_saii[spin]) * Ha
            yield v_n
