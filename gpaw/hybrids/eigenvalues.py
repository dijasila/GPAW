import json
from pathlib import Path
from typing import List, Union, Tuple, Generator, Dict

import numpy as np
from ase.units import Ha

from gpaw import GPAW
from gpaw.mpi import serial_comm
from gpaw.xc import XC
from gpaw.xc.tools import vxc
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC
from . import parse_name
from .coulomb import coulomb_inteaction
from .kpts import KPoint, RSKPoint, to_real_space
from .paw import calculate_paw_stuff
from .symmetry import Symmetry


def non_self_consistent_eigenvalues(calc: Union[GPAW, str, Path],
                                    xcname: str,
                                    n1: int = 0,
                                    n2: int = 0,
                                    kpt_indices: List[int] = None,
                                    snapshot: Union[str, Path] = None,
                                    ftol: float = 1e-9) -> np.ndarray:
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

    if kpt_indices is None:
        kpt_indices = np.arange(wfs.kd.nibzkpts)

    results = read_snapshot(snapshot)

    xcname, exx_fraction, omega = parse_name(xcname)

    if 'v_dft_sin' not in results:
        xc = XC(xcname)
        results.update(_semi_local(calc, xc, n1, n2, kpt_indices))
        write_snapshot(results, snapshot, wfs.world)

    # Non-local hybrid contribution:
    if 'v_hyb_nl_sin' not in results:
        results['v_hyb_nl_sin'] = [[] * wfs.nspins]

    kpt_indices_s = [kpt_indices[len(v_hyb_nl_in):]
                     for v_hyb_nl_in in results['v_hyb_nl_sin']]

    if any(len(i) > 0 for i in kpt_indices_s):
        for s, v_hyb_nl_n in _non_local(calc, xc, n1, n2, kpt_indices_s,
                                        ftol, exx_fraction, omega):
            results['v_hyb_nl_sin'][s].append(v_hyb_nl_n)
            write_snapshot(results, snapshot, wfs.world)

    return {name: np.array(array) * Ha for name, array in results.items()}


def _semi_local(calc: GPAW,
                xc,
                n1: int,
                n2: int,
                kpt_indices: List[int]
                ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    wfs = calc.wfs
    nspins = wfs.nspins
    e_dft_sin = np.array([[calc.get_eigenvalues(k, spin)[n1:n2]
                           for k in kpt_indices]
                          for spin in range(nspins)])
    v_dft_sin = vxc(calc)[:, kpt_indices, n1:n2]
    v_hyb_sl_sin = vxc(calc, xc)[:, kpt_indices, n1:n2]
    return {'e_dft_sin': e_dft_sin,
            'v_dft_sin': v_dft_sin,
            'v_hyb_sl_sin': v_hyb_sl_sin}


def _non_local(calc: GPAW,
               xc,
               n1: int,
               n2: int,
               kpt_indices_s: List[List[int]],
               ftol: float,
               exx_fraction: float,
               omega: float) -> Generator[np.ndarray, None, None]:
    wfs = calc.wfs
    kd = wfs.kd
    dens = calc.density
    setups = wfs.setups

    nocc = max(((kpt.f_n / kpt.weight) > ftol).sum()
               for kpt in wfs.mykpts)

    coulomb = coulomb_inteaction(omega, wfs.gd, kd)
    sym = Symmetry(kd)

    VV_saii, VC_aii, Delta_aiiL = calculate_paw_stuff(dens, setups)

    for spin, kpt_indices in enumerate(kpt_indices_s):
        if len(kpt_indices) == 0:
            continue
        K = kd.nibzkpts
        k1 = spin * K
        k2 = k1 + K
        kpts2 = [KPoint(kpt.psit.view(0, nocc),
                        kpt.projections.view(0, nocc),
                        kpt.f_n[:nocc] / kpt.weight,  # scale to [0, 1]
                        kd.ibzk_kc[kpt.k],
                        kd.weight_k[kpt.k])
                 for kpt in wfs.mykpts[k1:k2]]
        for i in kpt_indices:
            kpt = wfs.mykpts[k1 + i]
            kpt1 = KPoint(kpt.psit.view(n1, n2),
                          kpt.projections.view(n1, n2),
                          kpt.f_n[n1:n2] / kpt.weight,  # scale to [0, 1]
                          kd.ibzk_kc[kpt.k],
                          kd.weight_k[kpt.k])
            v_n = calculate_eigenvalues(
                kpt1, kpts2, VV_saii[spin], VC_aii, Delta_aiiL, coulomb, sym,
                wfs.world, wfs.setups, wfs.spos_ac)
            yield spin, v_n


def calculate_eigenvalues(kpt1, kpts2, VV_aii, VC_aii, Delta_aiiL,
                          coulomb, sym, comm,
                          setups, spos_ac):
    pd = kpt1.psit.pd
    gd = pd.gd.new_descriptor(comm=serial_comm)
    size = comm.size
    rank = comm.rank

    kd = kpts2[0].psit.pd.kd
    nsym = len(kd.symmetry.op_scc)
    assert len(kpts2) == kd.nibzkpts

    u1_nR = to_real_space(kpt1.psit)
    proj1 = kpt1.proj.broadcast()

    N1 = len(kpt1.psit.array)
    N2 = len(kpts2[0].psit.array)

    B = (N2 + size - 1) // size
    na = min(B * rank, N2)
    nb = min(na + B, N2)

    e_n = np.zeros(N1)
    e_nn = np.empty((N1, nb - na))

    for i2, kpt2 in enumerate(kpts2):
        u2_nR = to_real_space(kpt2.psit, na, nb)
        rskpt0 = RSKPoint(u2_nR,
                          kpt2.proj.broadcast().view(na, nb),
                          kpt2.f_n[na:nb],
                          kpt2.k_c,
                          kpt2.weight)
        for k, i in enumerate(kd.bz2ibz_k):
            if i != i2:
                continue
            s = kd.sym_k[k] + kd.time_reversal_k[k] * nsym
            rskpt2 = sym.apply_symmetry(s, rskpt0, setups, spos_ac)
            q_c = rskpt2.k_c - kpt1.k_c
            qd = KPointDescriptor([-q_c])
            pd12 = PWDescriptor(pd.ecut, gd, pd.dtype, kd=qd)
            ghat = PWLFC([data.ghat_l for data in setups], pd12)
            ghat.set_positions(spos_ac)
            v_G = coulomb.get_potential(pd12)
            Q_annL = [np.einsum('mi, ijL, nj -> mnL',
                                proj1[a],
                                Delta_iiL,
                                rskpt2.proj[a].conj())
                      for a, Delta_iiL in enumerate(Delta_aiiL)]
            rho_nG = ghat.pd.empty(nb - na, u1_nR.dtype)

            for n1, u1_R in enumerate(u1_nR):
                for u2_R, rho_G in zip(rskpt2.u_nR, rho_nG):
                    rho_G[:] = ghat.pd.fft(u1_R * u2_R.conj())

                ghat.add(rho_nG,
                         {a: Q_nnL[n1] for a, Q_nnL in enumerate(Q_annL)})

                for n2, rho_G in enumerate(rho_nG):
                    vrho_G = v_G * rho_G
                    e = ghat.pd.integrate(rho_G, vrho_G).real
                    e_nn[n1, n2] = e / kd.nbzkpts
            e_n -= e_nn.dot(rskpt2.f_n)

    for a, VV_ii in VV_aii.items():
        P_ni = proj1[a]
        vv_n = np.einsum('ni, ij, nj -> n',
                         P_ni.conj(), VV_ii, P_ni).real
        vc_n = np.einsum('ni, ij, nj -> n',
                         P_ni.conj(), VC_aii[a], P_ni).real
        e_n -= (2 * vv_n + vc_n)

    return e_n


def write_snapshot(results: Dict[str, List[List[List[float]]]],
                   snapshot: Path,
                   comm) -> None:
    if comm.rank == 0 and snapshot:
        dct = {name: [[[x for x in x_n]
                       for x_n in x_in]
                      for x_in in x_sin]
               for name, x_sin in results.items()}
        snapshot.write_text(json.dumps(dct))


def read_snapshot(snapshot: Path) -> Dict[str, List[List[List[float]]]]:
    if snapshot:
        snapshot = Path(snapshot)
        if snapshot.is_file():
            return json.loads(snapshot.read_text())
    return {}


