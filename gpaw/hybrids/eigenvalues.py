import json
from pathlib import Path
from typing import List, Union, Tuple, Generator

import numpy as np
from ase.units import Ha

from gpaw import GPAW
from gpaw.xc import XC
from gpaw.xc.tools import vxc
from . import parse_name
from .coulomb import coulomb_inteaction
from .kpts import KPoint
from .paw import calculate_paw_stuff
from .symmetry import Symmetry


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
        for v_hyb_nl_sn in _non_local(calc, xc, n1, n2, kpts[i0:],
                                      ftol, exx_fraction, omega):
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
               ftol: float,
               exx_fraction: float,
               omega: float) -> Generator[np.ndarray, None, None]:
    wfs = calc.wfs
    kd = wfs.kd
    dens = calc.density
    setups = wfs.setups

    nocc = max(((kpt.f_n / kpt.weight) > ftol).sum()
               for kpt in wfs.mykpts)
    nspins = wfs.nspins

    coulomb = coulomb_inteaction(omega, wfs.gd, kd)
    sym = Symmetry(kd)

    VV_saii, VC_aii, Delta_aiiL = calculate_paw_stuff(dens, setups)

    kpts = [KPoint(kpt.psit.view(0, nocc),
                   kpt.projections.view(0, nocc),
                   kpt.f_n[:nocc] / kpt.weight,  # scale to [0, 1]
                   kd.ibzk_kc[kpt.k],
                   kd.weight_k[kpt.k])
            for kpt in wfs.mykpts[k1:k2]]
    for k in kpts:
        for spin in range(nspins):
            v_n = xc.calculate_eigenvalue_contribution(
                spin, k, n1, n2, nocc, VV_saii[spin]) * Ha
            yield v_n


def calculate_eigenvalues(self, kpt1, kpts2, VV_aii):
    pd = kpt1.psit.pd
    gd = pd.gd.new_descriptor(comm=serial_comm)
    comm = self.comm
    size = comm.size
    rank = comm.rank
    self.N_c = gd.N_c
    kd = self.kd
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

    for k2, kpt2 in enumerate(kpts2):
        u2_nR = to_real_space(kpt2.psit, na, nb)
        rskpt0 = RSKPoint(u2_nR,
                          kpt2.proj.broadcast().view(na, nb),
                          kpt2.f_n[na:nb],
                          kpt2.k_c,
                          kpt2.weight)
        for K, k in enumerate(kd.bz2ibz_k):
            if k != k2:
                continue
            s = kd.sym_k[K] + kd.time_reversal_k[K] * nsym
            rskpt2 = self.apply_symmetry(s, rskpt0)
            q_c = rskpt2.k_c - kpt1.k_c
            qd = KPointDescriptor([-q_c])
            pd12 = PWDescriptor(pd.ecut, gd, pd.dtype, kd=qd)
            ghat = PWLFC([data.ghat_l for data in self.setups], pd12)
            ghat.set_positions(self.spos_ac)
            v_G = self.coulomb.get_potential(pd12)
            with self.timer('einsum'):
                Q_annL = [np.einsum('mi, ijL, nj -> mnL',
                                    proj1[a],
                                    Delta_iiL,
                                    rskpt2.proj[a].conj())
                          for a, Delta_iiL in enumerate(self.Delta_aiiL)]
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
                         P_ni.conj(), self.VC_aii[a], P_ni).real
        e_n -= (2 * vv_n + vc_n)

    return e_n
