from pathlib import Path
from typing import Union

from ase.units import Ha

from gpaw import GPAW
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

    return evc, evv * Ha


def calculate_energy_only(self, kpts, VV_aii):
    pd = kpts[0].psit.pd
    gd = pd.gd.new_descriptor(comm=mpi.serial_comm)
    comm = self.comm
    self.N_c = gd.N_c

    exxvv = 0.0
    for i1, i2, s, k1, k2, count in self.ipairs(kpts1, kpts2):
        q_c = k2.k_c - k1.k_c
        qd = KPointDescriptor([-q_c])

        with self.timer('ghat-init'):
            pd12 = PWDescriptor(pd.ecut, gd, pd.dtype, kd=qd)
            ghat = PWLFC([data.ghat_l for data in self.setups], pd12)
            ghat.set_positions(self.spos_ac)

        v_G = self.coulomb.get_potential(pd12)
        e_nn = self.calculate_exx_for_pair(k1, k2, ghat, v_G,
                                           kpts1[i1].psit.pd,
                                           kpts2[i2].psit.pd,
                                           kpts1[i1].psit.kpt,
                                           kpts2[i2].psit.kpt,
                                           k1.f_n,
                                           k2.f_n,
                                           s,
                                           count,
                                           v1_nG, v1_ani,
                                           v2_nG, v2_ani)

        e_nn *= count
        e = k1.f_n.dot(e_nn).dot(k2.f_n) / self.kd.nbzkpts
        if 0:
            print(i1, i2, s,
                  k1.k_c[2], k2.k_c[2], kpts1 is kpts2, count,
                  e_nn[0, 0], e)
        exxvv -= 0.5 * e
        ekin += e
        if e_kn is not None:
            e_kn[i1] -= e_nn.dot(k2.f_n)

    exxvc = 0.0
    for i, kpt in enumerate(kpts1):
        for a, VV_ii in VV_aii.items():
            P_ni = kpt.proj[a]
            vv_n = np.einsum('ni, ij, nj -> n',
                             P_ni.conj(), VV_ii, P_ni).real
            vc_n = np.einsum('ni, ij, nj -> n',
                             P_ni.conj(), self.VC_aii[a], P_ni).real
            exxvv -= vv_n.dot(kpt.f_n) * kpt.weight
            exxvc -= vc_n.dot(kpt.f_n) * kpt.weight
            if e_kn is not None:
                e_kn[i] -= (2 * vv_n + vc_n)

    self.timer.start('vexx')
    w_knG = {}
    if derivatives:
        G1 = comm.rank * pd.maxmyng
        G2 = (comm.rank + 1) * pd.maxmyng
        for v_nG, v_ani, kpt in zip(v_knG, v_kani, kpts1):
            comm.sum(v_nG)
            w_nG = v_nG[:, G1:G2].copy()
            w_knG[len(w_knG)] = w_nG
            for v_ni in v_ani.values():
                comm.sum(v_ni)
            v1_ani = {}
            for a, VV_ii in VV_aii.items():
                P_ni = kpt.proj[a]
                v_ni = P_ni.dot(self.VC_aii[a] + 2 * VV_ii)
                v1_ani[a] = v_ani[a] - v_ni
                ekin += (np.einsum('n, ni, ni',
                                   kpt.f_n, P_ni.conj(), v_ni).real *
                         kpt.weight)
            self.pt.add(w_nG, v1_ani, kpt.psit.kpt)
    self.timer.stop()

    return comm.sum(exxvv), comm.sum(exxvc), comm.sum(ekin), w_knG
