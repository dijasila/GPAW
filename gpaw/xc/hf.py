from collections import defaultdict
from math import pi

import numpy as np
from ase.units import Ha

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC
from gpaw.xc.exx import pawexxvv
from gpaw.utilities import unpack, unpack2


def create_symmetry_map(kd: KPointDescriptor):  # -> List[List[int]]
    sym = kd.symmetry
    U_scc = sym.op_scc
    assert (U_scc[0] == np.eye(3)).all()
    nsym = len(U_scc)
    compconj_s = np.zeros(nsym, bool)
    if sym.time_reversal and not sym.has_inversion:
        U_scc = np.concatenate([U_scc, -U_scc])
        compconj_s = np.zeros(nsym * 2, bool)
        compconj_s[nsym:] = True
        nsym *= 2

    map_ss = np.zeros((nsym, nsym), int)
    for s1 in range(nsym):
        for s2 in range(nsym):
            diff_s = abs(U_scc[s1].dot(U_scc).transpose((1, 0, 2)) -
                         U_scc[s2]).sum(2).sum(1)
            indices = (diff_s == 0).nonzero()[0]
            assert len(indices) == 1
            s = indices[0]
            assert compconj_s[s1] ^ compconj_s[s2] == compconj_s[s]
            map_ss[s1, s2] = s

    return map_ss


def find_kpt_pairs(kd: KPointDescriptor
                   ):  # -> List[Tuple[int, int int, float]]
    map_ss = create_symmetry_map(kd)
    nsym = len(kd.symmetry.op_scc)

    pairs = defaultdict(int)  # Dict[Tuple[int, int, int], int]
    for k1 in range(kd.nbzkpts):
        i1 = kd.bz2ibz_k[k1]
        s1 = kd.sym_k[k1] + kd.time_reversal_k[k1] * nsym
        for k2 in range(kd.nbzkpts):
            i2 = kd.bz2ibz_k[k2]
            s2 = kd.sym_k[k2] + kd.time_reversal_k[k2] * nsym
            s3 = map_ss[s1, s2]
            if i1 < i2:
                pairs[(i1, i2, s3)] += 1
            else:
                s4 = map_ss[s3, 0]
                if i1 == i2:
                    pairs[(i1, i1, min(s3, s4))] += 1
                else:
                    pairs[(i2, i1, s4)] += 1

    N = kd.nbzkpts**2
    assert sum(pairs.values()) == N
    return [(i1, i2, s, n / N) for (i1, i2, s), n in pairs.items()]


class BlockOfStates:
    def __init__(self, u_nR, projections, f_n, bz_index):
        self.u_nR = u_nR
        self.projections = projections
        self.f_n = f_n
        self.bz_index = bz_index

    def __len__(self):
        return len(self.u_nR)

    @staticmethod
    def from_kpt(kpt, pd):
        u_nR = pd.gd.empty(len(kpt.f_n), pd.dtype)
        for psit_G, u_R in zip(kpt.psit.array, u_nR):
            u_R[:] = pd.ifft(psit_G)
        return BlockOfStates(u_nR,
                             kpt.projections,
                             kpt.f_n / kpt.weight,
                             pd.kd.ibz2bz_k[kpt.k])

    def apply_symmetry(self, s, kd, setups, spos_ac):
        u_nR = np.empty_like(self.u_nR)
        projections = self.projections.new()

        bz_index2 = kd.bz2bz_ks[self.bz_index, s]
        assert bz_index2 >= 0

        k_c = kd.bzk_kc[self.bz_index]
        k2_c = kd.bzk_kc[bz_index2]

        time_reversal = kd.time_reversal_k[bz_index2]
        sign = 1 - 2 * time_reversal

        U_cc = kd.symmetry.op_scc[s]

        shift_c = sign * U_cc.dot(k_c) - k2_c
        ishift_c = shift_c.round().astype(int)
        assert np.allclose(ishift_c, shift_c)
        assert not ishift_c.any()

        N_c = u_nR.shape[1:]
        i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
        i = np.ravel_multi_index(i_cr, N_c, 'wrap')
        for u1_R, u2_R in zip(self.u_nR, u_nR):
            u1_R[:] = u2_R.ravel()[i].reshape(N_c)

        for a, id in enumerate(setups.id_a):
            b = kd.symmetry.a_sa[s, a]
            S_c = np.dot(spos_ac[a], U_cc) - spos_ac[b]
            x = np.exp(2j * pi * np.dot(k_c, S_c))
            U_ii = setups[a].R_sii[s].T * x
            projections[a][:] = self.projections[b].dot(U_ii)

        if time_reversal:
            np.conj(u_nR, out=u_nR)
            np.conj(projections.array, out=projections.array)
        return BlockOfStates(u_nR, projections, self.f_n, bz_index2)


def exx(b1: BlockOfStates,
        b2: BlockOfStates,
        Delta_aiiL,
        ghat,
        v_G):  # -> float
    Q_annL = [np.einsum('mi,ijL,nj->mnL',
                        b1.projections[a].conj(),
                        Delta_iiL,
                        b2.projections[a])
              for a, Delta_iiL in enumerate(Delta_aiiL)]

    exx_nn = np.empty((len(b1), len(b2)))
    rho_nG = ghat.pd.empty(len(b2), b1.u_nR.dtype)
    for n1, u1_R in enumerate(b1.u_nR):
        u1cc_R = u1_R.conj()
        n0 = n1 if b1 is b2 else 0
        for n2, rho_G in enumerate(rho_nG[n0:], n0):
            rho_G[:] = ghat.pd.fft(u1cc_R * b2.u_nR[n2])
        ghat.add(rho_nG[n0:],
                 {a: Q_nnL[n1, n0:]
                  for a, Q_nnL in enumerate(Q_annL)})
        for n2, rho_G in enumerate(rho_nG, n0):
            e = ghat.pd.integrate(rho_G, v_G * rho_G).real
            exx_nn[n1, n2] = e
            if b1 is b2:
                exx_nn[n2, n1] = e
    return exx_nn


def extract_exx_things(setups, D_asp):
    D_aiiL = []
    V_asii = []
    C_aii = []
    exxcc = 0.0
    for a, pawdata in enumerate(setups):
        D_aiiL.append(pawdata.Delta_iiL)
        V_sii = []
        for D_p in D_asp[a]:
            D_ii = unpack2(D_p)
            V_ii = pawexxvv(pawdata, D_ii)
            V_sii.append(V_ii)
        V_asii.append(V_sii)
        C_ii = unpack(pawdata.X_p)
        C_aii.append(C_ii)
        exxcc += pawdata.ExxC

    return D_aiiL, V_asii, C_aii, exxcc


def hf(calc, coulomb, spin=0):
    wfs = calc.wfs
    kd = wfs.kd
    pairs = find_kpt_pairs(kd)
    kpts = calc.wfs.mykpts

    D_aiiL, V_asii, C_aii, exxcc = extract_exx_things(wfs.setups,
                                                      calc.density.D_asp)

    exxvv = 0.0
    deps_kn = np.zeros((kd.nibzkpts, wfs.bd.nbands))
    k0 = -1
    for k1, k2, s, weight in sorted(pairs):
        kpt1 = kpts[k1]
        kpt2 = kpts[k2]

        if k1 != k0:
            b1 = BlockOfStates.from_kpt(kpt1, wfs.pd)
            k0 = k1
        if k2 == k1:
            b2 = b1
        else:
            b2 = BlockOfStates.from_kpt(kpt2, wfs.pd)
        if s != 0:
            b2 = b2.apply_symmetry(s, kd, wfs.setups, calc.spos_ac)

        k1_c = kd.ibzk_kc[k1]
        bzk2 = kd.ibz2bz_k[k2]
        bzk2 = kd.bz2bz_ks[bzk2, s]
        k2_c = kd.bzk_kc[bzk2]

        q_c = k1_c - k2_c
        qd = KPointDescriptor([q_c])

        pd = PWDescriptor(wfs.ecut, wfs.gd, wfs.dtype, kd=qd)

        ghat = PWLFC([pawdata.ghat_l for pawdata in wfs.setups], pd)
        ghat.set_positions(calc.spos_ac)

        v_G = coulomb.get_potential(pd)

        e_nn = exx(b1, b2, D_aiiL, ghat, v_G)
        exxvv += 0.5 * weight * b1.f_n.dot(e_nn).dot(b2.f_n)
        deps_kn[k1] += weight * e_nn.dot(b2.f_n)
        deps_kn[k2] += weight * b1.f_n.dot(e_nn)
        print(k1, k2, s, k1_c, k2_c, weight)

    return exxcc + exxvv, deps_kn


def run(c1, c2):
    import threading
    thread = None
    all2all = ...
    while True:
        array1 = c1.next()
        if thread:
            thread.join()
        thread = threading.Thread(target=all2all, args=[array1])
        thread.start()
        array2 = c2.next()
        thread.join()
        thread = threading.Thread(target=all2all, args=[array2])
        thread.start()


if __name__ == '__main__':
    from ase import Atoms
    from gpaw import GPAW, PW
    h = Atoms('H', cell=(3, 3, 7), pbc=(1, 1, 0))
    h.calc = GPAW(mode=PW(100, force_complex_dtype=True),
                  kpts=(1, 1, 1),
                  txt=None)
    h.get_potential_energy()
    # print('Using Wigner-Seitz truncated Coulomb interaction.')
    from gpaw.response.wstc import WignerSeitzTruncatedCoulomb
    wstc = WignerSeitzTruncatedCoulomb(h.calc.wfs.gd.cell_cv,
                                       h.calc.wfs.kd.N_c)
    e, e_kn = hf(h.calc, wstc)
    print(e * Ha, e_kn * Ha)

    from gpaw.xc.exx import EXX
    EXX(h.calc).calculate()

    if 0:
        kd = h.calc.wfs.kd
        print(kd.ibz2bz_k)
        print(kd.bz2ibz_k)
        print(kd.bz2bz_ks)
        print(kd.time_reversal_k)
        print(kd.symmetry.op_scc)
        # create_symmetry_map(kd)
        pairs = find_kpt_pairs(kd)
        print(pairs)
        print(len(pairs), sum(f for _, _, _, f in pairs))
