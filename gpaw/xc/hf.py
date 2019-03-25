from collections import defaultdict

import numpy as np

from gpaw.kpt_descriptor import KPointDescriptor


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
    def __init__(self, pd, kpt):
        self.pd = pd
        self.projections = kpt.projections
        self.psit_nR = [pd.ifft(psit_G)
                        for psit_G in kpt.psit.array]


def exx(b1: BlockOfStates,
        b2: BlockOfStates,
        Delta_aiiL,
        ghat,
        pd,
        v_G):  # -> float
    Q_anni = [np.einsum('mi,ijL,nj->mnL',
                        b1.projections[a].conj(),
                        Delta_iiL,
                        b2.projections[a])
              for a, Delta_iiL in enumerate(Delta_aiiL)]

    exx_nn = np.empty((len(b1), len(b2)))
    for n1, u1_R in enumerate(b1.u_nR):
        n0 = n1 if b1 is b2 else 0
        u2_nR = b2.u_nR[n0:]
        rho_nR = u1_R.conj() * u2_nR
        ghat.add({a: Q_nni[n1, n0:]
                  for a, Q_nni in enumerate(Q_anni)}, rho_nR)
        for n2, rho_R in enumerate(rho_nR, n0):
            rho_G = pd.fft(rho_R)
            e = pd.integrate(rho_G, v_G * rho_G)
            exx_nn[n1, n2] = e
            if b1 is b2:
                exx_nn[n2, n1] = e
    return exx_nn


def run(calc):
    print('Using Wigner-Seitz truncated Coulomb interaction.')
    wstc = WignerSeitzTruncatedCoulomb(calc.wfs.gd.cell_cv,
                                       calc.wfs.kd.N_c)


def hf(calc, coulomb):
    wfs = calc.wfs
    kd = wfs.kd
    pairs = find_kpt_pairs(kd)
    kpts = calc.wfs.mykpts
    i0 = -1
    for i1, i2, s, weight in sorted(pairs):
        kpt1 = kpts[i1]
        kpt2 = kpts[i2]
        if i1 != i0:
            b1 = BlockOfStates(wfs.pd, kpt1)
            i0 = i1
        if i2 == i1:
            b2 = b1
        else:
            b2 = BlockOfStates(wfs.pd, kpt2)
        if s != 0:
            b2 = b2.apply_symmetry(s)

        k1_c = kd.ibzk_kc[i1]
        k2 = kd.ibz2_bz_k[i2]
        k2 = kd.bz2bz_ks[k2, s]
        k2_c = kd.bzk_kc[k2]
        q_c = k1_c - k2_c
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(wfs.ecut, wfs.gd, wfs.dtype, kd=qd)
        ghat = PWLFC([pawdata.ghat_l for pawdata in wfs.setups], pd)
        exx(b1, b2)


def all2all():
    pass


def run(c1, c2):
    thread = None
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
    h.calc = GPAW(mode=PW(100),
                  kpts=(4, 4, 1),
                  txt=None)
    h.get_potential_energy()
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
    