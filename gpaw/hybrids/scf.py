import numpy as np

from gpaw.mpi import serial_comm
from .kpts import KPoint


def apply1(kpt, Htpsit_xG, wfs, spos_ac,
           coulomb, sym, VC_aii, VV_saii, Delta_aiiL):
    kd = wfs.kd
    K = kd.nibzkpts
    k1 = kpt.s * K
    k2 = k1 + K
    kpts = [KPoint(kpt.psit,
                   kpt.projections,
                   kpt.f_n / kpt.weight,  # scale to [0, 1] range
                   kd.ibzk_kc[kpt.k],
                   kd.weight_k[kpt.k])
            for kpt in wfs.mykpts[k1:k2]]
    evv, evc, ekin, v_knG = calculate(kpts, wfs,
                                      VC_aii, VV_saii[kpt.s],
                                      Delta_aiiL, sym,
                                      spos_ac, coulomb)


def calculate(self, kpts, wfs,
              VC_aii, VV_aii,
              Delta_aiiL, sym,
              spos_ac, coulomb):
    pd = kpts[0].psit.pd
    gd = pd.gd.new_descriptor(comm=serial_comm)
    comm = wfs.world
    nbands = len(kpts[0].psit.array)
    shapes = [(nbands, len(Delta_iiL))
              for Delta_iiL in self.Delta_aiiL]
    v_kani = [{a: np.zeros(shape, pd.dtype)
               for a, shape in enumerate(shapes)}
              for _ in range(len(kpts))]
    v_knG = [k.psit.pd.zeros(nbands, global_array=True, q=k.psit.kpt)
             for k in kpts]

    exxvv = 0.0
    ekin = 0.0
    for i1, i2, s, k1, k2, count in sym.pairs(kpts, comm, setups, spos_ac):
        q_c = k2.k_c - k1.k_c
        qd = KPointDescriptor([-q_c])

        with self.timer('ghat-init'):
            pd12 = PWDescriptor(pd.ecut, gd, pd.dtype, kd=qd)
            ghat = PWLFC([data.ghat_l for data in self.setups], pd12)
            ghat.set_positions(self.spos_ac)

        v1_nG = None
        v1_ani = None
        v2_nG = None
        v2_ani = None
        if derivatives:
            v1_nG = v_knG[i1]
            v1_ani = v_kani[i1]
            if kpts1 is kpts2:
                v2_nG = v_knG[i2]
                v2_ani = v_kani[i2]

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


def calculate_exx_for_pair(self,
                           k1,
                           k2,
                           ghat,
                           v_G,
                           pd1, pd2,
                           index1, index2,
                           f1_n, f2_n,
                           s,
                           count,
                           v1_nG=None,
                           v1_ani=None,
                           v2_nG=None,
                           v2_ani=None,
                           F_av=None):

    factor = 1.0 / self.kd.nbzkpts

    N1 = len(k1.u_nR)
    N2 = len(k2.u_nR)

    size = self.comm.size
    rank = self.comm.rank

    with self.timer('einsum'):
        Q_annL = [np.einsum('mi, ijL, nj -> mnL',
                            k1.proj[a],
                            Delta_iiL,
                            k2.proj[a].conj())
                  for a, Delta_iiL in enumerate(self.Delta_aiiL)]

    if v2_nG is not None:
        T, T_a, cc = self.symmetry_operation(self.inverse_s[s])

    if k1 is k2:
        n2max = (N1 + size - 1) // size
    else:
        n2max = N2

    e_nn = np.zeros((N1, N2))
    rho_nG = ghat.pd.empty(n2max, k1.u_nR.dtype)
    vrho_nG = ghat.pd.empty(n2max, k1.u_nR.dtype)

    for n1, u1_R in enumerate(k1.u_nR):
        if k1 is k2:
            B = (N1 - n1 + size - 1) // size
            n20 = 0
            n2a = min(n1 + rank * B, N2)
            n2b = min(n2a + B, N2)
        else:
            B = (N1 + size - 1) // size
            n20 = min(B * rank, N1)
            n2a = 0
            n2b = N2

        for n2, rho_G in enumerate(rho_nG[:n2b - n2a], n2a):
            rho_G[:] = ghat.pd.fft(u1_R * k2.u_nR[n2].conj())

        with self.timer('exx-add'):
            ghat.add(rho_nG[:n2b - n2a],
                     {a: Q_nnL[n1, n2a:n2b]
                      for a, Q_nnL in enumerate(Q_annL)})

        for n2, rho_G in enumerate(rho_nG[:n2b - n2a], n2a):
            vrho_G = v_G * rho_G
            if F_av:
                for a, v_xL in ghat.derivative(vrho_G).items():
                    print(a, v_xL.shape)
                1 / 0
            e = ghat.pd.integrate(rho_G, vrho_G).real
            e_nn[n1, n2] = e
            if k1 is k2:
                e_nn[n2, n1] = e
            vrho_nG[n2 - n2a] = vrho_G

            if v1_nG is not None:
                vrho_R = ghat.pd.ifft(vrho_G)
                if v2_nG is None:
                    assert k1 is not k2
                    v1_nG[n1] -= f2_n[n2] * factor * pd1.fft(
                        vrho_R * k2.u_nR[n2], index1, local=True)
                else:
                    x = factor * count / 2
                    if k1 is k2 and n1 != n2:
                        x *= 2
                    x1 = x / (self.kd.weight_k[index1] * self.kd.nbzkpts)
                    x2 = x / (self.kd.weight_k[index2] * self.kd.nbzkpts)
                    v1_nG[n1] -= f2_n[n2] * x1 * pd1.fft(
                        vrho_R * k2.u_nR[n2], index1, local=True)
                    v2_nG[n2 + n20] -= f1_n[n1] * x2 * pd2.fft(
                        T(vrho_R.conj() * u1_R), index2,
                        local=True)

        if v1_nG is not None and v2_nG is None:
            self.timer.start('ghat.int')
            for a, v_nL in ghat.integrate(vrho_nG[:n2b - n2a]).items():
                v_iin = self.Delta_aiiL[a].dot(v_nL.T)
                v1_ani[a][n1] -= np.einsum('ijn, nj, n -> i',
                                           v_iin,
                                           k2.proj[a][n2a:n2b],
                                           f2_n[n2a:n2b] * factor)
            self.timer.stop()

        if v1_nG is not None and v2_nG is not None:
            x = factor * count / self.kd.nbzkpts / 2
            x1 = x / self.kd.weight_k[index1]
            x2 = x / self.kd.weight_k[index2]
            if k1 is k2:
                x1 *= 2
                x2 *= 2

            self.timer.start('ghat.int2')
            for a, v_nL in ghat.integrate(vrho_nG[:n2b - n2a]).items():
                if k1 is k2 and n2a <= n1 < n2b:
                    v_nL[n1 - n2a] *= 0.5
                v_iin = self.Delta_aiiL[a].dot(v_nL.T)
                v1_ani[a][n1] -= np.einsum('ijn, nj, n -> i',
                                           v_iin,
                                           k2.proj[a][n2a:n2b],
                                           f2_n[n2a:n2b] * x1)
                b, S_c, U_ii = T_a[a]
                v_ni = np.einsum('ijn, j, ik -> nk',
                                 v_iin.conj(),
                                 k1.proj[b][n1],
                                 U_ii)
                if v_nL.dtype == complex:
                    v_ni *= np.exp(2j * pi * k2.k_c.dot(S_c))
                    if cc:
                        v_ni = v_ni.conj()
                v2_ani[a][n20 + n2a:n20 + n2b] -= v_ni * f1_n[n1] * x2
            self.timer.stop()

    return e_nn * factor


def apply2(kpt, psit_xG, Htpsit_xG, wfs):
    K = kd.nibzkpts
    k1 = (spin - kd.comm.rank) * K
    k2 = k1 + K
    kpts2 = [KPoint(kpt.psit,
                    kpt.projections,
                    kpt.f_n / kpt.weight,  # scale to [0, 1] range
                    kd.ibzk_kc[kpt.k],
                    kd.weight_k[kpt.k])
             for kpt in self.wfs.mykpts[k1:k2]]

    psit = kpt.psit.new(buf=psit_xG)
    P = kpt.projections.new()
    psit.matrix_elements(self.wfs.pt, out=P)

    kpts1 = [KPoint(psit,
                    P,
                    kpt.f_n + nan,
                    kd.ibzk_kc[kpt.k],
                    nan)]
    v_1xG = self.xx.calculate(
        kpts1, kpts2,
        VV_aii)
