import numpy as np


def get_occupations(wfs):
    f_sn = {}
    for kpt in wfs.kpt_u:
        n_kps = wfs.kd.nks // wfs.kd.nspins
        u = n_kps * kpt.s + kpt.q
        f_sn[u] = kpt.f_n.copy()
    if wfs.nspins == 2 and wfs.kd.comm.size > 1:
        if wfs.kd.comm.rank == 0:
            # occupation numbers
            size = np.array([0])
            wfs.kd.comm.receive(size, 1)
            f_2n = np.zeros(shape=(int(size[0])))
            wfs.kd.comm.receive(f_2n, 1)
            f_sn[1] = f_2n
            size = np.array([f_sn[0].shape[0]])
            wfs.kd.comm.send(size, 1)
            wfs.kd.comm.send(f_sn[0], 1)
        else:
            # occupations
            size = np.array([f_sn[1].shape[0]])
            wfs.kd.comm.send(size, 0)
            wfs.kd.comm.send(f_sn[1], 0)
            size = np.array([0])
            wfs.kd.comm.receive(size, 0)
            f_2n = np.zeros(shape=(int(size[0])))
            wfs.kd.comm.receive(f_2n, 0)
            f_sn[0] = f_2n

    return f_sn


def excite_and_sort(wfs, i, a, exctype='singlet', mode='fdpw'):

    if exctype == 'singlet':
        for kpt in wfs.kpt_u:
            n_kps = wfs.kd.nks // wfs.kd.nspins
            u = n_kps * kpt.s + kpt.q
            if kpt.s == 0:
                occ_ex_up = kpt.f_n.copy()
                lumo = len(occ_ex_up[occ_ex_up > 0])
                homo = lumo - 1
                indx = [homo + i, lumo + a]
                swindx = [lumo + a, homo + i]
                if mode == 'fdpw':
                    kpt.psit_nG[indx] = kpt.psit_nG[swindx]
                elif mode == 'lcao':
                    wfs.eigensolver.c_nm_ref[u][indx] = \
                        wfs.eigensolver.c_nm_ref[u][swindx]
                else:
                    raise KeyError
                kpt.eps_n[indx] = kpt.eps_n[swindx]
    elif exctype == 'triplet':
        f_sn = get_occupations(wfs)
        lumo = len(f_sn[0][f_sn[0] > 0])
        homo = lumo - 1
        f_sn[0][homo + i] -= 1
        f_sn[1][lumo + a] += 1
        # sort wfs
        for kpt in wfs.kpt_u:
            n_kps = wfs.kd.nks // wfs.kd.nspins
            u = n_kps * kpt.s + kpt.q
            kpt.f_n = f_sn[u].copy()
            occupied = kpt.f_n > 1.0e-10
            n_occ = len(kpt.f_n[occupied])
            if n_occ == 0.0:
                continue
            if np.min(kpt.f_n[:n_occ]) == 0:
                ind_occ = np.argwhere(occupied)
                ind_unocc = np.argwhere(~occupied)
                ind = np.vstack((ind_occ, ind_unocc))
                # Sort coefficients, occupation numbers, eigenvalues
                if mode == 'fdpw':
                    kpt.psit_nG[:] = np.squeeze(kpt.psit_nG[ind])
                elif mode == 'lcao':
                    wfs.eigensolver.c_nm_ref[u] = np.squeeze(
                        wfs.eigensolver.c_nm_ref[u][ind])
                else:
                    raise KeyError
                kpt.f_n = np.squeeze(kpt.f_n[ind])
                kpt.eps_n = np.squeeze(kpt.eps_n[ind])
    else:
        raise KeyError