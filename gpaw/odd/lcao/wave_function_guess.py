from gpaw.odd.pipek_mezey import PipekMezey as PM
from ase.dft.wannier import Wannier as W
from gpaw.odd.lcao.tools import random_skew_herm_matrix
import numpy as np
from scipy.linalg import expm
from gpaw.utilities.lapack import diagonalize

def gram_schmidt(C_nM, S_MM):

    for i in range(C_nM.shape[0]):
        C = np.zeros_like(C_nM[i])
        for j in range(i):
            C -= C_nM[j] * np.dot(C_nM[i].conj(),
                                  np.dot(S_MM, C_nM[j]))

        C_nM[i] += C

        C_nM[i] = C_nM[i] / np.sqrt(np.dot(C_nM[i].conj(),
                                           np.dot(S_MM, C_nM[i])))


def loewdin(C_nM, S_MM):

    """
    Loewdin based orthonormalization
    C_nM = sum_m C_nM[m] [1/sqrt(S)]_mn

    S_mn = (C_nM[m].conj(), S_MM C_nM[n])
    """
    S_overlapp = np.dot(C_nM.conj(), np.dot(S_MM, C_nM.T))

    ev = np.zeros(S_overlapp.shape[0], dtype=float)
    diagonalize(S_overlapp, ev)
    ev_sqrt = np.diag(1.0 / np.sqrt(ev))

    S = np.dot(np.dot(S_overlapp.T.conj(), ev_sqrt), S_overlapp)

    return np.dot(S.T, C_nM)


def get_initial_guess(dtype, calc, n_dim, name, occupied_only, log):

    C_nM_init = {}
    n_kps = calc.wfs.kd.nks // calc.wfs.kd.nspins

    if name == 'KS':
        for kpt in calc.wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            C_nM_init[k] = \
                np.copy(kpt.C_nM[:n_dim[k]])

    elif name is 'KS_PM':
        for kpt in calc.wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            calc.wfs.atomic_correction.calculate_projections(calc.wfs,
                                                             kpt)

            if sum(kpt.f_n) < 1.0e-3:
                C = loewdin(kpt.C_nM,
                            calc.wfs.S_qMM[kpt.q].conj())

                calc.density.gd.comm.broadcast(C, 0)
                C_nM_init[k] = C[:n_dim[k]].copy()
                kpt.C_nM = C.copy()
                continue

            pm = PM(calc, spin=kpt.s,
                    dtype=dtype, ftol=1e-4)
            pm.localize(tolerance=1.0e-4,
                        verbose=False)

            U = pm.W_k[kpt.q].T
            n_d = U.shape[0]
            kpt.C_nM[:n_d] = \
                np.dot(U, kpt.C_nM[:n_d])
            C = loewdin(kpt.C_nM,
                        calc.wfs.S_qMM[kpt.q].conj())

            calc.density.gd.comm.broadcast(C, 0)
            C_nM_init[k] = C[:n_dim[k]].copy()
            kpt.C_nM = C.copy()

    elif name is 'KS_W':
        for kpt in calc.wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            calc.wfs.atomic_correction.calculate_projections(calc.wfs,
                                                             kpt)
            n_occ = 0
            for f in kpt.f_n:
                if f > 1.0e-3:
                    n_occ += 1

            wan = W(n_occ, calc,
                    spin=kpt.s, verbose=False)
            wan.localize(tolerance=1.0e-4)

            U = wan.U_kww[kpt.q].T
            kpt.C_nM[:n_occ] = \
                np.dot(U, kpt.C_nM[:n_occ])

            C = loewdin(kpt.C_nM,
                        calc.wfs.S_qMM[kpt.q].conj())
            calc.density.gd.comm.broadcast(C, 0)
            C_nM_init[k] = C[:n_dim[k]].copy()
            kpt.C_nM = C.copy()

    elif name is 'KS_PM_random':

        for kpt in calc.wfs.kpt_u:
            if sum(kpt.f_n) < 1.0e-3:
                continue
            n_bands = kpt.C_nM.shape[0]
            n_occ = 0
            for f in kpt.f_n:
                if f > 1.0e-3:
                    n_occ += 1

            A_s = random_skew_herm_matrix(n_occ,
                                          -1.0e2, 1.0e2,
                                          dtype)
            U = expm(A_s)
            kpt.C_nM[:n_occ] = \
                np.dot(U.T, kpt.C_nM[:n_occ])
            A_s = random_skew_herm_matrix(n_bands - n_occ,
                                          -1.0e2, 1.0e2,
                                          dtype)
            U = expm(A_s)
            kpt.C_nM[n_occ:n_bands] = \
                np.dot(U.T,
                       kpt.C_nM[n_occ:n_bands])

            C = loewdin(kpt.C_nM, calc.wfs.S_qMM[0].conj())
            calc.density.gd.comm.broadcast(C, 0)
            kpt.C_nM = C.copy()

        for kpt in calc.wfs.kpt_u:

            k = n_kps * kpt.s + kpt.q

            calc.wfs.atomic_correction.calculate_projections(calc.wfs,
                                                             kpt)

            if sum(kpt.f_n) < 1.0e-3:
                continue

            pm = PM(calc, spin=kpt.s,
                    dtype=dtype, ftol=1.0e-4)
            pm.localize(tolerance=1.0e-4, verbose=False)

            U = pm.W_k[kpt.q].T
            n_d = U.shape[0]

            kpt.C_nM[:n_d] = \
                np.dot(U, kpt.C_nM[:n_d])

            C = loewdin(kpt.C_nM,
                        calc.wfs.S_qMM[kpt.q].conj())

            calc.density.gd.comm.broadcast(C, 0)
            C_nM_init[k] = C.copy()
            kpt.C_nM = C.copy()

    elif name is 'KS_W_random':

        for kpt in calc.wfs.kpt_u:
            if sum(kpt.f_n) < 1.0e-3:
                continue
            n_bands = kpt.C_nM.shape[0]
            n_occ = 0
            for f in kpt.f_n:
                if f > 1.0e-3:
                    n_occ += 1

            A_s = random_skew_herm_matrix(n_occ,
                                          -1.0e2, 1.0e2,
                                          dtype)
            U = expm(A_s)
            kpt.C_nM[:n_occ] = \
                np.dot(U.T, kpt.C_nM[:n_occ])
            A_s = random_skew_herm_matrix(n_bands - n_occ,
                                          -1.0e2, 1.0e2,
                                          dtype)
            U = expm(A_s)
            kpt.C_nM[n_occ:n_bands] = \
                np.dot(U.T,
                       kpt.C_nM[n_occ:n_bands])

            C = loewdin(kpt.C_nM, calc.wfs.S_qMM[0].conj())
            calc.density.gd.comm.broadcast(C, 0)
            kpt.C_nM = C.copy()

        for kpt in calc.wfs.kpt_u:

            k = n_kps * kpt.s + kpt.q
            calc.wfs.atomic_correction.calculate_projections(calc.wfs,
                                                             kpt)

            n_occ = 0
            for f in kpt.f_n:
                if f > 1.0e-3:
                    n_occ += 1

            wan = W(n_occ, calc,
                    spin=kpt.s, verbose=False)
            wan.localize(tolerance=1.0e-4)

            U = wan.U_kww[kpt.q].T

            kpt.C_nM[:n_occ] = \
                np.dot(U, kpt.C_nM[:n_occ])

            C = loewdin(kpt.C_nM,
                        calc.wfs.S_qMM[kpt.q].conj())

            calc.density.gd.comm.broadcast(C, 0)
            C_nM_init[k] = C.copy()
            kpt.C_nM = C.copy()

    elif name is 'load_from_file':
        for kpt in calc.wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            C = np.load('C_nM_' + str(k) + '.npy')
            C_nM_init[k] = np.copy(C[:n_dim[k]])
            kpt.C_nM[:n_dim[k]] = np.copy(C[:n_dim[k]])

    elif name is 'load_from_file_L':

        for kpt in calc.wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            C = np.load('C_nM_' + str(k) + '.npy')
            kpt.C_nM[:n_dim[k]] = C[:n_dim[k]].copy()

            # TODO: check S_qMM.conj()?
            kpt.C_nM = loewdin(kpt.C_nM,
                               calc.wfs.S_qMM[kpt.q].conj())

            C_nM_init[k] = kpt.C_nM[:n_dim[k]].copy()

        log('Load from a file and use Loewdin orth.')

    elif name is 'load_from_file_PM':
        for kpt in calc.wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            C = np.load('C_nM_' + str(k) + '.npy')
            kpt.C_nM[:n_dim[k]] = C[:n_dim[k]].copy()

            # TODO: check S_qMM.conj()?
            kpt.C_nM = loewdin(kpt.C_nM,
                               calc.wfs.S_qMM[kpt.q].conj())

            calc.wfs.atomic_correction.calculate_projections(calc.wfs,
                                                             kpt)

            if sum(kpt.f_n) < 1.0e-3:
                continue

            pm = PM(calc, spin=kpt.s,
                    dtype=dtype, ftol=1e-4)
            pm.localize(tolerance=1.0e-4,
                        verbose=False)

            U = pm.W_k[kpt.q].T
            n_d = U.shape[0]
            kpt.C_nM[:n_d] = \
                np.dot(U, kpt.C_nM[:n_d])
            C = loewdin(kpt.C_nM,
                        calc.wfs.S_qMM[kpt.q].conj())

            calc.density.gd.comm.broadcast(C, 0)
            C_nM_init[k] = C[:n_dim[k]].copy()
            kpt.C_nM = C.copy()

    # # TODO: check that this works properly!
    # elif name == 'atomic_basis_set':
    #
    #     if occupied_only is True:
    #         log('You chose atomic_basis_set for occupied states!')
    #         log('They are not spanned on KS manifold')
    #         log('KS-energy is re-calculated on atomic_basis_set')
    #
    #     for s in range(nspins):
    #         C_nM_init[s] = \
    #             np.eye(n_dim[s],
    #                    calc.wfs.basis_functions.Mmax).astype(dtype)
    #
    #         # gram_schmidt(C_nM_init[s], calc.wfs.S_qMM[0])
    #
    #         C_nM_init[s] = loewdin(C_nM_init[s],
    #                                calc.wfs.S_qMM[0])
    #
    #         C_nM[s] = np.copy(C_nM_init[s])
    #
    # # TODO: check that this works properly!
    # elif name is 'A_PM':
    #     for s in range(nspins):
    #         C_nM_init[s] = \
    #             np.eye(C_nM[s].shape[0],
    #                    calc.wfs.basis_functions.Mmax).astype(
    #                 dtype)
    #
    #         # gram_schmidt(C_nM_init[s], calc.wfs.S_qMM[0])
    #         C_nM_init[s] = loewdin(C_nM_init[s],
    #                                calc.wfs.S_qMM[0])
    #     for s in range(nspins):
    #         C_nM[s] = np.copy(C_nM_init[s])
    #
    #     # Make coefficients the same for all
    #     for s in range(nspins):
    #         C1_nM = C_nM[s]
    #         calc.density.gd.comm.broadcast(C1_nM, 0)
    #         C_nM[s] = np.copy(C1_nM)
    #
    #     for s in range(nspins):
    #         calc.wfs.atomic_correction.calculate_projections(
    #             calc.wfs, calc.wfs.kpt_u[s])
    #
    #     for s in range(nspins):
    #         pm = PM(calc, spin=s,
    #                 dtype=dtype, ftol=1e-10)
    #         pm.localize(tolerance=1e-10, verbose=False)
    #
    #         U = pm.W_k[0].T
    #         n_d = U.shape[0]
    #
    #         C_nM_init[s][:n_d] = \
    #             np.dot(U, C_nM[s][:n_d])
    #
    #         # gram_schmidt(C_nM_init[s], calc.wfs.S_qMM[0])
    #         C_nM_init[s] = loewdin(C_nM_init[s],
    #                                calc.wfs.S_qMM[0])
    #     for s in range(nspins):
    #         C_nM_init[s] = C_nM_init[s][:n_dim[s]]
    #
    # # TODO: add projections and check it!
    # elif name is 'A_W':
    #     for s in range(nspins):
    #         C_nM_init[s] = \
    #             np.eye(C_nM[s].shape[0],
    #                    calc.wfs.basis_functions.Mmax).astype(dtype)
    #
    #         # gram_schmidt(C_nM_init[s], calc.wfs.S_qMM[0])
    #         C_nM_init[s] = loewdin(C_nM_init[s],
    #                                calc.wfs.S_qMM[0])
    #         n_occ = 0
    #
    #         C_nM[s] = np.copy(C_nM_init[s])
    #         for f in calc.wfs.kpt_u[s].f_n:
    #             if f > 1.0e-3:
    #                 n_occ += 1
    #
    #         wan = W(n_occ, calc,
    #                 spin=s, verbose=False)
    #         wan.localize(tolerance=1e-14)
    #         U = wan.U_kww[0].T
    #
    #         C_nM_init[s][:n_occ] = \
    #             np.dot(U, C_nM[s][:n_occ])
    #
    #         # gram_schmidt(C_nM_init[s], calc.wfs.S_qMM[0])
    #         C_nM_init[s] = loewdin(C_nM_init[s],
    #                                calc.wfs.S_qMM[0])

    # # TODO: add projections and check it!
    # elif name is 'load_from_file_PM':
    #
    #     for s in range(nspins):
    #         C = np.load('C_nM_' + str(s) + '.npy')
    #
    #         C_nM_init[s] = \
    #             np.copy(C.astype(dtype))
    #
    #         n_occ = 0
    #
    #         for f in calc.wfs.kpt_u[s].f_n:
    #             if f > 1.0e-3:
    #                 n_occ += 1
    #
    #         if n_occ < 1.0e-3:
    #             continue
    #
    #         A_s = random_skew_herm_matrix(n_occ,
    #                                       -1.0e2, 1.0e2,
    #                                       dtype)
    #         U = expm(A_s)
    #         C_nM_init[s][:n_occ] = \
    #             np.dot(U.T, C_nM_init[s][:n_occ])
    #
    #         C_nM_init[s] = \
    #             loewdin(C_nM_init[s],
    #                     calc.wfs.S_qMM[0])
    #
    #     for kpt in calc.wfs.kpt_u:
    #         C1_nM = C_nM_init[n_kps * kpt.s + kpt.q]
    #         calc.density.gd.comm.broadcast(C1_nM, 0)
    #         C_nM_init[n_kps * kpt.s + kpt.q] = np.copy(C1_nM)
    #         kpt.C_nM = C_nM_init[n_kps * kpt.s + kpt.q].copy()
    #
    #     for s in range(nspins):
    #
    #         for z in range(nspins):
    #             calc.wfs.atomic_correction.calculate_projections(
    #                 calc.wfs, calc.wfs.kpt_u[z])
    #
    #         n_occ = 0
    #
    #         for f in calc.wfs.kpt_u[s].f_n:
    #             if f > 1.0e-3:
    #                 n_occ += 1
    #
    #         if n_occ < 1.0e-3:
    #             continue
    #
    #         pm = PM(calc, spin=s,
    #                 dtype=dtype, ftol=1e-10)
    #         pm.localize(tolerance=1e-10, verbose=False)
    #
    #         U = pm.W_k[0].T
    #
    #         C_nM_init[s][:n_occ] = \
    #             np.dot(U, calc.wfs.kpt_u[s].C_nM[:n_occ])
    #
    #         # gram_schmidt(C_nM_init[s], calc.wfs.S_qMM[0])
    #         C_nM_init[s] = loewdin(C_nM_init[s],
    #                                calc.wfs.S_qMM[0])
    #
    #     log('Load from file and use Pipek-Mezey')
    #     log('Also Loewdin')
    #
    # # TODO: add projections and check it!
    # elif name is 'load_from_file_W':
    #     for s in range(nspins):
    #
    #         C = np.load('C_nM_' + str(s) + '.npy')
    #         C_nM_init[s] = np.copy(C[:n_dim[s]])
    #
    #         n_occ = 0
    #         calc.wfs.kpt_u[s].C_nM = \
    #             np.load('C_nM_' + str(s) + '.npy')
    #
    #         for f in calc.wfs.kpt_u[s].f_n:
    #             if f > 1.0e-3:
    #                 n_occ += 1
    #
    #         wan = W(n_occ, calc,
    #                 spin=s, verbose=False)
    #         wan.localize(tolerance=1e-14)
    #         U = wan.U_kww[0].T
    #
    #         C_nM_init[s][:n_occ] = \
    #             np.dot(U, calc.wfs.kpt_u[s].C_nM[:n_occ])
    #
    #         # gram_schmidt(C_nM_init[s], calc.wfs.S_qMM[0])
    #         C_nM_init[s] = loewdin(C_nM_init[s],
    #                                calc.wfs.S_qMM[0])
    #     log('Load from a file and use Wannier functions')
    #     log('Also Loewdin')

    else:
        raise Exception('Check the \'initial orbitals\' parameter!')

    return C_nM_init



