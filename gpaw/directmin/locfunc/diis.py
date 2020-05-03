from ase.parallel import parprint
from gpaw.directmin.tools import matrix_function, get_n_occ
import numpy as np
from ase.units import Hartree
from gpaw.directmin.tools import get_random_um


def one_over_sqrt(x):
    return 1.0 / np.sqrt(x)


def inverse(x):
    return 1.0 / x


class SSLH:

    def __init__(self, obj_f, max_iter=333, g_tol=1.0e-4, min_iter=1):

        self.obj_f = obj_f
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.g_tol = g_tol

    def run(self, wfs, dens, log=None):

        log = parprint
        wfs.timer.start('Inner loop')
        i = 0
        while i < 100:
            self.iterate(wfs, dens, log)
            i += 1

        wfs.timer.stop('Inner loop')

    def iterate(self, wfs, dens, log):

        e_total = 0.0
        error = 0.0
        for kpt in wfs.kpt_u:
            n_occ = get_n_occ(kpt)
            e_sic, H = \
                self.obj_f.get_energy_and_hamiltonain_kpt(wfs, dens, kpt)
            H *= -1.0
            eigval, evec = np.linalg.eigh(H.T.conj() @ H)
            X = matrix_function(eigval, evec, func=one_over_sqrt)
            U = H @ X
            kpt.psit_nG[:n_occ] = np.tensordot(
                U.T, kpt.psit_nG[:n_occ], axes=1)
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

            e_total += e_sic
            error += np.max(np.absolute(H.T.conj() - H))

        print(Hartree * e_total, Hartree * error)


class DIIS_SSLH(SSLH):

    def __init__(self,  obj_f, wfs,
                 max_iter=333, g_tol=1.0e-4, min_iter=1):

        super(DIIS_SSLH, self).__init__(
            obj_f, max_iter=max_iter, g_tol=g_tol, min_iter=min_iter)
        self.R_ki= {}
        self.E_ki = {}
        self.D_ki = {}
        self.n_kps = wfs.kd.nks // wfs.kd.nspins
        self.psi_copy = {}

        for kpt in wfs.kpt_u:
            n_occ = get_n_occ(kpt)
            k = self.n_kps * kpt.s + kpt.q
            self.E_ki[k] = []
            self.R_ki[k] = []
            self.D_ki[k] = []
            self.D_ki[k].append(np.eye(n_occ, dtype=wfs.dtype))

        self.iter = 0
        self.memory = 4
        self.countme = 0

    def run(self, wfs, dens, log=None, max_iter=None, g_tol=None):

        if g_tol is not None:
            self.g_tol = g_tol
            self.max_iter = max_iter
        if max_iter is not None:
            self.max_iter = max_iter
        if log is None:
            log = parprint

        self.iter = 0
        self.psi_copy = {}
        for kpt in wfs.kpt_u:
            n_occ = get_n_occ(kpt)
            k = self.n_kps * kpt.s + kpt.q
            if self.countme == 0 and wfs.dtype is complex:
                U = get_random_um(n_occ, wfs.dtype)
                kpt.psit_nG[:n_occ] = np.tensordot(U,
                    kpt.psit_nG[:n_occ].copy(), axes=1)
            self.psi_copy[k] = kpt.psit_nG[:n_occ].copy()

            self.E_ki[k] = []
            self.R_ki[k] = []
            self.D_ki[k] = []
            self.D_ki[k].append(np.eye(n_occ, dtype=wfs.dtype))

        self.countme += 1
        wfs.timer.start('Inner loop')
        i = 1
        while i < self.max_iter:
            self.iterate(wfs, dens, log)
            log(i, self.e_tot, self.error)
            i += 1
            if self.error < self.g_tol and i > self.min_iter:
                break
        del self.psi_copy
        wfs.timer.stop('Inner loop')

    def iterate(self, wfs, dens, log):

        e_total = 0.0
        error = []

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            n_occ = get_n_occ(kpt)
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
            e_sic, R = \
                self.obj_f.get_energy_and_hamiltonain_kpt(wfs,
                                                          dens,
                                                          kpt)
            R *= -1.0
            self.R_ki[k].append(R)
            E = R - R.T.conj()
            self.E_ki[k].append(E)

            if self.iter == 0:
                eigval, evec = np.linalg.eigh(R.T.conj() @ R)
                X = matrix_function(eigval, evec, func=one_over_sqrt)
                self.D_ki[k].append(R @ X)

            else:
                B = np.einsum(
                    'irs,jrs->ij', np.conj(self.E_ki[k]), self.E_ki[k])
                mI = -np.ones(B.shape[0])
                B = np.vstack([
                    np.hstack([B, mI[:, np.newaxis]]),
                    np.hstack([mI[:, np.newaxis].T,
                               np.array([0.0])[:, np.newaxis]])])
                rhs = np.zeros(shape=B.shape[0])
                rhs[B.shape[0]-1] = -1.0
                coef = np.linalg.solve(B, rhs)
                C = np.einsum('i,irs->rs', coef[:-1], self.D_ki[k])

                kpt.psit_nG[:n_occ] = np.tensordot(
                    C.T, self.psi_copy[k], axes=1)
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

                S = dot(
                    wfs, kpt.psit_nG[:n_occ],
                    kpt.psit_nG[:n_occ], kpt)
                tR = np.einsum('i,irs->rs', coef[:-1], self.R_ki[k])

                # calculate inverse matrix S
                eigval, evec = np.linalg.eigh(S)
                S_inv = matrix_function(eigval, evec, func=inverse)

                eigval, evec = np.linalg.eig(tR.T.conj() @ S_inv @ tR)
                X = matrix_function(eigval, evec, func=one_over_sqrt)

                V = S_inv @ tR @ X
                self.D_ki[k].append(C @ V)

            if len(self.D_ki[k]) > self.memory:
                del self.D_ki[k][0]
                del self.E_ki[k][0]
                del self.R_ki[k][0]

            kpt.psit_nG[:n_occ] = np.tensordot(
                self.D_ki[k][-1].T, self.psi_copy[k], axes=1)
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

            e_total += e_sic
            error.append(np.max(np.absolute(self.E_ki[k][-1])))

        self.error = Hartree * np.max(error)
        self.e_tot =  Hartree * e_total
        self.iter += 1


def dot(wfs, psi_1, psi_2, kpt):

    def dS(a, P_ni):
        return np.dot(P_ni, wfs.setups[a].dO_ii)

    if len(psi_1.shape) == 3:
        ndim = 1
    else:
        ndim = psi_1.shape[0]

    P1_ai = wfs.pt.dict(shape=ndim)
    P2_ai = wfs.pt.dict(shape=ndim)
    wfs.pt.integrate(psi_1, P1_ai, kpt.q)
    wfs.pt.integrate(psi_2, P2_ai, kpt.q)
    dot_prod = wfs.gd.integrate(psi_1, psi_2, False)

    if ndim == 1:
        if wfs.dtype is complex:
            paw_dot_prod = np.array([[0.0 + 0.0j]])
        else:
            paw_dot_prod = np.array([[0.0]])

        for a in P1_ai.keys():
            paw_dot_prod += \
                np.dot(dS(a, P2_ai[a]), P1_ai[a].T.conj())
        if len(psi_1.shape) == 4:
            sum_dot = dot_prod + paw_dot_prod
        else:
            sum_dot = [[dot_prod]] + paw_dot_prod
        # self.wfs.gd.comm.sum(sum_dot)
    else:
        ind_u = np.triu_indices(dot_prod.shape[0], 1)
        dot_prod[(ind_u[1], ind_u[0])] = dot_prod[ind_u].conj()

        paw_dot_prod = np.zeros_like(dot_prod)
        for a in P1_ai.keys():
            paw_dot_prod += \
                np.dot(dS(a, P2_ai[a]), P1_ai[a].T.conj()).T
        sum_dot = dot_prod + paw_dot_prod
    sum_dot = np.ascontiguousarray(sum_dot)
    wfs.gd.comm.sum(sum_dot)

    return sum_dot


