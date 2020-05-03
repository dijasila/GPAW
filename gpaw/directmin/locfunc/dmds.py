from ase.parallel import parprint
from gpaw.directmin.tools import get_n_occ, matrix_function
import numpy as np
from ase.units import Hartree
from gpaw.directmin.locfunc.diis import dot
from gpaw.directmin.fd.inner_loop import InnerLoop


def one_over_sqrt(x):
    return 1.0 / np.sqrt(x)


def inverse(x):
    return 1.0 / x


class DMDIIS:

    def __init__(self,  obj_f, wfs,
                 max_iter=333, g_tol=1.0e-4, min_iter=1):

        self.obj_f = obj_f
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.g_tol = g_tol
        self.n_kps = wfs.kd.nks // wfs.kd.nspins
        self.countme = 0
        self.R_ki= {}
        self.E_ki = {}
        self.D_ki = {}
        self.memory = 4
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            n_occ = get_n_occ(kpt)
            self.E_ki[k] = []
            self.R_ki[k] = []
            self.D_ki[k] = []
            self.D_ki[k].append(np.eye(n_occ, dtype=wfs.dtype))

        self.iloop = InnerLoop(
            self.obj_f, wfs)

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

        self.error = self.update_E_and_R(
            wfs, dens, self.E_ki, self.R_ki)
        self.inner_loop = False

        if self.error > 1.0e-2 or self.countme == 0:

            psi_copy = {}
            n_kps = wfs.kd.nks // wfs.kd.nspins
            for kpt in wfs.kpt_u:
                k = n_kps * kpt.s + kpt.q
                psi_copy[k] = kpt.psit_nG[:].copy()

            g_tol = 1.0e-2
            maxiter = 7
            if self.countme == 0:
                g_tol = 1.0e-4
                maxiter = 15

            self.iloop.g_tol = g_tol / Hartree
            self.iloop.maxiter = maxiter
            self.iloop.run(0.0, psi_copy, wfs, dens, None, 0)

            self.inner_loop = True
            n_kps = wfs.kd.nks // wfs.kd.nspins
            for kpt in wfs.kpt_u:
                k = n_kps * kpt.s + kpt.q
                kpt.psit_nG[:] = psi_copy[k]

            del psi_copy

        self.iter = 0
        self.psi_copy = {}
        for kpt in wfs.kpt_u:
            n_occ = get_n_occ(kpt)
            k = self.n_kps * kpt.s + kpt.q
            self.psi_copy[k] = kpt.psit_nG[:n_occ].copy()
            self.E_ki[k] = []
            self.R_ki[k] = []
            self.D_ki[k] = []
            if self.inner_loop:
                self.D_ki[k].append(self.iloop.U_k[k])
                kpt.psit_nG[:n_occ] = np.tensordot(
                    self.D_ki[k][-1].T, self.psi_copy[k], axes=1)
            else:
                self.D_ki[k].append(np.eye(n_occ, dtype=wfs.dtype))

        self.countme += 1
        wfs.timer.start('Inner loop')
        i = 1
        while i < self.max_iter:
            self.iterate(wfs, dens, log)
            # log(i, self.e_tot, self.error)
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
        self.e_tot = Hartree * e_total
        self.iter += 1

    def update_E_and_R(self, wfs, dens, E_ki, R_ki):

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
            R_ki[k].append(R)
            E = R - R.T.conj()
            E_ki[k].append(E)
            error.append(np.max(np.absolute(E_ki[k][-1])))

        return Hartree * np.max(error)
