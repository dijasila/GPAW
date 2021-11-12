import numpy as np
from gpaw.directmin.etdm import random_a, get_n_occ
from ase.units import Hartree
from gpaw.mpi import world
from gpaw.io.logger import GPAWLogger
from copy import deepcopy


class Derivatives:

    def __init__(self, etdm, wfs, c_ref=None, a_vec_u=None,
                 update_c_ref=False, eps=1.0e-7, random_amat=False):
        """
        :param etdm:
        :param wfs:
        :param c_ref: reference orbitals C_ref
        :param a_vec_u: skew-Hermitian matrix A
        :param update_c_ref: if True update reference orbitals
        :param eps: finite difference displacement
        :param random_amat: if True, use random matrix A
        """

        self.eps = eps

        # initialize vectors of elements matrix A
        if a_vec_u is None:
            self.a_vec_u = {u: np.zeros_like(v) for u,
                                                    v in etdm.a_vec_u.items()}

        if random_amat:
            for kpt in wfs.kpt_u:
                u = etdm.kpointval(kpt)
                a = random_a(etdm.a_vec_u[u].shape, wfs.dtype)
                wfs.gd.comm.broadcast(a, 0)
                self.a_vec_u[u] = a

        # initialize orbitals:
        if c_ref is None:
            self.c_ref = etdm.dm_helper.reference_orbitals
        else:
            self.c_ref = c_ref

        # update ref orbitals if needed
        if update_c_ref:
            etdm.rotate_wavefunctions(wfs, self.a_vec_u, etdm.n_dim,
                                      self.c_ref)
            etdm.dm_helper.set_reference_orbitals(wfs, etdm.n_dim)
            self.c_ref = etdm.dm_helper.reference_orbitals
            self.a_vec_u = {u: np.zeros_like(v) for u,
                                                    v in etdm.a_vec_u.items()}

    def get_analytical_derivatives(self, etdm, ham, wfs, dens,
                                   what2calc='gradient'):
        """
           Calculate analytical gradient or approximation to the Hessian
           with respect to the elements of a skew-Hermitian matrix

        :param etdm:
        :param ham:
        :param wfs:
        :param dens:
        :param what2calc: calculate gradient or Hessian
        :return: analytical gradient or Hessian
        """

        assert what2calc in ['gradient', 'hessian']

        if what2calc == 'gradient':
            # calculate analytical gradient
            analytical_der = etdm.get_energy_and_gradients(self.a_vec_u,
                                                           etdm.n_dim,
                                                           ham, wfs, dens,
                                                           self.c_ref)[1]
        else:
            # Calculate analytical approximation to hessian
            analytical_der = np.hstack([etdm.get_hessian(kpt).copy()
                                        for kpt in wfs.kpt_u])
            analytical_der = construct_real_hessian(analytical_der)
            analytical_der = np.diag(analytical_der)

        return analytical_der

    def get_numerical_derivatives(self, etdm, ham, wfs, dens,
                                  what2calc='gradient'):
        """
           Calculate numerical gradient or Hessian with respect to
           the elements of a skew-Hermitian matrix using central finite
           differences

        :param etdm:
        :param ham:
        :param wfs:
        :param dens:
        :param what2calc: calculate gradient or Hessian
        :return: numerical gradient or Hessian
        """

        assert what2calc in ['gradient', 'hessian']

        # total dimensionality if matrices are real
        dim = sum([len(a) for a in self.a_vec_u.values()])
        steps = [1.0, 1.0j] if etdm.dtype == complex else [1.0]
        use_energy_or_gradient = {'gradient': 0, 'hessian': 1}

        matrix_exp = etdm.matrix_exp
        if what2calc == 'gradient':
            numerical_der = {u: np.zeros_like(v) for u,
                                                     v in self.a_vec_u.items()}
        else:
            numerical_der = np.zeros(shape=(len(steps) * dim,
                                            len(steps) * dim))
            # have to use exact gradient when Hessian is calculated
            etdm.matrix_exp = 'egdecomp'

        row = 0
        f = use_energy_or_gradient[what2calc]
        for step in steps:
            for kpt in wfs.kpt_u:
                u = etdm.kpointval(kpt)
                for i in range(len(self.a_vec_u[u])):
                    a = self.a_vec_u[u][i]

                    self.a_vec_u[u][i] = a + step * self.eps
                    fplus = etdm.get_energy_and_gradients(
                        self.a_vec_u, etdm.n_dim, ham, wfs, dens,
                        self.c_ref)[f]

                    self.a_vec_u[u][i] = a - step * self.eps
                    fminus = etdm.get_energy_and_gradients(
                        self.a_vec_u, etdm.n_dim, ham, wfs, dens,
                        self.c_ref)[f]

                    derf = apply_central_finite_difference_approx(
                        fplus, fminus, self.eps)

                    if what2calc == 'gradient':
                        numerical_der[u][i] += step * derf
                    else:
                        numerical_der[row] = construct_real_hessian(derf)

                    row += 1
                    self.a_vec_u[u][i] = a

        if what2calc == 'hessian':
            etdm.matrix_exp = matrix_exp

        return numerical_der


class Davidson(object):
    def __init__(self, etdm, logfile, fd_mode='central', m=np.inf, h=1e-7, eps=1e-2,
                 cap_krylov=False, ef=False, print_level=0,
                 remember_sp_order=False, sp_order=None):
        self.etdm = etdm
        self.fd_mode = fd_mode
        self.remember_sp_order = remember_sp_order
        self.sp_order = sp_order
        self.log_sp_order_once = True
        self.V = None
        self.C = None
        self.M = None
        self.W = None
        self.H = None
        self.lambda_ = None
        self.y = None
        self.x = None
        self.r = None
        self.l = None
        self.h = h
        self.m = m
        self.converged = None
        self.error = None
        self.n_iter = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.reset = None
        self.eps = eps
        self.grad = None
        self.cap_krylov = cap_krylov
        self.ef = ef
        self.dim = {}
        self.dimtot = None
        self.nocc = {}
        self.nbands = None
        self.c_nm_ref = None
        self.print_level = print_level
        self.logfile = logfile
        if self.print_level > 0:
            self.logger = GPAWLogger(world)
            self.logger.fd = logfile
        if self.ef:
            self.lambda_all = None
            self.y_all = None
            self.x_all = None

    def todict(self):
        return {'name': 'Davidson',
                'logfile': self.logfile,
                'fd_mode': self.fd_mode,
                'm': self.m,
                'h': self.h,
                'eps': self.eps,
                'cap_krylov': self.cap_krylov,
                'ef': self.ef,
                'print_level': self.print_level,
                'remember_sp_order': self.remember_sp_order,
                'sp_order': self.sp_order}

    def introduce(self):
        if self.print_level > 0:
            self.logger('|-------------------------------------------------------|')
            self.logger('|             Davidson partial diagonalizer             |')
            self.logger('|-------------------------------------------------------|\n',
                flush=True)

    def run(self, wfs, ham, dens, use_prev=False):
        self.initialize(wfs, use_prev)
        if not self.ef:
            self.etdm.dm_helper.sort_wavefunctions_mom(wfs)
        self.n_iter = 0
        self.c_nm_ref = [deepcopy(wfs.kpt_u[x].C_nM) \
                         for x in range(len(wfs.kpt_u))]
        if self.fd_mode == 'forward' and self.grad is None:
            a_mat_u = {}
            n_dim = {}
            for kpt in wfs.kpt_u:
                u = self.etdm.n_kps * kpt.s + kpt.q
                n_dim[u] = wfs.bd.nbands
                a_mat_u[u] = np.zeros_like(self.etdm.a_mat_u[u])
            self.grad = self.etdm.get_energy_and_gradients(
                a_mat_u, n_dim, ham, wfs, dens, self.c_nm_ref)[1]
        while not self.converged:
            self.iterate(wfs, ham, dens)
        if self.remember_sp_order:
            if self.sp_order is None:
                sp_order = 0
                for i in range(len(self.lambda_)):
                    if self.lambda_[i] < 1e-8:
                        sp_order += 1
                    else:
                        break
                self.sp_order = sp_order
                if self.sp_order == 0:
                    self.sp_order = 1
                self.logger('Saved target saddle point order as ' \
                    + str(self.sp_order) + ' for future partial ' \
                        'diagonalizations.', flush=True)
            elif self.log_sp_order_once:
                self.log_sp_order_once = False
                self.logger('Using target saddle point order of ' \
                    + str(self.sp_order) + '.', flush=True)
        if self.ef:
            self.x_all = []
            for i in range(len(self.lambda_all)):
                self.x_all.append(
                    np.dot(self.V[:, :len(self.lambda_all)], self.y_all[i].T))
            self.x_all = np.asarray(self.x_all).T
        for kpt in wfs.kpt_u:
            k = self.etdm.n_kps * kpt.s + kpt.q
            kpt.C_nM = deepcopy(self.c_nm_ref[k])
        if not self.ef:
            for kpt in wfs.kpt_u:
                self.etdm.dm_helper.sort_wavefunctions(ham, wfs, kpt)

    def initialize(self, wfs, use_prev=False):
        dimz = 2 if self.etdm.dtype == complex else 1
        self.introduce()
        self.reset = False
        self.converged = False
        self.l = 0
        self.V = None
        appr_sp_order = 0
        dia = []
        self.dimtot = 0
        for k, kpt in enumerate(wfs.kpt_u):
            hdia = self.etdm.get_hessian(kpt)
            self.dim[k] = len(hdia)
            self.dimtot += len(hdia)
            dia += list(hdia.copy())
            self.nocc[k] = get_n_occ(kpt)
        self.nbands = wfs.bd.nbands
        if use_prev:
            for i in range(len(self.lambda_all)):
                if self.lambda_all[i] < -1e-8:
                    appr_sp_order += 1
                    if self.etdm.dtype == complex:
                        dia[i] = self.lambda_all[i] + 1.0j * self.lambda_all[i]
                    else:
                        dia[i] = self.lambda_all[i]
        else:
            for i in range(len(dia)):
                if np.real(dia[i]) < -1e-8:
                    appr_sp_order += 1
        self.M = np.zeros(shape=self.dimtot * dimz)
        for i in range(self.dimtot * dimz):
            self.M[i] = np.real(dia[i % self.dimtot])
        if self.sp_order is not None:
            self.l = self.sp_order
        else:
            self.l = appr_sp_order if self.ef else appr_sp_order + 2
        if self.l == 0:
            self.l = 1
        self.W = None
        self.error = [np.inf for x in range(self.l)]
        rng = np.random.default_rng()
        reps = 1e-3
        wfs.timer.start('Initial Krylov space')
        if use_prev:
            self.V = deepcopy(self.x)
            for i in range(self.l):
                for k in range(self.dimtot):
                    for l in range(dimz):
                        rand = np.zeros(shape=2)
                        if world.rank == 0:
                            rand[0] = rng.random()
                            rand[1] = 1 if rng.random() > 0.5 else -1
                        else:
                            rand[0] = 0.0
                            rand[1] = 0.0
                        world.broadcast(rand, 0)
                        self.V[i][l * self.dimtot + k] \
                            += rand[1] * reps * rand[0]
        else:
            self.V = []
            for i in range(self.l):
                rdia = np.real(dia).copy()
                imin = int(np.where(rdia == min(rdia))[0][0])
                rdia[imin] = np.inf
                v = np.zeros(self.dimtot * dimz)
                v[imin] = 1.0
                if self.etdm.dtype == complex:
                    v[self.dimtot + imin] = 1.0
                for l in range(self.dimtot):
                    for m in range(dimz):
                        if l == imin:
                            continue
                        rand = np.zeros(shape=2)
                        if world.rank == 0:
                            rand[0] = rng.random()
                            rand[1] = 1 if rng.random() > 0.5 else -1
                        else:
                            rand[0] = 0.0
                            rand[1] = 0.0
                        world.broadcast(rand, 0)
                        v[m * self.dimtot + l] = rand[1] * reps * rand[0]
                self.V.append(v / np.linalg.norm(v))
            self.V = np.asarray(self.V)
        wfs.timer.start('Modified Gram-Schmidt')
        self.V = mgs(self.V)
        wfs.timer.stop('Modified Gram-Schmidt')
        self.V = self.V.T
        wfs.timer.stop('Initial Krylov space')
        if self.print_level > 0:
            text = 'Davidson will target the ' + str(self.l) \
                   + ' lowest eigenpairs'
            if self.sp_order is None:
                text += '.'
            else:
                text += ' as recovered from previous calculation.'
            self.logger(text, flush=True)

    def iterate(self, wfs, ham, dens):
        wfs.timer.start('FD Hessian vector product')
        if self.W is None:
            self.W = []
            Vt = self.V.T
            for i in range(len(Vt)):
                self.W.append(self.get_fd_hessian(Vt[i], wfs, ham, dens))
            self.reset = False
        else:
            added = len(self.V[0]) - len(self.W[0])
            self.W = self.W.T.tolist()
            Vt = self.V.T
            for i in range(added):
                self.W.append(self.get_fd_hessian(
                    Vt[-added + i], wfs, ham, dens))
        self.W = np.asarray(self.W).T
        wfs.timer.stop('FD Hessian vector product')
        wfs.timer.start('Rayleigh matrix formation')
        #self.H[k] = np.zeros(shape=(self.l[k], self.l[k]))
        #mmm(1.0, self.V[k], 'N', self.W[k], 'T', 0.0, self.H[k])
        self.H = np.dot(self.V.T, self.W)
        wfs.timer.stop('Rayleigh matrix formation')
        self.n_iter += 1
        wfs.timer.start('Rayleigh matrix diagonalization')
        eigv, eigvec = np.linalg.eigh(self.H)
        wfs.timer.stop('Rayleigh matrix diagonalization')
        eigvec = eigvec.T
        if self.ef:
            self.lambda_all = deepcopy(eigv)
            self.y_all = deepcopy(eigvec)
        self.lambda_ = eigv[: self.l]
        self.y = eigvec[: self.l]
        wfs.timer.start('Ritz vector calculation')
        self.x = []
        for i in range(self.l):
            self.x.append(np.dot(self.V, self.y[i].T))
        self.x = np.asarray(self.x)
        wfs.timer.stop('Ritz vector calculation')
        wfs.timer.start('Residual calculation')
        self.r = []
        for i in range(self.l):
            self.r.append(self.x[i] * self.lambda_[i] \
                          - np.dot(self.W, self.y[i].T))
        self.r = np.asarray(self.r)
        wfs.timer.stop('Residual calculation')
        for i in range(self.l):
            self.error[i] = np.abs(self.r[i]).max()
        converged = True
        for i in range(self.l):
            converged = converged and self.error[i] < self.eps
        self.converged = deepcopy(converged)
        if converged:
            self.eigenvalues = deepcopy(self.lambda_)
            self.eigenvectors = deepcopy(self.x)
            self.log(0)
            return
        n_dim = len(self.V)
        wfs.timer.start('Preconditioner calculation')
        self.C = np.zeros(shape=(self.l, n_dim))
        for i in range(self.l):
            self.C[i] = -np.abs(np.repeat(self.lambda_[i], n_dim) - self.M)**-1
            for l in range(len(self.C[i])):
                if self.C[i][l] > -0.1 * Hartree:
                    self.C[i][l] = -0.1 * Hartree
        wfs.timer.stop('Preconditioner calculation')
        wfs.timer.start('Krylov space augmentation')
        wfs.timer.start('New directions')
        t = []
        for i in range(self.l):
            t.append(self.C[i] * self.r[i])
        t = np.asarray(t)
        if len(self.V[0]) <= self.l + self.m:
            self.V = self.V.T.tolist()
            for i in range(self.l):
                self.V.append(t[i])
        elif not self.cap_krylov:
            self.reset = True
            self.V = deepcopy(self.x.tolist())
            for i in range(len(t)):
                self.V.append(t[i])
            self.W = None
        wfs.timer.stop('New directions')
        self.V = np.asarray(self.V)
        if self.cap_krylov:
            if len(self.V) > self.l + self.m:
                if self.print_level > 0:
                    self.logger('Krylov space exceeded maximum size. '
                            'Partial diagonalization is not fully converged.',
                        flush=True)
                self.converged = True
        wfs.timer.start('Modified Gram-Schmidt')
        self.V = mgs(self.V)
        wfs.timer.stop('Modified Gram-Schmidt')
        self.V = self.V.T
        wfs.timer.stop('Krylov space augmentation')
        self.log(self.l)

    def log(self, l):
        if self.print_level > 0:
            self.logger('Dimensionality of Krylov space: ' + str(len(self.V[0]) - l),
                flush=True)
            if self.reset:
                self.logger('Reset Krylov space', flush=True)
            self.logger('\nEigenvalues:\n', flush=True)
            text = ''
            for i in range(self.l):
                text += '%10d '
            indices = text % tuple(range(1, self.l + 1))
            self.logger(indices, flush=True)
            text = ''
            for i in range(self.l):
                text += '%10.6f '
            self.logger(text % tuple(self.lambda_), flush=True)
            self.logger('\nResidual maximum components:\n', flush=True)
            self.logger(indices, flush=True)
            text = ''
            for i in range(self.l):
                text += '%10.6f '
            self.logger(text % tuple(self.error), flush=True)

    def get_fd_hessian(self, vin, wfs, ham, dens):
        v = self.h * vin
        c_nm = deepcopy(self.c_nm_ref)
        a_mat_u = {}
        n_dim = {}
        start = 0
        end = 0
        for k in range(len(wfs.kpt_u)):
            a_mat_u[k] = np.zeros_like(self.etdm.a_mat_u[k])
            n_dim[k] = wfs.bd.nbands
            end += self.dim[k]
            a_mat_u[k] += v[start : end]
            if self.etdm.dtype == complex:
                a_mat_u[k] += 1.0j * v[self.dimtot + start : self.dimtot + end]
            start += self.dim[k]
        gp = self.etdm.get_energy_and_gradients(
            a_mat_u, n_dim, ham, wfs, dens, c_nm)[1]
        for k in range(len(wfs.kpt_u)):
            a_mat_u[k] = np.zeros_like(self.etdm.a_mat_u[k])
        hessi = []
        if self.fd_mode == 'central':
            start = 0
            end = 0
            for k in range(len(wfs.kpt_u)):
                a_mat_u[k] = np.zeros_like(self.etdm.a_mat_u[k])
                end += self.dim[k]
                a_mat_u[k] -= v[start: end]
                if self.etdm.dtype == complex:
                    a_mat_u[k] \
                        -= 1.0j * v[self.dimtot + start : self.dimtot + end]
                start += self.dim[k]
            gm = self.etdm.get_energy_and_gradients(
                a_mat_u, n_dim, ham, wfs, dens, c_nm)[1]
            for k in range(len(wfs.kpt_u)):
                hessi += list((gp[k] - gm[k]) * 0.5 / self.h)
        elif self.fd_mode == 'forward':
            for k in range(len(wfs.kpt_u)):
                hessi += list((gp[k] - self.grad[k]) / self.h)
        if self.etdm.dtype == complex:
            hessc = np.zeros(shape=(2 * self.dimtot))
            hessc[: self.dimtot] = np.real(hessi)
            hessc[self.dimtot :] = np.imag(hessi)
            return hessc
        else:
            return np.asarray(hessi)


def mgs(vin):

    v = deepcopy(vin)
    q = np.zeros_like(v)
    for i in range(len(v)):
        q[i] = v[i] / np.linalg.norm(v[i])
        for k in range(len(v)):
            v[k] = v[k] - np.dot(np.dot(q[i].T, v[k]), q[i])
    return q

def construct_real_hessian(hess):

    if hess.dtype == complex:
        hess_real = np.hstack((np.real(hess), np.imag(hess)))
    else:
        hess_real = hess

    return hess_real

def apply_central_finite_difference_approx(fplus, fminus, eps):

    if isinstance(fplus, dict) and isinstance(fminus, dict):
        assert (len(fplus) == len(fminus))
        derf = np.hstack([(fplus[k] - fminus[k]) * 0.5 / eps
                          for k in fplus.keys()])
    elif isinstance(fplus, float) and isinstance(fminus, float):
        derf = (fplus - fminus) * 0.5 / eps
    else:
        raise ValueError()

    return derf
