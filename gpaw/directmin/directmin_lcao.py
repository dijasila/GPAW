from ase.units import Hartree
import numpy as np
from gpaw.utilities.blas import mmm  # , dotc, dotu
from gpaw.directmin.tools import expm_ed, D_matrix
from gpaw.directmin.sd_lcao import SteepestDescent, FRcg, HZcg, \
    QuickMin, LBFGS, LBFGS_P
from gpaw.directmin.ls_lcao import UnitStepLength, \
    StrongWolfeConditions, Parabola
from gpaw.lcao.eigensolver import DirectLCAO
from scipy.linalg import expm, expm_frechet


class DirectMinLCAO(DirectLCAO):

    def __init__(self, diagonalizer=None, error=np.inf,
                 search_direction_algorithm='LBFGS_P',
                 line_search_algorithm='SwcAwc',
                 initial_orbitals='KS',
                 initial_rotation='zero',
                 memory_lbfgs=3,
                 use_prec=True):

        super(DirectMinLCAO, self).__init__(diagonalizer, error)

        self.sda = search_direction_algorithm
        self.lsa = line_search_algorithm
        self.initial_rotation = initial_rotation
        self.initial_orbitals = initial_orbitals
        self.get_en_and_grad_iters = 0
        self.update_refs_counter = 0
        self.memory_lbfgs = memory_lbfgs
        self.use_prec = use_prec
        self.iters = 0

    def __str__(self):

        return 'Direct Minimisation'

    def initialize_2(self, wfs, dens):
        dens.direct_min = True  # turn off the mixer
        self.dtype = wfs.dtype
        self.n_kps = wfs.kd.nks // wfs.kd.nspins

        self.n_dim = {}  # dimensionality of the problem.
                         # this implementation rotates among all bands
        for kpt in wfs.kpt_u:
            u = kpt.s * self.n_kps + kpt.q
            self.n_dim[u] = wfs.bd.nbands

        # choose search direction and line search algorithm
        self.initialize_sd_and_ls(wfs, self.sda, self.lsa)

        self.a_mat_u = {}  # skew-hermitian matrix to be exponented
        self.g_mat_u = {}  # gradient matrix
        self.c_nm_ref = {}  # reference orbitals to be rotated

        self.evecs = {}   # eigendecomposition fo a
        self.evals = {}

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            self.a_mat_u[k] = np.zeros(shape=(self.n_dim[k],
                                              self.n_dim[k]),
                                       dtype=self.dtype)
            self.g_mat_u[k] = np.zeros(shape=(self.n_dim[k],
                                              self.n_dim[k]),
                                       dtype=self.dtype)
            # use initial KS orbitals, but can be others
            self.c_nm_ref[k] = np.copy(kpt.C_nM[:self.n_dim[k]])
            self.evecs[k] = None
            self.evals[k] = None

        self.alpha = 1.0  # step length
        self.phi = [None, None]  # energy at alpha and alpha old
        self.der_phi = [None, None] # gradients at alpha and alpha old
        self.precond = None

        self.iters = 1

        self.nvalence = wfs.nvalence
        self.kd_comm = wfs.kd.comm

    def initialize_sd_and_ls(self, wfs, method, ls_method):

        if method == 'SD':
            self.search_direction = SteepestDescent(wfs)
        elif method == 'FRcg':
            self.search_direction = FRcg(wfs)
        elif method == 'HZcg':
            self.search_direction = HZcg(wfs)
        elif method == 'QuickMin':
            self.search_direction = QuickMin(wfs)
        elif method == 'LBFGS':
            self.search_direction = LBFGS(wfs, self.memory_lbfgs)
        elif method == 'LBFGS_P':
            self.search_direction = LBFGS_P(wfs, self.memory_lbfgs)
        else:
            raise NotImplementedError('Check keyword for '
                                      'search direction!')

        if ls_method == 'UnitStep':
            self.line_search = \
                UnitStepLength(self.evaluate_phi_and_der_phi)
        elif ls_method == 'Parabola':
            self.line_search = Parabola(self.evaluate_phi_and_der_phi)
        elif ls_method == 'SwcAwc':
            self.line_search = \
                StrongWolfeConditions(self.evaluate_phi_and_der_phi,
                                      method=method,
                                      awc=True,
                                      max_iter=5
                                      )
        else:
            raise NotImplementedError('Check keyword for '
                                      'line search!')

    def iterate(self, ham, wfs, dens, occ):

        assert dens.mixer.driver.name == 'dummy', \
            'please, use: mixer=DummyMixer()'
        assert wfs.bd.nbands == wfs.basis_functions.Mmax, \
            'please, use: nbands=\'nao\''

        wfs.timer.start('Direct Minimisation step')

        if self.iters == 0:
            # need to initialize c_nm, eps, f_n and so on.
            # first iteration is diagonilisation using super class
            super().iterate(ham, wfs)
            occ.calculate(wfs)
            self.initialize_2(wfs, dens)
            # wfs.timer.stop('Direct Minimisation step')
            # return

        wfs.timer.start('Preconditioning:')
        self.precond = \
            self.update_preconditioning_and_ref_orbitals(ham, wfs,
                                                         dens, occ,
                                                         self.use_prec)
        wfs.timer.stop('Preconditioning:')

        a = self.a_mat_u
        n_dim = self.n_dim
        alpha = self.alpha
        phi = self.phi
        c_ref = self.c_nm_ref
        der_phi = self.der_phi

        if self.iters == 1:
            phi[0], g = self.get_energy_and_gradients(a, n_dim, ham,
                                                      wfs, dens, occ,
                                                      c_ref)
        else:
            g = self.g_mat_u

        wfs.timer.start('Get Search Direction')
        p = self.get_search_direction(a, g, self.precond, wfs)
        wfs.timer.stop('Get Search Direction')
        der_phi_c = 0.0
        for k in g.keys():
            if self.dtype is complex:
                il1 = np.tril_indices(g[k].shape[0])
            else:
                il1 = np.tril_indices(g[k].shape[0], -1)
            der_phi_c += np.dot(g[k][il1].conj(),
                                p[k][il1]).real
            # der_phi_c += dotc(g[k][il1], p[k][il1]).real
        der_phi_c = wfs.kd.comm.sum(der_phi_c)
        der_phi[0] = der_phi_c

        phi_c, der_phi_c = phi[0], der_phi[0]

        alpha, phi[0], der_phi[0], g = \
            self.line_search.step_length_update(a, p, n_dim,
                                                ham, wfs, dens, occ,
                                                c_ref,
                                                phi_0=phi[0],
                                                der_phi_0=der_phi[0],
                                                phi_old=phi[1],
                                                der_phi_old=der_phi[1],
                                                alpha_max=3.0,
                                                alpha_old=alpha)

        if wfs.gd.comm.size > 1:
            wfs.timer.start('Broadcast gradients')
            alpha_phi_der_phi = np.array([alpha, phi[0], der_phi[0]])
            wfs.gd.comm.broadcast(alpha_phi_der_phi, 0)
            alpha = alpha_phi_der_phi[0]
            phi[0] = alpha_phi_der_phi[1]
            der_phi[0] = alpha_phi_der_phi[2]
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                wfs.gd.comm.broadcast(g[k], 0)
            wfs.timer.stop('Broadcast gradients')

        phi[1], der_phi[1] = phi_c, der_phi_c

        # calculate new matrices for optimal step length
        for k in a.keys():
            a[k] += alpha * p[k]
        self.g_mat_u = g
        # print(self.iters, phi[0]*Hartree, der_phi[0])
        self.iters += 1

        wfs.timer.stop('Direct Minimisation step')

    def get_energy_and_gradients(self, a_mat_u, n_dim, ham, wfs, dens,
                                 occ, c_nm_ref):

        """
        Energy E = E[C exp(A)]. Gradients G_ij[C, A] = dE/dA_ij

        :param a_mat: A
        :param c_nm_ref: C
        :param n_dim:
        :return:
        """

        wfs.timer.start('Unitary rotation')
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            if n_dim[k] == 0:
                continue
            # this method is based on diagonalisation
            # u_nn, self.evecs[k], self.evals[k] =\
            #     expm_ed(a_mat_u[k], evalevec=True)

            # Pade
            u_nn = expm(a_mat_u[k])

            # u_nn = expm_frechet(a_mat_u[k],
            #                     np.zeros_like(a_mat_u[k]),
            #                     compute_expm=True,
            #                     check_finite=False)[0]

            kpt.C_nM[:n_dim[k]] = np.dot(u_nn.T,
                                         c_nm_ref[k][:n_dim[k]])
            #
            # mmm(1.0, np.ascontiguousarray(u_nn), 'T',
            #     np.ascontiguousarray(c_nm_ref[k][:n_dim[k]]), 'N',
            #     0.0,
            #     kpt.C_nM[:n_dim[k]])
            del u_nn
            wfs.atomic_correction.calculate_projections(wfs, kpt)
        wfs.timer.stop('Unitary rotation')

        wfs.timer.start('Update Kohn-Sham energy')
        e_total = self.update_ks_energy(ham, wfs, dens, occ)
        wfs.timer.stop('Update Kohn-Sham energy')

        wfs.timer.start('Calculate gradients')
        g_mat_u = {}
        self._error = 0.0
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            if n_dim[k] == 0:
                g_mat_u[k] = np.zeros_like(a_mat_u[k])
                continue
            h_mm = self.calculate_hamiltonian_matrix(ham, wfs, kpt)
            # make matrix hermitian
            ind_l = np.tril_indices(h_mm.shape[0], -1)
            h_mm[(ind_l[1], ind_l[0])] = h_mm[ind_l].conj()
            g_mat_u[k], error = self.get_gradients(h_mm, kpt.C_nM,
                                                   kpt.f_n,
                                                   a_mat_u[k],
                                                   self.evecs[k],
                                                   self.evals[k],
                                                   kpt, wfs.timer)
            self._error += error
        self._error = self.kd_comm.sum(self._error)
        wfs.timer.stop('Calculate gradients')

        self.get_en_and_grad_iters += 1

        return e_total, g_mat_u

    def update_ks_energy(self, ham, wfs, dens, occ):

        # using new states update KS
        # print('Call energy')
        dens.update(wfs)
        ham.update(dens, wfs, False)
        e_ks = ham.get_energy(occ, False)
        return e_ks

    def get_gradients(self, h_mm, c_nm, f_n, a_mat, evec, evals,
                      kpt, timer):

        timer.start('Construct Gradient Matrix')
        hc_mn = np.zeros(shape=(c_nm.shape[1], c_nm.shape[0]),
                         dtype=self.dtype)
        mmm(1.0, h_mm.conj(), 'N', c_nm, 'T', 0.0, hc_mn)
        k = self.n_kps * kpt.s + kpt.q
        if self.n_dim[k] != c_nm.shape[1]:
            h_mm = np.zeros(shape=(self.n_dim[k], self.n_dim[k]),
                            dtype=self.dtype)
        mmm(1.0, c_nm.conj(), 'N', hc_mn, 'N', 0.0, h_mm)
        timer.stop('Construct Gradient Matrix')

        # let's also calculate residual here.
        # it's extra calculation though, maybe it's better to use
        # norm of grad
        timer.start('Residual')
        n_occ = 0
        for f in kpt.f_n:
            if f > 1.0e-10:
                n_occ += 1
        # what if there are empty states between occupied?
        rhs = np.zeros(shape=(c_nm.shape[1], n_occ),
                       dtype=self.dtype)
        rhs2 = np.zeros(shape=(c_nm.shape[1], n_occ),
                        dtype=self.dtype)
        mmm(1.0, kpt.S_MM.conj(), 'N', c_nm[:n_occ], 'T', 0.0, rhs)
        mmm(1.0, rhs, 'N', h_mm[:n_occ, :n_occ], 'N', 0.0, rhs2)
        hc_mn = hc_mn[:, :n_occ] - rhs2[:, :n_occ]
        norm = []
        for i in range(n_occ):
            norm.append(np.dot(hc_mn[:,i].conj(),
                               hc_mn[:,i]).real * kpt.f_n[i])
            # needs to be cont. to use this
            # x = np.ascontiguousarray(hc_mn[:,i])
            # norm.append(dotc(x, x).real * kpt.f_n[i])

        error = sum(norm) * Hartree ** 2 / self.nvalence
        del rhs, rhs2, hc_mn, norm
        timer.stop('Residual')
        # continue with gradients
        timer.start('Construct Gradient Matrix')
        h_mm = f_n[:, np.newaxis] * h_mm - f_n * h_mm

        # this one uses eigendecomposition of a_mat
        # grad = evec @ h_mm.T.conj() @ evec.T.conj()
        # grad = grad * D_matrix(evals)
        # grad = evec.T.conj() @ grad @ evec
        # for i in range(grad.shape[0]):
        #     grad[i][i] *= 0.5

        # the same using mmm, doesn't seens to be faster though
        # grad = np.empty_like(evec)
        # h_mm = h_mm.astype(complex)
        # mmm(1.0, h_mm, 'N', evec, 'N', 0.0, grad)
        # mmm(1.0, grad, 'C', evec, 'N', 0.0, h_mm)
        # # do we have this operation in blas?
        # grad = h_mm * D_matrix(evals)
        # mmm(1.0, evec, 'N', grad, 'N', 0.0, h_mm)
        # mmm(1.0, h_mm, 'N', evec, 'C', 0.0, grad)
        # grad.ravel()[::grad.shape[1] + 1] *= 0.5

        # frechet derivative, unfortunately it calculates unitary
        # matrix which we already calculated before. Could it be used?

        u, grad = expm_frechet(a_mat, h_mm.T.conj(),
                               compute_expm=True,
                               check_finite=False)

        grad = grad @ u.T.conj()
        grad.ravel()[::grad.shape[1] + 1] *= 0.5

        timer.stop('Construct Gradient Matrix')

        if a_mat.dtype == float:
            return 2.0 * grad.real, error
        else:
            return 2.0 * grad, error

    def get_search_direction(self, a_mat_u, g_mat_u, precond, wfs):

        # structure of vector is
        # (x_1_up, x_2_up,..,y_1_up, y_2_up,..,
        #  x_1_down, x_2_down,..,y_1_down, y_2_down,.. )

        g_vec = {}
        a_vec = {}

        for k in a_mat_u.keys():
            if self.dtype is complex:
                il1 = np.tril_indices(a_mat_u[k].shape[0])
            else:
                il1 = np.tril_indices(a_mat_u[k].shape[0], -1)

            a_vec[k] = a_mat_u[k][il1]
            g_vec[k] = g_mat_u[k][il1]

        p_vec = self.search_direction.update_data(wfs, a_vec, g_vec,
                                                  precond)
        del a_vec, g_vec

        p_mat_u = {}
        for k in p_vec.keys():
            p_mat_u[k] = np.zeros_like(a_mat_u[k])
            if self.dtype is complex:
                il1 = np.tril_indices(a_mat_u[k].shape[0])
            else:
                il1 = np.tril_indices(a_mat_u[k].shape[0], -1)
            p_mat_u[k][il1] = p_vec[k]
            # make it skew-hermitian
            ind_l = np.tril_indices(p_mat_u[k].shape[0], -1)
            p_mat_u[k][(ind_l[1], ind_l[0])] = -p_mat_u[k][ind_l].conj()
        del p_vec

        return p_mat_u

    def evaluate_phi_and_der_phi(self, a_mat_u, p_mat_u, n_dim, alpha,
                                 ham, wfs, dens, occ, c_ref,
                                 phi=None, g_mat_u=None):
        """
        phi = f(x_k + alpha_k*p_k)
        der_phi = \grad f(x_k + alpha_k*p_k) \cdot p_k
        :return:  phi, der_phi # floats
        """
        if phi is None or g_mat_u is None:
            x_mat_u = {k: a_mat_u[k] + alpha * p_mat_u[k]
                       for k in a_mat_u.keys()}
            phi, g_mat_u = \
                self.get_energy_and_gradients(x_mat_u, n_dim,
                                              ham, wfs, dens, occ,
                                              c_ref
                                              )
            del x_mat_u
        else:
            pass

        der_phi = 0.0
        for k in p_mat_u.keys():
            if self.dtype is complex:
                il1 = np.tril_indices(p_mat_u[k].shape[0])
            else:
                il1 = np.tril_indices(p_mat_u[k].shape[0], -1)

            der_phi += np.dot(g_mat_u[k][il1].conj(),
                              p_mat_u[k][il1]).real
            # der_phi += dotc(g_mat_u[k][il1],
            #                 p_mat_u[k][il1]).real

        der_phi = wfs.kd.comm.sum(der_phi)

        return phi, der_phi, g_mat_u

    def update_preconditioning_and_ref_orbitals(self, ham, wfs, dens,
                                                occ, use_prec):
        counter = 25
        if self.iters % counter == 0 or self.iters == 1:
            if self.iters > 1:
                # print('update')
                # we need to update eps_n, f_n
                super().iterate(ham, wfs)
                occ.calculate(wfs)
                # probably choose new reference orbitals?
                self.initialize_2(wfs, dens)
        if use_prec:
            if self.sda != 'LBFGS_P':
                if self.iters % counter == 0 or self.iters == 1:
                    self.precond = {}
                    for kpt in wfs.kpt_u:
                        k = self.n_kps * kpt.s + kpt.q
                        heiss = self.get_hessian(kpt)
                        if self.dtype is float:
                            self.precond[k] = np.zeros_like(heiss)
                            for i in range(heiss.shape[0]):
                                if abs(heiss[i]) < 1.0e-4:
                                    self.precond[k][i] = 1.0
                                else:
                                    self.precond[k][i] = \
                                        1.0 / (heiss[i].real)
                        else:
                            self.precond[k] = np.zeros_like(heiss)
                            for i in range(heiss.shape[0]):
                                if abs(heiss[i]) < 1.0e-4:
                                    self.precond[k][i] = 1.0 + 1.0j
                                else:
                                    self.precond[k][i] = 1.0 / \
                                                         heiss[i].real + \
                                                         1.0j / \
                                                         heiss[i].imag
                    return self.precond
                else:
                    return self.precond
            else:
                self.precond = {}
                for kpt in wfs.kpt_u:
                    k = self.n_kps * kpt.s + kpt.q
                    heiss = self.get_hessian(kpt)
                    if self.dtype is float:
                        self.precond[k] = np.zeros_like(heiss)
                        self.precond[k] = 1.0 / (
                                0.75 * heiss +
                                0.25 * self.search_direction.beta_0 ** (-1))
                    else:
                        self.precond[k] = np.zeros_like(heiss)
                        self.precond[k] = \
                            1.0 / (0.75 * heiss.real +
                                   0.25 * self.search_direction.beta_0 ** (-1)) + \
                            1.0j / (0.75 * heiss.imag +
                                    0.25 * self.search_direction.beta_0 ** (-1))
                return self.precond
        else:
            return None

    def get_hessian(self, kpt):
        f_n = kpt.f_n
        eps_n = kpt.eps_n
        if self.dtype is complex:
            il1 = np.tril_indices(eps_n.shape[0])
        else:
            il1 = np.tril_indices(eps_n.shape[0], -1)
        il1 = list(il1)
        heiss = np.zeros(len(il1[0]), dtype=self.dtype)
        x = 0
        for l, m in zip(*il1):
            df = f_n[l] - f_n[m]
            heiss[x] = -2.0 * (eps_n[l] - eps_n[m]) * df
            if self.dtype is complex:
                heiss[x] += 1.0j * heiss[x]
                if abs(heiss[x]) < 1.0e-10:
                    heiss[x] = 0.0 + 0.0j
            else:
                if abs(heiss[x]) < 1.0e-10:
                    heiss[x] = 0.0
            x += 1

        return heiss

    def calculate_residual(self, kpt, H_MM, S_MM, wfs):
        return np.inf

    def get_canonical_representation(self, ham, wfs, dens):
        # choose canonical orbitals which diagonolize
        # largange matrix. need to do subspace rotation but
        # this also must work
        super().iterate(ham, wfs)
        self.initialize_2(wfs, dens)

    def reset(self):
        self._error = np.inf
        self.iters = 0
