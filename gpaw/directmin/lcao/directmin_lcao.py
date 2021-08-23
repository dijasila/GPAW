"""
A class for finding optimal
orbitals of the KS-DFT or PZ-SIC
functionals using exponential transformation
direct minimization

arXiv:2101.12597 [physics.comp-ph]
Comput. Phys. Commun. 267, 108047 (2021).
https://doi.org/10.1016/j.cpc.2021.108047
"""


import numpy as np
from ase.parallel import parprint
from gpaw.utilities.blas import mmm
from gpaw.directmin.lcao.tools import expm_ed, expm_ed_unit_inv
from gpaw.lcao.eigensolver import DirectLCAO
from scipy.linalg import expm
from gpaw.utilities.tools import tri2full
from gpaw.directmin.lcao import search_direction, line_search_algorithm
from gpaw.directmin.functional.lcao import get_functional
from gpaw import BadParallelization


class DirectMinLCAO(DirectLCAO):

    def __init__(self, diagonalizer=None,
                 searchdir_algo='l-bfgs-p',
                 linesearch_algo='swc-awc',
                 update_ref_orbs_counter=20,
                 update_ref_orbs_canonical=False,
                 update_precond_counter=1000,
                 use_prec=True, matrix_exp='pade-approx',
                 representation='sparse',
                 functional='ks',
                 orthonormalization='gramschmidt',
                 randomizeorbitals=False,
                 checkgraderror=False,
                 localizationtype=None,
                 need_localization=True,
                 need_init_orbs=True
                 ):
        """
        This class performs the exponential transformation
        direct minimization:
        E = E[C_ref e^{A}]
        C_ref is reference orbitals
        A is a skew-hermitian matrix which needs to be found

        :param diagonalizer: inherent from direct-lcao eigensolver
        :param searchdir_algo: algo for calc search direction (e.g.LBFGS)
        :param linesearch_algo: line search (e.g. strong Wolfe conditions)
        :param update_ref_orbs_counter: (when to update C_ref)
        :param update_ref_orbs_canonical: update C_ref to can. orb.
        :param update_precond_counter: when to update the preconditioner
        :param use_prec: use preconditioner or not
        :param matrix_exp: algorithm for calc matrix exponential and grad.
        'pade-approx', 'egdecomp', 'egdecomp-u-invar' (used with u-invar
        represnt.),
        :param representation: the way A are stored,
        'sparse', 'full', 'u-invar',
        :param functional: KS or PZ-SIC functionals
        :param orthonormalization: gram-schmidt, loewdin, or eigenstates of ham
        'gramschmidt', 'loewdin', 'diag', respectively
        :param randomizeorbitals: if need a noise in initial guess
        :param checkgraderror: check error in estimation of gradient
        :param localizationtype: Foster-Boys, Pipek-Mezey, Edm.-Rudenb.
        :param need_localization: use localized orbitals as initial guess
        :param need_init_orbs: if false then use coef. stored in kpt.C_nM
        """

        super(DirectMinLCAO, self).__init__(diagonalizer)

        assert representation in ['sparse', 'u-invar', 'full'], 'Value Error'
        assert matrix_exp in ['egdecomp', 'egdecomp-u-invar', 'pade-approx'], \
            'Value Error'
        if matrix_exp == 'egdecomp-u-invar':
            assert representation == 'u-invar', 'Use u-invar representation ' \
                                                'with egdecomp-u-invar'
        assert orthonormalization in ['gramschmidt', 'loewdin', 'diag'], \
            'Value Error'

        self.localizationtype = localizationtype
        self.eg_count = 0
        self.update_ref_orbs_counter = update_ref_orbs_counter
        self.update_ref_orbs_canonical = update_ref_orbs_canonical
        self.update_precond_counter = update_precond_counter
        self.use_prec = use_prec
        self.matrix_exp = matrix_exp
        self.iters = 0
        self.restart = False
        self.name = 'direct-min-lcao'
        self.localizationtype = localizationtype
        self.need_localization = need_localization
        self.need_init_orbs = need_init_orbs
        self.randomizeorbitals = randomizeorbitals
        self.representation = representation
        self.orthonormalization = orthonormalization

        self.searchdir_algo = search_direction(searchdir_algo)
        if self.searchdir_algo.name == 'l-bfgs-p' and not self.use_prec:
            raise ValueError('Use l-bfgs-p with use_prec=True')

        self.line_search = line_search_algorithm(linesearch_algo,
                                                 self.evaluate_phi_and_der_phi,
                                                 self.searchdir_algo)
        self.func = get_functional(functional)

        self.checkgraderror = checkgraderror
        self._normcomm, self._normg = 0., 0.
        self._error = 0

        # these are things we cannot initialize now
        self.dtype = None
        self.nkpts = None
        # dimensionality of the problem.
        # this implementation rotates among all bands
        # values: matrices, keys: kpt number (and spins)
        self.a_mat_u = {}  # skew-hermitian matrix to be exponented
        self.g_mat_u = {}  # gradient matrix
        self.c_nm_ref = {}  # reference orbitals to be rotated
        self.evecs = {}   # eigenvectors for i*a_mat_u
        self.evals = {}   # eigenvalues for i*a_mat_u
        self.ind_up = {}
        self.n_dim = {}
        self.alpha = 1.0  # step length
        self.phi_2i = [None, None]  # energy at last two iterations
        self.der_phi_2i = [None, None]  # energy gradient w.r.t. alpha
        self.iters = 0
        self.hess = {}  # hessian for l-bfgs-p
        self.precond = {}  # precondiner for other methods

        # for mom
        self.initial_occupation_numbers = None

        self.initialized = False

    def __repr__(self):

        sda_name = self.searchdir_algo.name
        lsa_name = self.line_search.name

        sds = {'sd': 'Steepest Descent',
               'fr-cg': 'Fletcher-Reeves conj. grad. method',
               'quick-min': 'Molecular-dynamics based algorithm',
               'l-bfgs': 'L-BFGS algorithm',
               'l-bfgs-p': 'L-BFGS algorithm with preconditioning',
               'l-sr1p': 'Limited-memory SR1P algorithm'}

        lss = {'max-step': 'step size equals one',
               'parabola': 'Parabolic line search',
               'swc-awc': 'Inexact line search based on cubic interpolation,\n'
                          '                    strong and approximate Wolfe '
                          'conditions'}

        repr_string = 'Direct minimisation using exponential ' \
                      'transformation.\n'
        repr_string += '       ' \
                       'Search ' \
                       'direction: {}\n'.format(sds[sda_name])
        repr_string += '       ' \
                       'Line ' \
                       'search: {}\n'.format(lss[lsa_name])
        repr_string += '       ' \
                       'Preconditioning: {}\n'.format(self.use_prec)
        repr_string += '       ' \
                       'WARNING: do not use it for metals as ' \
                       'occupation numbers are\n' \
                       '                ' \
                       'not found variationally\n'

        return repr_string

    def init_me(self, wfs, ham, dens):

        self.dtype = wfs.dtype
        self.nkpts = wfs.kd.nibzkpts
        for kpt in wfs.kpt_u:
            u = self.kpointval(kpt)
            # dimensionality of the problem.
            # this implementation rotates among all bands
            self.n_dim[u] = wfs.bd.nbands

        # need to initialize c_nm, eps, f_n and so on.
        # for this we call super class which make one diagonalization step
        # or if we load wfs from a file or from previous step then
        # just do orthonormalization procedure
        self.initialize_orbitals(wfs, ham)

        # mom
        wfs.calculate_occupation_numbers(dens.fixed)
        occ_name = getattr(wfs.occupations, "name", None)
        if occ_name == 'mom':
            self.initial_occupation_numbers = wfs.occupations.numbers.copy()
            self.initialize_mom(wfs, dens)

        # randomize orbitals?
        if self.randomizeorbitals:
            for kpt in wfs.kpt_u:
                self.randomize_orbitals_kpt(wfs, kpt)
            self.randomizeorbitals = False

        # initialize matrices
        self.set_variable_matrices(wfs)

        # if no empty state no need to optimize
        for k in self.ind_up:
            if not self.ind_up[k][0].size or not self.ind_up[k][1].size:
                self.n_dim[k] = 0

        self.initialized = True

    def set_variable_matrices(self, wfs):

        # Matrices are sparse and Skew-Hermitian.
        # They have this structure:
        #  A_BigMatrix =
        #
        # (  A_1          A_2 )
        # ( -A_2.T.conj() 0   )
        #
        # where 0 is a zero-matrix of size of (M-N) * (M-N)
        #
        # A_1 i skew-hermitian matrix of N * N,
        # N-number of occupied states
        # A_2 is matrix of size of (M-N) * N,
        # M - number of basis functions
        #
        # if the energy functional is unitary invariant
        # then A_1 = 0
        # (see Hutter J et. al, J. Chem. Phys. 101, 3862 (1994))
        #
        # We will keep A_1 as we would like to work with metals,
        # SIC, and molecules with different occupation numbers.
        # this corresponds to 'sparse' representation
        #
        # Thus, for the 'sparse' we need to store upper
        # triangular part of A_1, and matrix A_2, so in total
        # (M-N) * N + N * (N - 1)/2 = N * (M - (N + 1)/2) elements
        #
        # we will store these elements as a vector and
        # also will store indices of the A_BigMatrix
        # which correspond to these elements.
        #
        # 'u-invar' corresponds to the case when we want to
        # store only A_2, that is this representaion is sparser

        for kpt in wfs.kpt_u:
            n_occ = get_n_occ(kpt)
            u = self.kpointval(kpt)
            # M - one dimension of the A_BigMatrix
            M = self.n_dim[u]
            if self.representation == 'sparse':
                # let's take all upper triangular indices
                # of A_BigMatrix and then delete indices from ind_up
                # which correspond to 0 matrix in in A_BigMatrix.
                ind_up = np.triu_indices(self.n_dim[u], 1)
                zero_ind = -((M - n_occ) * (M - n_occ - 1)) // 2
                if zero_ind == 0:
                    zero_ind = None
                self.ind_up[u] = (ind_up[0][:zero_ind].copy(),
                                  ind_up[1][:zero_ind].copy())
            elif self.representation == 'u-invar':
                i1, i2 = [], []
                for i in range(n_occ):
                    for j in range(n_occ, M):
                        i1.append(i)
                        i2.append(j)
                self.ind_up[u] = (np.asarray(i1), np.asarray(i2))
            else:
                self.ind_up[u] = None

            if self.representation in ['sparse', 'u-invar']:
                shape_of_arr = len(self.ind_up[u][0])
            else:
                shape_of_arr = (self.n_dim[u], self.n_dim[u])

            self.a_mat_u[u] = np.zeros(shape=shape_of_arr, dtype=self.dtype)
            self.g_mat_u[u] = np.zeros(shape=shape_of_arr, dtype=self.dtype)
            # use initial KS orbitals
            self.c_nm_ref[u] = np.copy(kpt.C_nM[:self.n_dim[u]])
            self.evecs[u] = None
            self.evals[u] = None

        self.iters = 1

    def iterate(self, ham, wfs, dens, log):
        """
        One iteration of direct optimization
        for occupied states

        :param ham:
        :param wfs:
        :param dens:
        :param log:
        :return:
        """
        with wfs.timer('Direct Minimisation step'):

            self.check_assertions(wfs, dens)
            self.update_ref_orbitals(wfs, ham, dens)

            with wfs.timer('Preconditioning:'):
                precond = self.update_preconditioning(wfs, self.use_prec)

            a_mat_u = self.a_mat_u
            n_dim = self.n_dim
            alpha = self.alpha
            phi_2i = self.phi_2i
            der_phi_2i = self.der_phi_2i
            c_ref = self.c_nm_ref

            if self.iters == 1:
                phi_2i[0], g_mat_u = \
                    self.get_energy_and_gradients(a_mat_u, n_dim, ham, wfs,
                                                  dens, c_ref)
            else:
                g_mat_u = self.g_mat_u

            with wfs.timer('Get Search Direction'):
                p_mat_u = self.get_search_direction(a_mat_u, g_mat_u,
                                                    precond, wfs)

            # recalculate derivative with new search direction
            der_phi_2i[0] = 0.0
            for k in g_mat_u:
                if self.representation in ['sparse', 'u-invar']:
                    der_phi_2i[0] += g_mat_u[k].conj() @ p_mat_u[k]
                else:
                    il1 = get_indices(g_mat_u[k].shape[0], self.dtype)
                    der_phi_2i[0] += g_mat_u[k][il1].conj() @ p_mat_u[k][il1]
            der_phi_2i[0] = der_phi_2i[0].real
            der_phi_2i[0] = wfs.kd.comm.sum(der_phi_2i[0])

            alpha, phi_alpha, der_phi_alpha, g_mat_u = \
                self.line_search.step_length_update(a_mat_u, p_mat_u,
                                                    n_dim, ham, wfs, dens,
                                                    c_ref,
                                                    phi_0=phi_2i[0],
                                                    der_phi_0=der_phi_2i[0],
                                                    phi_old=phi_2i[1],
                                                    der_phi_old=der_phi_2i[1],
                                                    alpha_max=5.0,
                                                    alpha_old=alpha,
                                                    kpdescr=wfs.kd)

            if wfs.gd.comm.size > 1:
                with wfs.timer('Broadcast gradients'):
                    alpha_phi_der_phi = np.array([alpha, phi_2i[0],
                                                  der_phi_2i[0]])
                    wfs.gd.comm.broadcast(alpha_phi_der_phi, 0)
                    alpha = alpha_phi_der_phi[0]
                    phi_2i[0] = alpha_phi_der_phi[1]
                    der_phi_2i[0] = alpha_phi_der_phi[2]
                    for kpt in wfs.kpt_u:
                        k = self.kpointval(kpt)
                        wfs.gd.comm.broadcast(g_mat_u[k], 0)

            # calculate new matrices for optimal step length
            for k in a_mat_u:
                a_mat_u[k] += alpha * p_mat_u[k]
            self.alpha = alpha
            self.g_mat_u = g_mat_u
            self.iters += 1

            # and 'shift' phi, der_phi for the next iteration
            phi_2i[1], der_phi_2i[1] = phi_2i[0], der_phi_2i[0]
            phi_2i[0], der_phi_2i[0] = phi_alpha, der_phi_alpha,

    def get_energy_and_gradients(self, a_mat_u, n_dim, ham, wfs, dens,
                                 c_nm_ref):

        """
        Energy E = E[C exp(A)]. Gradients G_ij[C, A] = dE/dA_ij

        :param a_mat_u: A
        :param c_nm_ref: C
        :param n_dim:
        :return:
        """

        self.rotate_wavefunctions(wfs, a_mat_u, n_dim, c_nm_ref)

        e_total = self.update_ks_energy(ham, wfs, dens)

        with wfs.timer('Calculate gradients'):
            g_mat_u = {}
            self._error = 0.0
            self.e_sic = 0.0  # this is odd energy
            for kpt in wfs.kpt_u:
                k = self.kpointval(kpt)
                if n_dim[k] == 0:
                    g_mat_u[k] = np.zeros_like(a_mat_u[k])
                    continue
                h_mm = self.calculate_hamiltonian_matrix(ham, wfs, kpt)
                # make matrix hermitian
                tri2full(h_mm)
                g_mat_u[k], error = self.func.get_gradients(
                    h_mm, kpt.C_nM, kpt.f_n, self.evecs[k], self.evals[k],
                    kpt, wfs, wfs.timer, self.matrix_exp,
                    self.representation, self.ind_up[k])

                self._error += error
            self._error = wfs.kd.comm.sum(self._error)
            self.e_sic = wfs.kd.comm.sum(self.e_sic)

        self.eg_count += 1

        if self.representation == 'full' and self.checkgraderror:
            norm = 0.0
            for kpt in wfs.kpt_u:
                u = self.kpointval(kpt)
                normt = np.linalg.norm(g_mat_u[u] @ self.a_mat_u[u] -
                                       self.a_mat_u[u] @ g_mat_u[u])
                if norm < normt:
                    norm = normt
            norm2 = 0.0
            for kpt in wfs.kpt_u:
                u = self.kpointval(kpt)
                normt = np.linalg.norm(g_mat_u[u])
                if norm2 < normt:
                    norm2 = normt

            self._normcomm = norm
            self._normg = norm2

        return e_total + self.e_sic, g_mat_u

    def update_ks_energy(self, ham, wfs, dens):
        """
        Update Kohn-Sham energy
        It assumes the temperature is zero K.
        """

        dens.update(wfs)
        ham.update(dens, wfs, False)

        return ham.get_energy(0.0, wfs, False)

    def get_search_direction(self, a_mat_u, g_mat_u, precond, wfs):
        """
        calculate search direction according to chosen
        optimization algorithm (L-BFGS for example)
        """

        if self.representation in ['sparse', 'u-invar']:
            p_mat_u = self.searchdir_algo.update_data(wfs, a_mat_u, g_mat_u,
                                                      precond)
        else:
            g_vec = {}
            a_vec = {}

            for k in a_mat_u:
                il1 = get_indices(a_mat_u[k].shape[0], self.dtype)
                a_vec[k] = a_mat_u[k][il1]
                g_vec[k] = g_mat_u[k][il1]

            p_vec = self.searchdir_algo.update_data(wfs, a_vec, g_vec, precond)
            p_mat_u = {}
            for k in p_vec:
                p_mat_u[k] = np.zeros_like(a_mat_u[k])
                il1 = get_indices(p_mat_u[k].shape[0], self.dtype)
                p_mat_u[k][il1] = p_vec[k]
                # make it skew-hermitian
                il1 = np.tril_indices(p_mat_u[k].shape[0], -1)
                p_mat_u[k][(il1[1], il1[0])] = -p_mat_u[k][il1].conj()

        return p_mat_u

    def evaluate_phi_and_der_phi(self, a_mat_u, p_mat_u, n_dim, alpha,
                                 ham, wfs, dens, c_ref,
                                 phi=None, g_mat_u=None):
        """
        phi = f(x_k + alpha_k*p_k)
        der_phi = \\grad f(x_k + alpha_k*p_k) \\cdot p_k
        :return:  phi, der_phi # floats
        """
        if phi is None or g_mat_u is None:
            x_mat_u = {k: a_mat_u[k] + alpha * p_mat_u[k]
                       for k in a_mat_u}
            phi, g_mat_u = \
                self.get_energy_and_gradients(x_mat_u, n_dim,
                                              ham, wfs, dens,
                                              c_ref)

        der_phi = 0.0
        if self.representation in ['sparse', 'u-invar']:
            for k in p_mat_u:
                der_phi += g_mat_u[k].conj() @ p_mat_u[k]
        else:
            for k in p_mat_u:
                il1 = get_indices(p_mat_u[k].shape[0], self.dtype)
                der_phi += g_mat_u[k][il1].conj() @ p_mat_u[k][il1]

        der_phi = der_phi.real
        der_phi = wfs.kd.comm.sum(der_phi)

        return phi, der_phi, g_mat_u

    def update_ref_orbitals(self, wfs, ham, dens):
        """
        update orbitals which are chosen as reference
        orbitals to which rotation is applied

        :param wfs:
        :param ham:
        :return:
        """

        if self.representation == 'full':
            badgrad = self._normcomm > self._normg / 3. and self.checkgraderror
        else:
            badgrad = False
        counter = self.update_ref_orbs_counter
        if (self.iters % counter == 0 and self.iters > 1) or \
                (self.restart and self.iters > 1) or badgrad:
            self.iters = 1
            if self.update_ref_orbs_canonical or self.restart:
                self.get_canonical_representation(ham, wfs, dens)
            else:
                for kpt in wfs.kpt_u:
                    u = self.kpointval(kpt)
                    self.c_nm_ref[u] = kpt.C_nM.copy()
                    self.a_mat_u[u] = np.zeros_like(self.a_mat_u[u])

            # choose search direction and line search algorithm
            # as you need to restart it
            self.searchdir_algo = search_direction(self.searchdir_algo.name)
            self.line_search = \
                line_search_algorithm(self.line_search.name,
                                      self.evaluate_phi_and_der_phi,
                                      self.searchdir_algo)

    def update_preconditioning(self, wfs, use_prec):
        """
        update preconditioning
        """

        counter = self.update_precond_counter
        if use_prec:
            if self.searchdir_algo.name != 'l-bfgs-p':
                if self.iters % counter == 0 or self.iters == 1:
                    for kpt in wfs.kpt_u:
                        k = self.kpointval(kpt)
                        hess = self.get_hessian(kpt)
                        if self.dtype is float:
                            self.precond[k] = np.zeros_like(hess)
                            for i in range(hess.shape[0]):
                                if abs(hess[i]) < 1.0e-4:
                                    self.precond[k][i] = 1.0
                                else:
                                    self.precond[k][i] = \
                                        1.0 / (hess[i].real)
                        else:
                            self.precond[k] = np.zeros_like(hess)
                            for i in range(hess.shape[0]):
                                if abs(hess[i]) < 1.0e-4:
                                    self.precond[k][i] = 1.0 + 1.0j
                                else:
                                    self.precond[k][i] = \
                                        1.0 / hess[i].real + \
                                        1.0j / hess[i].imag
                    return self.precond
                else:
                    return self.precond
            else:
                # it's a bit messy, here you store self.heis,
                # but in 'if' above self.precond
                precond = {}
                for kpt in wfs.kpt_u:
                    k = self.kpointval(kpt)
                    w = kpt.weight / (3.0 - wfs.nspins)
                    if self.iters % counter == 0 or self.iters == 1:
                        self.hess[k] = self.get_hessian(kpt)
                    hess = self.hess[k]
                    beta0 = self.searchdir_algo.beta_0
                    if self.dtype is float:
                        precond[k] = \
                            1. / (0.75 * hess +
                                  w * 0.25 * beta0 ** (-1))
                    else:
                        precond[k] = \
                            1. / (0.75 * hess.real +
                                  w * 0.25 * beta0 ** (-1)) + \
                            1.j / (0.75 * hess.imag +
                                   w * 0.25 * beta0 ** (-1))
                return precond
        else:
            return None

    def get_hessian(self, kpt):
        """
        calculate approximate hessian:

        h_{lm, lm} = -2.0 * (eps_n[l] - eps_n[m]) * (f[l] - f[m])
        other elements are zero
        :param kpt:
        :return:
        """

        f_n = kpt.f_n
        eps_n = kpt.eps_n
        if self.representation in ['sparse', 'u-invar']:
            u = self.kpointval(kpt)
            il1 = list(self.ind_up[u])
        else:
            il1 = get_indices(eps_n.shape[0], self.dtype)
            il1 = list(il1)

        hess = np.zeros(len(il1[0]), dtype=self.dtype)
        x = 0
        for l, m in zip(*il1):
            df = f_n[l] - f_n[m]
            hess[x] = -2.0 * (eps_n[l] - eps_n[m]) * df
            if self.dtype is complex:
                hess[x] += 1.0j * hess[x]
                if abs(hess[x]) < 1.0e-10:
                    hess[x] = 0.0 + 0.0j
            else:
                if abs(hess[x]) < 1.0e-10:
                    hess[x] = 0.0
            x += 1

        return hess

    def get_canonical_representation(self, ham, wfs, dens,
                                     sort_eigenvalues=False):
        """
        choose canonical orbitals which diagonalise
        lagrange matrix. it's probably necessary
        to do subspace rotation with equally
        occupied states.
        In this case, the total energy remains the same,
        as it's unitary invariant within equally occupied subspaces.
        """

        with wfs.timer('Get canonical representation'):
            for kpt in wfs.kpt_u:
                # wfs.atomic_correction.calculate_projections(wfs, kpt)
                h_mm = self.calculate_hamiltonian_matrix(ham, wfs, kpt)
                tri2full(h_mm)
                if self.update_ref_orbs_canonical or self.restart:
                    # Diagonalize entire Hamiltonian matrix
                    with wfs.timer('Diagonalize and rotate'):
                        kpt.C_nM, kpt.eps_n = rotate_subspace(
                            h_mm, kpt.C_nM)
                else:
                    # Diagonalize equally occupied subspaces separately
                    n_init = 0
                    while True:
                        n_fin = \
                            find_equally_occupied_subspace(kpt.f_n, n_init)
                        with wfs.timer('Diagonalize and rotate'):
                            kpt.C_nM[n_init:n_init + n_fin, :], \
                                kpt.eps_n[n_init:n_init + n_fin] = \
                                rotate_subspace(
                                    h_mm, kpt.C_nM[n_init:n_init + n_fin, :])
                        n_init += n_fin
                        if n_init == len(kpt.f_n):
                            break
                        elif n_init > len(kpt.f_n):
                            raise RuntimeError('Bug is here!')

                with wfs.timer('Calculate projections'):
                    wfs.atomic_correction.calculate_projections(wfs, kpt)
            self._e_entropy = wfs.calculate_occupation_numbers(dens.fixed)
            occ_name = getattr(wfs.occupations, "name", None)
            if occ_name == 'mom':
                if not sort_eigenvalues:
                    self.sort_wavefunctions_mom(wfs)
                else:
                    self.sort_wavefunctions(ham, wfs, use_eps=True)
                    not_update = not wfs.occupations.update_numbers
                    fixed_occ = wfs.occupations.use_fixed_occupations
                    if not_update or fixed_occ:
                        wfs.occupations.numbers = \
                            self.initial_occupation_numbers

            for kpt in wfs.kpt_u:
                u = self.kpointval(kpt)
                self.c_nm_ref[u] = kpt.C_nM.copy()
                self.a_mat_u[u] = np.zeros_like(self.a_mat_u[u])

    def reset(self):
        super(DirectMinLCAO, self).reset()
        self._error = np.inf
        self.initialized = False

    def sort_wavefunctions(self, ham, wfs, use_eps=False):
        """
        Sort wavefunctions according to the eigenvalues or
        the diagonal elements of the Hamiltonian matrix.
        """
        with wfs.timer('Sort WFS'):
            for kpt in wfs.kpt_u:
                if use_eps:
                    orbital_energies = kpt.eps_n
                else:
                    h_mm = self.calculate_hamiltonian_matrix(ham, wfs, kpt)
                    tri2full(h_mm)
                    hc_mn = np.zeros(shape=(kpt.C_nM.shape[1],
                                            kpt.C_nM.shape[0]),
                                     dtype=kpt.C_nM.dtype)
                    mmm(1.0, h_mm.conj(), 'N', kpt.C_nM, 'T', 0.0, hc_mn)
                    mmm(1.0, kpt.C_nM.conj(), 'N', hc_mn, 'N', 0.0, h_mm)
                    orbital_energies = h_mm.diagonal().real.copy()
                # label each orbital energy
                # add some noise to get rid off degeneracy
                orbital_energies += \
                    np.random.rand(len(orbital_energies)) * 1.0e-8
                oe_labeled = {}
                for i, lamb in enumerate(orbital_energies):
                    oe_labeled[str(round(lamb, 12))] = i
                # now sort orb energies
                oe_sorted = np.sort(orbital_energies)
                # run over sorted orbital energies and get their label
                ind = []
                for x in oe_sorted:
                    i = oe_labeled[str(round(x, 12))]
                    ind.append(i)
                # check if it is necessary to sort
                x = np.max(abs(np.array(ind) - np.arange(len(ind))))
                if x > 0:
                    # now sort wfs according to orbital energies
                    kpt.C_nM[np.arange(len(ind)), :] = kpt.C_nM[ind, :]
                    kpt.f_n[np.arange(len(ind))] = kpt.f_n[ind]
                    kpt.eps_n[np.arange(len(ind))] = orbital_energies[ind]
                    occ_name = getattr(wfs.occupations, "name", None)
                    if occ_name == 'mom':
                        # OccupationsMOM.numbers needs to be updated after
                        # sorting
                        self.update_mom_numbers(wfs, kpt)

    def sort_wavefunctions_mom(self, wfs):
        """
        Sort wave functions according to the occupation
        numbers so that there are no holes in the
        distribution of occupation numbers
        :return:
        """
        changedocc = False
        for kpt in wfs.kpt_u:
            f_sn = kpt.f_n.copy()
            if wfs.gd.comm.rank == 0:
                occupied = kpt.f_n > 1.0e-10
                n_occ = len(kpt.f_n[occupied])
                if n_occ == 0.0:
                    return
                if np.min(kpt.f_n[:n_occ]) == 0:
                    ind_occ = np.argwhere(occupied)
                    ind_unocc = np.argwhere(~occupied)
                    ind = np.vstack((ind_occ, ind_unocc))
                    kpt.C_nM = np.squeeze(kpt.C_nM[ind])
                    kpt.f_n = np.squeeze(kpt.f_n[ind])
                    kpt.eps_n = np.squeeze(kpt.eps_n[ind])
            # Broadcast coefficients, occupation numbers, eigenvalues
            wfs.gd.comm.broadcast(kpt.eps_n, 0)
            wfs.gd.comm.broadcast(kpt.f_n, 0)
            wfs.gd.comm.broadcast(kpt.C_nM, 0)
            if not np.allclose(kpt.f_n, f_sn):
                changedocc = True
                wfs.atomic_correction.calculate_projections(wfs, kpt)
                # OccupationsMOM.numbers needs to be updated after sorting
                self.update_mom_numbers(wfs, kpt)

        return changedocc

    def todict(self):

        return {'name': self.name,
                'searchdir_algo': self.searchdir_algo.todict(),
                'linesearch_algo': self.line_search.todict(),
                'localizationtype': self.localizationtype,
                'update_ref_orbs_counter': self.update_ref_orbs_counter,
                'update_precond_counter': self.update_precond_counter,
                'use_prec': self.use_prec,
                'matrix_exp': self.matrix_exp,
                'representation': self.representation,
                'functional': self.func.todict(),
                'orthonormalization': self.orthonormalization
                }

    def finite_diff_appr_of_derivative(self, ham, wfs, dens, c_nm_ref=None,
                                       eps=1.0e-7, random_amat=False,
                                       update_c_nm_ref=False, seed=None,
                                       what2calc='gradient'):

        """
           calculate gradient or hessian with respect to skew-hermitian
           matrix using finite differences with random noise
           this is usually used to test the implementation of anal. gradient

        :param ham:
        :param wfs:
        :param dens:
        :param c_nm_ref: reference orbitals
        :param eps: finite difference step
        :param random_amat: start at random skew-hermitian matrix
        :param update_c_nm_ref: before calculations do c_ref <- c_ref e*A
        :param seed: seed for random generator
        :param what2calc: gradient or hessian
        :return: analytical and numerical
        """

        assert self.representation in ['sparse', 'u-invar']
        assert what2calc in ['gradient', 'hessian']

        a_m = {u: np.zeros_like(v) for u, v in self.a_mat_u.items()}
        n_dim = self.n_dim
        # total dimensionality if matrices are real:
        dim = sum([len(a) for a in a_m.values()])
        matrix_exp = self.matrix_exp
        g_a = None
        g_n = None
        hess_a = None
        hess_n = None
        dim_z, disp = (2, [eps, 1.0j * eps]) \
            if self.dtype == complex else (1, [eps])

        if what2calc == 'gradient':
            g_n = {u: np.zeros_like(v) for u, v in a_m.items()}
        else:
            hess_n = np.zeros(shape=(dim_z * dim, dim_z * dim))
            # have to use exact gradient when hessian is calculated
            self.matrix_exp = 'egdecomp'

        if c_nm_ref is None:
            c_nm_ref = self.c_nm_ref

        # init matrices and use random if needed
        for kpt in wfs.kpt_u:
            u = self.kpointval(kpt)
            if random_amat:
                a = random_a(a_m[u].shape, wfs.dtype, seed)
                wfs.gd.comm.broadcast(a, 0)
                a_m[u] = a
            # update ref orbitals if needed
            if update_c_nm_ref:
                amat = vec2skewmat(a_m[u], n_dim[u], self.ind_up[u], wfs.dtype)
                u_nn = expm(amat)
                c_nm_ref[u] = u_nn.T @ kpt.C_nM[:u_nn.shape[0]]
                a_m[u] = np.zeros_like(a_m[u])

        if what2calc == 'gradient':
            # calc analytical gradient
            g_a = self.get_energy_and_gradients(a_m, n_dim, ham, wfs, dens,
                                                c_nm_ref)[1]
            tmp = 0
        else:
            # calc analytical approximation to hessian
            hess_a = []
            for kpt in wfs.kpt_u:
                hess_a += list(self.get_hessian(kpt).copy())
            hess_a = np.diag(hess_a)
            tmp = 1

        row = 0
        for z in range(dim_z):
            for kpt in wfs.kpt_u:
                u = self.kpointval(kpt)
                for i in range(len(a_m[u])):
                    parprint(z, u, i)
                    a = a_m[u][i]
                    a_m[u][i] = a + disp[z]
                    valp = self.get_energy_and_gradients(
                        a_m, n_dim, ham, wfs, dens, c_nm_ref)[tmp]
                    a_m[u][i] = a - disp[z]
                    valm = self.get_energy_and_gradients(
                        a_m, n_dim, ham, wfs, dens, c_nm_ref)[tmp]
                    if what2calc == 'gradient':
                        g_n[u][i] += disp[z] * (valp - valm) * 0.5 / eps ** 2
                    else:
                        hess = []
                        for k in range(len(wfs.kpt_u)):
                            hess += list((valp[k] - valm[k]) * 0.5 / eps)
                        hess = np.asarray(hess)
                        if dim_z == 2:
                            hessc = np.zeros(shape=dim_z * dim)
                            hessc[: dim] = np.real(hess)
                            hessc[dim:] = np.imag(hess)
                            hess_n[row] = hessc
                        else:
                            hess_n[row] = hess
                    row += 1
                    a_m[u][i] = a

        if what2calc == 'gradient':
            return g_a, g_n
        else:
            self.matrix_exp = matrix_exp
            return hess_a, hess_n

    def rotate_wavefunctions(self, wfs, a_mat_u, n_dim, c_nm_ref):

        """
        Appply unitary transformation U = exp(a_mat_u) to
        coefficients c_nm_ref

        :param wfs:
        :param a_mat_u:
        :param n_dim:
        :param c_nm_ref:
        :return:
        """

        with wfs.timer('Unitary rotation'):
            for kpt in wfs.kpt_u:
                k = self.kpointval(kpt)
                if n_dim[k] == 0:
                    continue

                if self.gd.comm.rank == 0:
                    if self.representation in ['sparse', 'u-invar']:
                        if self.matrix_exp == 'egdecomp-u-invar' and \
                                self.representation == 'u-invar':
                            n_occ = get_n_occ(kpt)
                            n_v = wfs.bd.nbands - n_occ
                            a = a_mat_u[k].reshape(n_occ, n_v)
                        else:
                            a = vec2skewmat(a_mat_u[k], n_dim[k],
                                            self.ind_up[k], self.dtype)
                    else:
                        a = a_mat_u[k]

                    if self.matrix_exp == 'pade-approx':
                        # this function takes a lot of memory
                        # for large matrices... what can we do?
                        with wfs.timer('Pade Approximants'):
                            u_nn = expm(a)
                    elif self.matrix_exp == 'egdecomp':
                        # this method is based on diagonalisation
                        with wfs.timer('Eigendecomposition'):
                            u_nn, evecs, evals = expm_ed(a, evalevec=True)
                    elif self.matrix_exp == 'egdecomp-u-invar':
                        with wfs.timer('Eigendecomposition'):
                            u_nn = expm_ed_unit_inv(a, oo_vo_blockonly=False)
                    else:
                        raise ValueError('Check the keyword '
                                         'for matrix_exp. \n'
                                         'Must be '
                                         '\'pade-approx\' or '
                                         '\'egdecomp\'')

                    dimens1 = u_nn.shape[0]
                    dimens2 = u_nn.shape[1]
                    kpt.C_nM[:dimens2] = u_nn.T @ c_nm_ref[k][:dimens1]

                with wfs.timer('Broadcast coefficients'):
                    self.gd.comm.broadcast(kpt.C_nM, 0)

                if self.matrix_exp == 'egdecomp':
                    with wfs.timer('Broadcast evecs and evals'):
                        if self.gd.comm.rank != 0:
                            evecs = np.zeros(shape=(n_dim[k], n_dim[k]),
                                             dtype=complex)
                            evals = np.zeros(shape=n_dim[k],
                                             dtype=float)

                        self.gd.comm.broadcast(evecs, 0)
                        self.gd.comm.broadcast(evals, 0)
                        self.evecs[k], self.evals[k] = evecs, evals
                with wfs.timer('Calculate projections'):
                    wfs.atomic_correction.calculate_projections(wfs, kpt)

    def initialize_orbitals(self, wfs, ham):
        """
        if it is the first use of the scf then initialize
        coefficient matrix using eigensolver
        and then localise orbitals
        """

        # if it is the first use of the scf then initialize
        # coefficient matrix using eigensolver
        orthname = self.orthonormalization
        need_canon_coef = \
            (not wfs.coefficients_read_from_file and self.need_init_orbs)
        if need_canon_coef or orthname == 'diag':
            super(DirectMinLCAO, self).iterate(ham, wfs)
        else:
            wfs.orthonormalize(type=orthname)
        wfs.coefficients_read_from_file = False
        self.need_init_orbs = False

    def check_assertions(self, wfs, dens):

        assert dens.mixer.driver.basemixerclass.name == 'no-mixing', \
            'Please, use: mixer={\'backend\': \'no-mixing\'}'
        assert wfs.bd.nbands == wfs.basis_functions.Mmax, \
            'Please, use: nbands=\'nao\''
        if not wfs.bd.comm.size == 1:
            raise BadParallelization(
                'Band parallelization is not supported')
        if wfs.ksl.using_blacs:
            raise BadParallelization(
                'ScaLapack parallelization is not supported')
        if wfs.occupations.name != 'mom':
            errormsg = \
                'Please, use occupations={\'name\': \'fixed-uniform\'}'
            assert wfs.occupations.name == 'fixed-uniform', errormsg

    def initialize_mom(self, wfs, dens):
        # Reinitialize the MOM reference orbitals
        # after orthogonalization/localization
        wfs.occupations.initialize_reference_orbitals()
        wfs.calculate_occupation_numbers(dens.fixed)
        self.sort_wavefunctions_mom(wfs)

    def check_mom(self, wfs, dens):
        occ_name = getattr(wfs.occupations, "name", None)
        if occ_name == 'mom':
            self._e_entropy = wfs.calculate_occupation_numbers(dens.fixed)
            self.restart = self.sort_wavefunctions_mom(wfs)

    def update_mom_numbers(self, wfs, kpt):
        if wfs.collinear and wfs.nspins == 1:
            degeneracy = 2
        else:
            degeneracy = 1
        wfs.occupations.numbers[kpt.s] = \
            kpt.f_n / (kpt.weightk * degeneracy)

    def randomize_orbitals_kpt(self, wfs, kpt):
        """
        add random noise to C_nM but keep them still orthonormal
        """
        nst = kpt.C_nM.shape[0]
        wt = kpt.weight * 0.01
        arand = wt * random_a((nst, nst), wfs.dtype, None)
        arand = arand - arand.T.conj()
        wfs.gd.comm.broadcast(arand, 0)
        kpt.C_nM[:] = expm(arand) @ kpt.C_nM[:]
        wfs.atomic_correction.calculate_projections(wfs, kpt)

    def kpointval(self, kpt):
        return self.nkpts * kpt.s + kpt.q

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, e):
        self._error = e


def get_indices(dimens, dtype):

    if dtype == complex:
        il1 = np.tril_indices(dimens)
    else:
        il1 = np.tril_indices(dimens, -1)

    return il1


def get_n_occ(kpt):
    """
    get number of occupied orbitals
    """
    nbands = len(kpt.f_n)
    n_occ = 0
    while n_occ < nbands and kpt.f_n[n_occ] > 1e-10:
        n_occ += 1
    return n_occ


def find_equally_occupied_subspace(f_n, index=0):
    n_occ = 0
    f1 = f_n[index]
    for f2 in f_n[index:]:
        if abs(f1 - f2) < 1.0e-8:
            n_occ += 1
        else:
            return n_occ
    return n_occ


def rotate_subspace(h_mm, c_nm):
    """
    choose canonical orbitals
    """
    l_nn = (c_nm @ h_mm @ c_nm.conj().T).conj()
    # check if diagonal then don't rotate? it could save a bit of time
    eps, w = np.linalg.eigh(l_nn)
    return w.T.conj() @ c_nm, eps


def random_a(shape, dtype, seed):

    np.random.seed(seed)
    a = np.random.random_sample(shape)
    if dtype == complex:
        a = a.astype(complex)
        a += 1.0j * np.random.random_sample(shape)

    return a


def vec2skewmat(a_vec, dim, ind_up, dtype):

    a = np.zeros(shape=(dim, dim), dtype=dtype)
    a[ind_up] = a_vec
    a -= a.T.conj()

    return a
