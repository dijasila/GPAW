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
from gpaw.directmin.tools import expm_ed, expm_ed_unit_inv
from gpaw.directmin.lcao.directmin_lcao import DirectMinLCAO
from scipy.linalg import expm
from gpaw.directmin import search_direction, line_search_algorithm
from gpaw.directmin.functional import get_functional
from gpaw import BadParallelization


class ETDM:

    """
    Exponential Transformation Direct Minimzation (ETDM)
    """

    def __init__(self,
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
        self.name = 'etdm'
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
        self._norm_commutator, self._norm_grad = 0., 0.
        self.error = 0

        # these are things we cannot initialize now
        self.dtype = None
        self.nkpts = None
        self.gd = None
        self.nbands = None

        # dimensionality of the problem.
        # this implementation rotates among all bands
        # values: matrices, keys: kpt number (and spins)
        self.a_mat_u = {}  # skew-hermitian matrix to be exponented
        self.g_mat_u = {}  # gradient matrix
        self.evecs = {}   # eigenvectors for i*a_mat_u
        self.evals = {}   # eigenvalues for i*a_mat_u
        self.ind_up = {}
        self.n_dim = {}
        self.alpha = 1.0  # step length
        self.phi_2i = [None, None]  # energy at last two iterations
        self.der_phi_2i = [None, None]  # energy gradient w.r.t. alpha
        self.hess = {}  # approximate hessian

        # for mom
        self.initial_occupation_numbers = None

        # in this attribute we sotre the object specific to each mode
        self.dm_helper = None

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

    def initialize(self, gd, dtype, nbands, nkpts, nao, using_blacs,
                   bd_comm_size, kpt_u):

        assert nbands == nao, \
            'Please, use: nbands=\'nao\''
        if not bd_comm_size == 1:
            raise BadParallelization(
                'Band parallelization is not supported')
        if using_blacs:
            raise BadParallelization(
                'ScaLapack parallelization is not supported')

        self.gd = gd
        self.dtype = dtype
        self.nbands = nbands
        self.nkpts = nkpts

        for kpt in kpt_u:
            u = self.kpointval(kpt)
            # dimensionality of the problem.
            # this implementation rotates among all bands
            self.n_dim[u] = self.nbands

        self.initialized = True

        # need reference orbitals
        self.dm_helper = None

    def initialize_dm_helper(self, wfs, ham, dens):

        self.dm_helper = DirectMinLCAO(
            wfs, ham, self.nkpts,
            diagonalizer=None,
            orthonormalization=self.orthonormalization,
            need_init_orbs=self.need_init_orbs
        )
        self.need_init_orbs = self.dm_helper.need_init_orbs
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
        self.set_variable_matrices(wfs.kpt_u)
        # if no empty state no need to optimize
        for k in self.ind_up:
            if not self.ind_up[k][0].size or not self.ind_up[k][1].size:
                self.n_dim[k] = 0
        # set reference orbitals
        self.dm_helper.set_reference_orbitals(wfs, self.n_dim)

    def set_variable_matrices(self, kpt_u):

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
        # store only A_2, that is this representation is sparser

        for kpt in kpt_u:
            n_occ = get_n_occ(kpt)
            u = self.kpointval(kpt)
            # M - one dimension of the A_BigMatrix
            M = self.n_dim[u]
            if self.representation == 'u-invar':
                i1, i2 = [], []
                for i in range(n_occ):
                    for j in range(n_occ, M):
                        i1.append(i)
                        i2.append(j)
                self.ind_up[u] = (np.asarray(i1), np.asarray(i2))
            else:
                if self.representation == 'full' and self.dtype == complex:
                    # Take indices of all upper triangular and diagonal
                    # elements of A_BigMatrix
                    self.ind_up[u] = np.triu_indices(self.n_dim[u])
                else:
                    self.ind_up[u] = np.triu_indices(self.n_dim[u], 1)
                    if self.representation == 'sparse':
                        # Delete indices of elements that correspond
                        # to 0 matrix in A_BigMatrix
                        zero_ind = -((M - n_occ) * (M - n_occ - 1)) // 2
                        if zero_ind == 0:
                            zero_ind = None
                        self.ind_up[u] = (self.ind_up[u][0][:zero_ind].copy(),
                                          self.ind_up[u][1][:zero_ind].copy())

            shape_of_arr = len(self.ind_up[u][0])

            self.a_mat_u[u] = np.zeros(shape=shape_of_arr, dtype=self.dtype)
            self.g_mat_u[u] = np.zeros(shape=shape_of_arr, dtype=self.dtype)
            self.evecs[u] = None
            self.evals[u] = None

        self.iters = 1

    def iterate(self, ham, wfs, dens):
        """
        One iteration of direct optimization
        for occupied states

        :param ham:
        :param wfs:
        :param dens:
        :return:
        """
        with wfs.timer('Direct Minimisation step'):
            self.update_ref_orbitals(wfs, ham, dens)
            with wfs.timer('Preconditioning:'):
                precond = self.get_preconditioning(wfs, self.use_prec)

            a_mat_u = self.a_mat_u
            n_dim = self.n_dim
            alpha = self.alpha
            phi_2i = self.phi_2i
            der_phi_2i = self.der_phi_2i
            c_ref = self.dm_helper.reference_orbitals

            if self.iters == 1:
                phi_2i[0], g_mat_u = \
                    self.get_energy_and_gradients(a_mat_u, n_dim, ham, wfs,
                                                  dens, c_ref)
            else:
                g_mat_u = self.g_mat_u

            with wfs.timer('Get Search Direction'):
                # calculate search direction according to chosen
                # optimization algorithm (e.g. L-BFGS)
                p_mat_u = self.searchdir_algo.update_data(wfs, a_mat_u,
                                                          g_mat_u, precond)

            # recalculate derivative with new search direction
            der_phi_2i[0] = 0.0
            for k in g_mat_u:
                der_phi_2i[0] += g_mat_u[k].conj() @ p_mat_u[k]
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
            self.error = 0.0
            self.e_sic = 0.0  # this is odd energy
            for kpt in wfs.kpt_u:
                k = self.kpointval(kpt)
                if n_dim[k] == 0:
                    g_mat_u[k] = np.zeros_like(a_mat_u[k])
                    continue
                g_mat_u[k], error = self.dm_helper.calc_grad(
                    wfs, ham, kpt, self.func, self.evecs[k], self.evals[k],
                    self.matrix_exp, self.representation, self.ind_up[k])

                self.error += error
            self.error = wfs.kd.comm.sum(self.error)
            self.e_sic = wfs.kd.comm.sum(self.e_sic)

        self.eg_count += 1

        if self.representation == 'full' and self.checkgraderror:
            self._norm_commutator = 0.0
            for kpt in wfs.kpt_u:
                u = self.kpointval(kpt)
                a = vec2skewmat(a_mat_u[u], self.n_dim[u],
                                self.ind_up[u], wfs.dtype)
                g = vec2skewmat(g_mat_u[u], self.n_dim[u],
                                self.ind_up[u], wfs.dtype)

                tmp = np.linalg.norm(g @ a - a @ g)
                if self._norm_commutator < tmp:
                    self._norm_commutator = tmp

                tmp = np.linalg.norm(g)
                if self._norm_grad < tmp:
                    self._norm_grad = tmp

        return e_total + self.e_sic, g_mat_u

    def update_ks_energy(self, ham, wfs, dens):
        """
        Update Kohn-Sham energy
        It assumes the temperature is zero K.
        """

        dens.update(wfs)
        ham.update(dens, wfs, False)

        return ham.get_energy(0.0, wfs, False)

    def evaluate_phi_and_der_phi(self, a_mat_u, p_mat_u, n_dim, alpha,
                                 ham, wfs, dens, c_ref,
                                 phi=None, g_mat_u=None):
        """
        phi = f(x_k + alpha_k*p_k)
        der_phi = \\grad f(x_k + alpha_k*p_k) \\cdot p_k
        :return:  phi, der_phi # floats
        """
        if phi is None or g_mat_u is None:
            x_mat_u = {k: a_mat_u[k] + alpha * p_mat_u[k] for k in a_mat_u}
            phi, g_mat_u = self.get_energy_and_gradients(x_mat_u, n_dim,
                                                         ham, wfs, dens, c_ref)

        der_phi = 0.0
        for k in p_mat_u:
            der_phi += g_mat_u[k].conj() @ p_mat_u[k]

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
            badgrad = self._norm_commutator > self._norm_grad / 3. and \
                self.checkgraderror
        else:
            badgrad = False
        counter = self.update_ref_orbs_counter
        if (self.iters % counter == 0 and self.iters > 1) or \
                (self.restart and self.iters > 1) or badgrad:
            self.iters = 1
            if self.update_ref_orbs_canonical or self.restart:
                self.get_canonical_representation(ham, wfs, dens)
            else:
                self.dm_helper.set_reference_orbitals(wfs, self.n_dim)
                for kpt in wfs.kpt_u:
                    u = self.kpointval(kpt)
                    self.a_mat_u[u] = np.zeros_like(self.a_mat_u[u])

            # Erase memory of search direction algorithm
            self.searchdir_algo.reset()

    def get_preconditioning(self, wfs, use_prec):

        if not use_prec:
            return None

        if self.searchdir_algo.name == 'l-bfgs-p':
            beta0 = self.searchdir_algo.beta_0
            gamma = 0.25
        else:
            beta0 = 1.0
            gamma = 0.0

        counter = self.update_precond_counter
        precond = {}
        for kpt in wfs.kpt_u:
            k = self.kpointval(kpt)
            w = kpt.weight / (3.0 - wfs.nspins)
            if self.iters % counter == 0 or self.iters == 1:
                self.hess[k] = self.get_hessian(kpt)
            hess = self.hess[k]
            precond[k] = np.zeros_like(hess)
            correction = w * gamma * beta0 ** (-1)
            if self.searchdir_algo.name != 'l-bfgs-p':
                correction = np.zeros_like(hess)
                zeros = abs(hess) < 1.0e-4
                correction[zeros] = 1.0
            precond[k] += 1. / ((1 - gamma) * hess.real + correction)
            if self.dtype == complex:
                precond[k] += 1.j / ((1 - gamma) * hess.imag + correction)

        return precond

    def get_hessian(self, kpt):
        """
        calculate approximate hessian:

        h_{lm, lm} = -2.0 * (eps_n[l] - eps_n[m]) * (f[l] - f[m])
        other elements are zero
        """

        f_n = kpt.f_n
        eps_n = kpt.eps_n
        u = self.kpointval(kpt)
        il1 = list(self.ind_up[u])

        hess = np.zeros(len(il1[0]), dtype=self.dtype)
        x = 0
        for n, m in zip(*il1):
            df = f_n[n] - f_n[m]
            hess[x] = -2.0 * (eps_n[n] - eps_n[m]) * df
            if abs(hess[x]) < 1.0e-10:
                hess[x] = 0.0
            if self.dtype == complex:
                hess[x] += 1.0j * hess[x]
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
                self.dm_helper.update_to_canonical_orbitals(
                    wfs, ham, kpt, self.update_ref_orbs_canonical,
                    self.restart)

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

            self.dm_helper.set_reference_orbitals(wfs, self.n_dim)
            for kpt in wfs.kpt_u:
                u = self.kpointval(kpt)
                self.a_mat_u[u] = np.zeros_like(self.a_mat_u[u])

    def reset(self):

        self.dm_helper = None
        self.error = np.inf
        self.initialized = False
        self.searchdir_algo.reset()

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
                    orbital_energies = self.dm_helper.orbital_energies(
                        wfs, ham, kpt)
                ind = np.argsort(orbital_energies)
                # check if it is necessary to sort
                x = np.max(abs(ind - np.arange(len(ind))))

                if x > 0:
                    # now sort wfs according to orbital energies
                    self.dm_helper.sort_orbitals(wfs, kpt, ind)
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
            occupied = kpt.f_n > 1.0e-10
            n_occ = len(kpt.f_n[occupied])
            if n_occ == 0.0:
                continue
            if np.min(kpt.f_n[:n_occ]) == 0:
                ind_occ = np.argwhere(occupied)
                ind_unocc = np.argwhere(~occupied)
                ind = np.vstack((ind_occ, ind_unocc))
                ind = np.squeeze(ind)
                self.dm_helper.sort_orbitals(wfs, kpt, ind)
                kpt.f_n = kpt.f_n[ind]
                kpt.eps_n = kpt.eps_n[ind]

                # OccupationsMOM.numbers needs to be updated after sorting
                self.update_mom_numbers(wfs, kpt)

                changedocc = True

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

                u_nn = self.get_exponential_matrix_kpt(wfs, kpt,
                                                       a_mat_u,
                                                       n_dim)

                self.dm_helper.appy_transformation_kpt(
                    wfs, u_nn.T, kpt, c_nm_ref[k], False, False)

                with wfs.timer('Calculate projections'):
                    self.dm_helper.update_projections(wfs, kpt)

    def get_exponential_matrix_kpt(self, wfs, kpt, a_mat_u, n_dim):
        """
        Get unitary matrix U as the exponential of a skew-Hermitian
        matrix a_mat_u (U = exp(a_mat_u))
        """

        k = self.kpointval(kpt)

        if self.gd.comm.rank == 0:
            if self.matrix_exp == 'egdecomp-u-invar' and \
                    self.representation == 'u-invar':
                n_occ = get_n_occ(kpt)
                n_v = n_dim[k] - n_occ
                a = a_mat_u[k].reshape(n_occ, n_v)
            else:
                a = vec2skewmat(a_mat_u[k], n_dim[k],
                                self.ind_up[k], self.dtype)

            if self.matrix_exp == 'pade-approx':
                # this function takes a lot of memory
                # for large matrices... what can we do?
                with wfs.timer('Pade Approximants'):
                    u_nn = np.ascontiguousarray(expm(a))
            elif self.matrix_exp == 'egdecomp':
                # this method is based on diagonalisation
                with wfs.timer('Eigendecomposition'):
                    u_nn, evecs, evals = expm_ed(a, evalevec=True)
            elif self.matrix_exp == 'egdecomp-u-invar':
                with wfs.timer('Eigendecomposition'):
                    u_nn = expm_ed_unit_inv(a, oo_vo_blockonly=False)

        with wfs.timer('Broadcast u_nn'):
            if self.gd.comm.rank != 0:
                u_nn = np.zeros(shape=(n_dim[k], n_dim[k]),
                                dtype=wfs.dtype)
            self.gd.comm.broadcast(u_nn, 0)

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

        return u_nn

    def check_assertions(self, wfs, dens):

        assert dens.mixer.driver.basemixerclass.name == 'no-mixing', \
            'Please, use: mixer={\'backend\': \'no-mixing\'}'
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
        add random noise to orbitals but keep them still orthonormal
        """
        nst = self.nbands
        wt = kpt.weight * 0.01
        arand = wt * random_a((nst, nst), wfs.dtype)
        arand = arand - arand.T.conj()
        wfs.gd.comm.broadcast(arand, 0)
        self.dm_helper.appy_transformation_kpt(wfs, expm(arand), kpt)

    def kpointval(self, kpt):
        return self.nkpts * kpt.s + kpt.q

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, e):
        self._error = e


def get_n_occ(kpt):
    """
    get number of occupied orbitals
    """
    return len(kpt.f_n) - np.searchsorted(kpt.f_n[::-1], 1e-10)


def random_a(shape, dtype):

    a = np.random.random_sample(shape)
    if dtype == complex:
        a = a.astype(complex)
        a += 1.0j * np.random.random_sample(shape)

    return a


def vec2skewmat(a_vec, dim, ind_up, dtype):

    a = np.zeros(shape=(dim, dim), dtype=dtype)
    a[ind_up] = a_vec
    a -= a.T.conj()
    np.fill_diagonal(a, a.diagonal() * 0.5)
    return a
