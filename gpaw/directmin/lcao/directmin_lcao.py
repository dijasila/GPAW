"""
A class for finding optimal
orbitals of the KS-DFT or PZ-SIC
functionals using exponential transformation
direct minimization

arXiv:2101.12597 [physics.comp-ph]
Comput. Phys. Commun. 267, 108047 (2021).
https://doi.org/10.1016/j.cpc.2021.108047
"""


from ase.units import Hartree
import numpy as np
from gpaw.utilities.blas import mmm
from gpaw.directmin.lcao.tools import d_matrix, expm_ed, expm_ed_unit_inv
from gpaw.lcao.eigensolver import DirectLCAO
from scipy.linalg import expm
from gpaw.utilities.tools import tri2full
from gpaw.directmin.lcao import search_direction, line_search_algorithm
from gpaw.xc import xc_string_to_dict
from ase.utils import basestring
from gpaw.directmin.functional.lcao import get_functional


class DirectMinLCAO(DirectLCAO):

    def __init__(self, diagonalizer=None,
                 searchdir_algo='LBFGS_P',
                 linesearch_algo='SwcAwc',
                 update_ref_orbs_counter=20,
                 update_ref_orbs_canonical=False,
                 update_precond_counter=1000,
                 use_prec=True, matrix_exp='pade_approx',
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
        This class describes the exponential transformation
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
        'pade_approx', 'egdecomp', 'egdecomp2' (used with u_invar represnt.),
        :param representation: the way A are stored,
        'sparse', 'full', 'u_invar',
        :param functional: KS or PZ-SIC functionals
        :param orthonormalization:
        :param randomizeorbitals: if need a noise in initial guess
        :param checkgraderror: check error in estimation of gradient
        :param localizationtype: Foster-Boys, Pipek-Mezey, Edm.-Rudenb.
        :param need_localization: use localized orbitals as initial guess
        :param need_init_orbs: if false then use coef. stored in kpt.C_nM
        """

        super(DirectMinLCAO, self).__init__(diagonalizer)

        self.sda = searchdir_algo
        self.lsa = linesearch_algo
        self.localizationtype = localizationtype
        self.eg_count = 0
        self.update_ref_orbs_counter = update_ref_orbs_counter
        self.update_ref_orbs_canonical = update_ref_orbs_canonical
        self.update_precond_counter = update_precond_counter
        self.use_prec = use_prec
        self.matrix_exp = matrix_exp
        self.representation = representation
        self.orthonormalization = orthonormalization
        self.iters = 0
        self.restart = False
        self.name = 'direct-min-lcao'
        self.localizationtype = localizationtype
        self.need_localization = need_localization
        self.need_init_orbs = need_init_orbs

        self.a_mat_u = None  # skew-hermitian matrix to be exponented
        self.g_mat_u = None  # gradient matrix
        self.c_nm_ref = None  # reference orbitals to be rotated

        self.functional = functional
        self.randomizeorbitals = randomizeorbitals

        self.initialized = False

        if isinstance(self.functional, basestring):
            self.functional = xc_string_to_dict(self.functional)
        if isinstance(self.sda, basestring):
            self.sda = xc_string_to_dict(self.sda)
        if isinstance(self.lsa, basestring):
            self.lsa = xc_string_to_dict(self.lsa)
            self.lsa['method'] = self.sda['name']

        if isinstance(self.representation, basestring):
            assert self.representation in ['sparse', 'u_invar', 'full'], \
                'Value Error'
            self.representation = \
                xc_string_to_dict(self.representation)

        if isinstance(self.orthonormalization, basestring):
            assert self.orthonormalization in [
                'gramschmidt', 'loewdin', 'diag'], \
                'Value Error'
            self.orthonormalization = \
                xc_string_to_dict(self.orthonormalization)

        if self.sda['name'] == 'LBFGS_P' and not self.use_prec:
            raise ValueError('Use LBFGS_P with use_prec=True')

        if matrix_exp == 'egdecomp2':
            assert self.representation['name'] == 'u_invar', \
                'Use u_invar representation with egdecomp2'

        self.checkgraderror = checkgraderror
        self._normcomm, self._normg = 0., 0.

    def __repr__(self):

        sds = {'SD': 'Steepest Descent',
               'FRcg': 'Fletcher-Reeves conj. grad. method',
               'HZcg': 'Hager-Zhang conj. grad. method',
               'QuickMin': 'Molecular-dynamics based algorithm',
               'LBFGS': 'LBFGS algorithm',
               'LBFGS_P': 'LBFGS algorithm with preconditioning',
               'LBFGS_P2': 'LBFGS algorithm with preconditioning',
               'LSR1P': 'Limited-memory SR1P algorithm'}

        lss = {'UnitStep': 'step size equals one',
               'Parabola': 'Parabolic line search',
               'SwcAwc': 'Inexact line search based '
                         'on cubic interpolation,\n'
                         '                    strong'
                         ' and approximate Wolfe conditions'}

        repr_string = 'Direct minimisation using exponential ' \
                      'transformation.\n'
        repr_string += '       ' \
                       'Search ' \
                       'direction: {}\n'.format(sds[self.sda['name']])
        repr_string += '       ' \
                       'Line ' \
                       'search: {}\n'.format(lss[self.lsa['name']])
        repr_string += '       ' \
                       'Preconditioning: {}\n'.format(self.use_prec)
        repr_string += '       ' \
                       'WARNING: do not use it for metals as ' \
                       'occupation numbers are\n' \
                       '                ' \
                       'not found variationally\n'

        return repr_string

    def init_me(self, wfs, ham, dens, log):
        # need to initialize c_nm, eps, f_n and so on.
        self.initialize_orbitals(wfs, ham, dens, log)
        wfs.calculate_occupation_numbers(dens.fixed)
        occ_name = getattr(wfs.occupations, "name", None)
        if occ_name == 'mom':
            # We need to initialize the MOM reference orbitals
            # before sorting coefficients and occupation numbers
            wfs.occupations.initialize_reference_orbitals()
            self.sort_wavefunctions_mom(wfs)
            wfs.calculate_occupation_numbers(dens.fixed)
        self.localize_wfs(wfs, dens, ham, log)
        if occ_name == 'mom':
            # Need to reinitialize the MOM reference orbitals
            # before sorting coefficients and occupation numbers
            wfs.occupations.initialize_reference_orbitals()
            self.sort_wavefunctions_mom(wfs)
            wfs.calculate_occupation_numbers(dens.fixed)
        self.update_ks_energy(ham, wfs, dens)
        self.initialize_2(wfs, dens, ham)

    def initialize_2(self, wfs, dens, ham):

        self.dtype = wfs.dtype
        self.n_kps = wfs.kd.nibzkpts

        # dimensionality of the problem.
        # this implementation rotates among all bands
        self.n_dim = {}
        for kpt in wfs.kpt_u:
            u = kpt.s * self.n_kps + kpt.q
            self.n_dim[u] = wfs.bd.nbands

        # values: matrices, keys: kpt number (and spins)
        self.a_mat_u = {}  # skew-hermitian matrix to be exponented
        self.g_mat_u = {}  # gradient matrix
        self.c_nm_ref = {}  # reference orbitals to be rotated

        self.evecs = {}   # eigenvectors for i*a_mat_u
        self.evals = {}   # eigenvalues for i*a_mat_u
        self.ind_up = {}

        if self.representation['name'] in ['sparse', 'u_invar']:
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
            # 'u_invar' corresponds to the case when we want to
            # store only A_2, that is this representaion is sparser

            M = wfs.bd.nbands  # M - one dimension of the A_BigMatrix
            if self.representation['name'] == 'sparse':
                # let's take all upper triangular indices
                # of A_BigMatrix and then delete indices from ind_up
                # which correspond to 0 matrix in in A_BigMatrix.
                ind_up = np.triu_indices(M, 1)
                for kpt in wfs.kpt_u:
                    n_occ = get_n_occ(kpt)
                    u = self.n_kps * kpt.s + kpt.q
                    zero_ind = ((M - n_occ) * (M - n_occ - 1)) // 2
                    # TODO: This breaks if there is only one unoccupied band
                    self.ind_up[u] = (ind_up[0][:-zero_ind].copy(),
                                      ind_up[1][:-zero_ind].copy())
                del ind_up
            else:
                # take indices of A_2 only
                for kpt in wfs.kpt_u:
                    n_occ = get_n_occ(kpt)
                    u = self.n_kps * kpt.s + kpt.q
                    i1, i2 = [], []
                    for i in range(n_occ):
                        for j in range(n_occ, M):
                            i1.append(i)
                            i2.append(j)
                    self.ind_up[u] = (np.asarray(i1), np.asarray(i2))

        for kpt in wfs.kpt_u:
            u = self.n_kps * kpt.s + kpt.q
            if self.representation['name'] in ['sparse', 'u_invar']:
                shape_of_arr = len(self.ind_up[u][0])
            else:
                self.ind_up[u] = None
                shape_of_arr = (self.n_dim[u], self.n_dim[u])

            if self.randomizeorbitals:
                nst = kpt.C_nM.shape[0]
                wt = kpt.weight * 0.01
                arand = wt * (np.random.rand(nst, nst)).astype(wfs.dtype)
                if wfs.dtype is complex:
                    arand += 1.j * np.random.rand(nst, nst) * wt
                arand = arand - arand.T.conj()
                kpt.C_nM[:] = expm(arand) @ kpt.C_nM[:]
                wfs.atomic_correction.calculate_projections(wfs, kpt)
            self.a_mat_u[u] = np.zeros(shape=shape_of_arr,
                                       dtype=self.dtype)
            self.g_mat_u[u] = np.zeros(shape=shape_of_arr,
                                       dtype=self.dtype)
            # use initial KS orbitals, but can be others
            self.c_nm_ref[u] = np.copy(kpt.C_nM[:self.n_dim[u]])
            self.evecs[u] = None
            self.evals[u] = None

        self.randomizeorbitals = False
        self.alpha = 1.0  # step length
        self.phi_2i = [None, None]  # energy at last two iterations
        self.der_phi_2i = [None, None]  # energy gradient w.r.t. alpha
        self.precond = None

        self.iters = 1
        self.nvalence = wfs.nvalence
        self.nbands = wfs.bd.nbands
        self.kd_comm = wfs.kd.comm
        self.hess = {}  # hessian for LBFGS-P
        self.precond = {}  # precondiner for other methods

        # choose search direction and line search algorithm
        if isinstance(self.sda, (basestring, dict)):
            self.search_direction = search_direction(self.sda, wfs)
        else:
            raise Exception('Check Search Direction Parameters')

        if isinstance(self.lsa, (basestring, dict)):
            self.line_search = \
                line_search_algorithm(self.lsa,
                                      self.evaluate_phi_and_der_phi)
        else:
            raise Exception('Check Search Direction Parameters')

        # odd corrections
        if isinstance(self.functional, (basestring, dict)):
            self.func = \
                get_functional(self.functional, wfs, dens, ham)
        elif self.func is None:
            pass
        else:
            raise Exception('Check ODD Parameters')
        self.initialized = True

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
        self.check_assertions(wfs, dens)

        wfs.timer.start('Direct Minimisation step')
        self.update_ref_orbitals(wfs, ham, dens)
        wfs.timer.start('Preconditioning:')
        precond = self.update_preconditioning(wfs, self.use_prec)
        wfs.timer.stop('Preconditioning:')

        if str(self.search_direction) == 'LBFGS_P2':
            for kpt in wfs.kpt_u:
                u = kpt.s * self.n_kps + kpt.q
                self.c_nm_ref[u] = kpt.C_nM.copy()

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

        wfs.timer.start('Get Search Direction')
        p_mat_u = self.get_search_direction(a_mat_u, g_mat_u, precond,
                                            wfs)
        wfs.timer.stop('Get Search Direction')

        # recalculate derivative with new search direction
        der_phi_2i[0] = 0.0
        for k in g_mat_u.keys():
            if self.representation['name'] in ['sparse', 'u_invar']:
                der_phi_2i[0] += np.dot(g_mat_u[k].conj(),
                                        p_mat_u[k]).real
            else:
                il1 = get_indices(g_mat_u[k].shape[0], self.dtype)
                der_phi_2i[0] += np.dot(g_mat_u[k][il1].conj(),
                                        p_mat_u[k][il1]).real
                # der_phi_c += dotc(g[k][il1], p[k][il1]).real
        der_phi_2i[0] = wfs.kd.comm.sum(der_phi_2i[0])

        if str(self.search_direction) == 'LBFGS_P2':
            for kpt in wfs.kpt_u:
                u = kpt.s * self.n_kps + kpt.q
                a_mat_u[u] = np.zeros_like(a_mat_u[u])

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
            wfs.timer.start('Broadcast gradients')
            alpha_phi_der_phi = np.array([alpha, phi_2i[0],
                                          der_phi_2i[0]])
            wfs.gd.comm.broadcast(alpha_phi_der_phi, 0)
            alpha = alpha_phi_der_phi[0]
            phi_2i[0] = alpha_phi_der_phi[1]
            der_phi_2i[0] = alpha_phi_der_phi[2]
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                wfs.gd.comm.broadcast(g_mat_u[k], 0)
            wfs.timer.stop('Broadcast gradients')

        # calculate new matrices for optimal step length
        for k in a_mat_u.keys():
            if str(self.search_direction) == 'LBFGS_P2':
                a_mat_u[k] = alpha * p_mat_u[k]
            else:
                a_mat_u[k] += alpha * p_mat_u[k]
        self.alpha = alpha
        self.g_mat_u = g_mat_u
        self.iters += 1

        # and 'shift' phi, der_phi for the next iteration
        phi_2i[1], der_phi_2i[1] = phi_2i[0], der_phi_2i[0]
        phi_2i[0], der_phi_2i[0] = phi_alpha, der_phi_alpha,

        wfs.timer.stop('Direct Minimisation step')

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

        wfs.timer.start('Calculate gradients')
        g_mat_u = {}
        self._error = 0.0
        self.e_sic = 0.0  # this is odd energy
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            if n_dim[k] == 0:
                g_mat_u[k] = np.zeros_like(a_mat_u[k])
                continue
            h_mm = self.calculate_hamiltonian_matrix(ham, wfs, kpt)
            # make matrix hermitian
            tri2full(h_mm)
            g_mat_u[k], error = self.func.get_gradients(
                h_mm, kpt.C_nM, kpt.f_n, self.evecs[k], self.evals[k],
                kpt, wfs, wfs.timer, self.matrix_exp,
                self.representation['name'], self.ind_up[k])

            self._error += error
        self._error = self.kd_comm.sum(self._error)
        self.e_sic = self.kd_comm.sum(self.e_sic)
        wfs.timer.stop('Calculate gradients')

        self.eg_count += 1

        if self.representation['name'] == 'full' and self.checkgraderror:
            norm = 0.0
            for kpt in wfs.kpt_u:
                u = kpt.s * self.n_kps + kpt.q
                normt = np.linalg.norm(
                    g_mat_u[u] @ self.a_mat_u[u] -
                    self.a_mat_u[u] @ g_mat_u[u])
                if norm < normt:
                    norm = normt
            norm2 = 0.0
            for kpt in wfs.kpt_u:
                u = kpt.s * self.n_kps + kpt.q
                normt = np.linalg.norm(g_mat_u[u])
                if norm2 < normt:
                    norm2 = normt

            self._normcomm = norm
            self._normg = norm2

        return e_total + self.e_sic, g_mat_u

    def update_ks_energy(self, ham, wfs, dens):
        """
        Update Kohn-Shame energy
        It assumes the temperature is zero K.


        :param ham:
        :param wfs:
        :param dens:
        :return:
        """

        # wfs.timer.start('Update Kohn-Sham energy')
        dens.update(wfs)
        ham.update(dens, wfs, False)
        # wfs.timer.stop('Update Kohn-Sham energy')

        return ham.get_energy(0.0, wfs, False)

    def get_gradients(self, h_mm, c_nm, f_n, evec, evals, kpt, timer):

        """
        calculate derivatives of the energy
        with respect to skew-hermitian
        matrix elements

        :param h_mm:
        :param c_nm:
        :param f_n:
        :param evec:
        :param evals:
        :param kpt:
        :param timer:
        :return:
        """

        timer.start('Construct Gradient Matrix')
        hc_mn = np.zeros(shape=(c_nm.shape[1], c_nm.shape[0]),
                         dtype=self.dtype)
        mmm(1.0, h_mm.conj(), 'N', c_nm, 'T', 0.0, hc_mn)
        if c_nm.shape[0] != c_nm.shape[1]:
            h_mm = np.zeros(shape=(c_nm.shape[0], c_nm.shape[0]),
                            dtype=self.dtype)
        mmm(1.0, c_nm.conj(), 'N', hc_mn, 'N', 0.0, h_mm)
        timer.stop('Construct Gradient Matrix')

        # let's also calculate residual here.
        # it's extra calculation though, maybe it's better to use
        # norm of grad as convergence criteria..
        timer.start('Residual')
        n_occ = 0
        nbands = len(f_n)
        while n_occ < nbands and f_n[n_occ] > 1e-10:
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
            norm.append(np.dot(hc_mn[:, i].conj(),
                               hc_mn[:, i]).real * kpt.f_n[i])
            # needs to be contig. to use this:
            # x = np.ascontiguousarray(hc_mn[:,i])
            # norm.append(dotc(x, x).real * kpt.f_n[i])

        error = sum(norm) * Hartree ** 2 / self.nvalence
        del rhs, rhs2, hc_mn, norm
        timer.stop('Residual')

        # continue with gradients
        timer.start('Construct Gradient Matrix')
        h_mm = f_n * h_mm - f_n[:, np.newaxis] * h_mm
        if self.matrix_exp in ['pade_approx', 'egdecomp2']:
            # timer.start('Frechet derivative')
            # frechet derivative, unfortunately it calculates unitary
            # matrix which we already calculated before. Could it be used?
            # it also requires a lot of memory so don't use it now
            # u, grad = expm_frechet(a_mat, h_mm,
            #                        compute_expm=True,
            #                        check_finite=False)
            # grad = grad @ u.T.conj()
            # timer.stop('Frechet derivative')
            grad = np.ascontiguousarray(h_mm)
        elif self.matrix_exp == 'egdecomp':
            timer.start('Use Eigendecomposition')
            grad = np.dot(evec.T.conj(), np.dot(h_mm, evec))
            grad = grad * d_matrix(evals)
            grad = np.dot(evec, np.dot(grad, evec.T.conj()))
            timer.stop('Use Eigendecomposition')
            for i in range(grad.shape[0]):
                grad[i][i] *= 0.5
        else:
            raise ValueError('Check the keyword '
                             'for matrix_exp. \n'
                             'Must be '
                             '\'pade_approx\' or '
                             '\'egdecomp\'')

        if self.dtype == float:
            grad = grad.real
        if self.representation['name'] in ['sparse', 'u_invar']:
            u = self.n_kps * kpt.s + kpt.q
            grad = grad[self.ind_up[u]]
        timer.stop('Construct Gradient Matrix')

        return 2.0 * grad, error

    def get_search_direction(self, a_mat_u, g_mat_u, precond, wfs):
        """
        calculate search direction according to chosen
        optimization algorithm (LBFGS for example)

        :param a_mat_u:
        :param g_mat_u:
        :param precond:
        :param wfs:
        :return:
        """

        if self.representation['name'] in ['sparse', 'u_invar']:
            p_mat_u = self.search_direction.update_data(wfs, a_mat_u,
                                                        g_mat_u,
                                                        precond)
        else:
            g_vec = {}
            a_vec = {}

            for k in a_mat_u.keys():
                il1 = get_indices(a_mat_u[k].shape[0], self.dtype)
                a_vec[k] = a_mat_u[k][il1]
                g_vec[k] = g_mat_u[k][il1]

            p_vec = self.search_direction.update_data(wfs, a_vec,
                                                      g_vec, precond)
            del a_vec, g_vec

            p_mat_u = {}
            for k in p_vec.keys():
                p_mat_u[k] = np.zeros_like(a_mat_u[k])
                il1 = get_indices(p_mat_u[k].shape[0], self.dtype)
                p_mat_u[k][il1] = p_vec[k]
                # make it skew-hermitian
                il1 = np.tril_indices(p_mat_u[k].shape[0], -1)
                p_mat_u[k][(il1[1], il1[0])] = -p_mat_u[k][il1].conj()

            del p_vec

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
                       for k in a_mat_u.keys()}
            phi, g_mat_u = \
                self.get_energy_and_gradients(x_mat_u, n_dim,
                                              ham, wfs, dens,
                                              c_ref
                                              )
            del x_mat_u
        else:
            pass

        der_phi = 0.0
        if self.representation['name'] in ['sparse', 'u_invar']:
            for k in p_mat_u.keys():
                der_phi += np.dot(g_mat_u[k].conj(),
                                  p_mat_u[k]).real
        else:
            for k in p_mat_u.keys():

                il1 = get_indices(p_mat_u[k].shape[0], self.dtype)

                der_phi += np.dot(g_mat_u[k][il1].conj(),
                                  p_mat_u[k][il1]).real
                # der_phi += dotc(g_mat_u[k][il1],
                #                 p_mat_u[k][il1]).real

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

        if str(self.search_direction) == 'LBFGS_P2':
            return 0

        if self.representation['name'] == 'full':
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
                    u = kpt.s * self.n_kps + kpt.q
                    # self.sort_wavefunctions(ham, wfs, kpt)
                    self.c_nm_ref[u] = kpt.C_nM.copy()
                    self.a_mat_u[u] = np.zeros_like(self.a_mat_u[u])
                    # self.sort_wavefunctions(ham, wfs, kpt)

            # choose search direction and line search algorithm
            # as you need to restart it
            if isinstance(self.sda, (basestring, dict)):
                self.search_direction = search_direction(self.sda, wfs)
            else:
                raise Exception('Check Search Direction Parameters')

            if isinstance(self.lsa, (basestring, dict)):
                self.line_search = \
                    line_search_algorithm(self.lsa,
                                          self.evaluate_phi_and_der_phi)
            else:
                raise Exception('Check Search Direction Parameters')

    def update_preconditioning(self, wfs, use_prec):
        """
        update preconditioning

        :param wfs:
        :param use_prec:
        :return:
        """

        counter = self.update_precond_counter
        if use_prec:
            if self.sda['name'] != 'LBFGS_P':
                if self.iters % counter == 0 or self.iters == 1:
                    for kpt in wfs.kpt_u:
                        k = self.n_kps * kpt.s + kpt.q
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
                    k = self.n_kps * kpt.s + kpt.q
                    w = kpt.weight / (3.0 - wfs.nspins)
                    if self.iters % counter == 0 or self.iters == 1:
                        self.hess[k] = self.get_hessian(kpt)
                    hess = self.hess[k]
                    beta0 = self.search_direction.beta_0
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
        if self.representation['name'] in ['sparse', 'u_invar']:
            u = self.n_kps * kpt.s + kpt.q
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

    def calculate_residual(self, kpt, H_MM, S_MM, wfs):
        return np.inf

    def get_canonical_representation(self, ham, wfs, dens,
                                     update_eigenvalues=True,
                                     update_wfs=False):
        """
        choose canonical orbitals which diagonalise
        lagrange matrix. it's probably necessary
        to do subspace rotation with equally
        occupied states.
        In this case, the total energy remains the same,
        as it's unitary invariant within equally occupied subspaces.

        :param ham:
        :param wfs:
        :param update_eigenvalues:
        :param update_wfs:
        :return:
        """

        wfs.timer.start('Get canonical representation')

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
                        raise SystemExit('Bug is here!')

            with wfs.timer('Calculate projections'):
                wfs.atomic_correction.calculate_projections(wfs, kpt)
        self._e_entropy = wfs.calculate_occupation_numbers(dens.fixed)
        occ_name = getattr(wfs.occupations, "name", None)
        if occ_name == 'mom':
            self.sort_wavefunctions_mom(wfs)
            self._e_entropy = wfs.calculate_occupation_numbers(
                dens.fixed)

        for kpt in wfs.kpt_u:
            u = kpt.s * self.n_kps + kpt.q
            self.c_nm_ref[u] = kpt.C_nM.copy()
            self.a_mat_u[u] = np.zeros_like(self.a_mat_u[u])

        wfs.timer.stop('Get canonical representation')

    def reset(self):
        super(DirectMinLCAO, self).reset()
        self._error = np.inf
        self.initialized = False

    def sort_wavefunctions(self, ham, wfs, kpt):
        """
        this function sorts wavefunctions according
        it's orbitals energies, not eigenvalues.
        :return:
        """
        wfs.timer.start('Sort WFS')
        h_mm = self.calculate_hamiltonian_matrix(ham, wfs, kpt)
        tri2full(h_mm)
        hc_mn = np.zeros(shape=(kpt.C_nM.shape[1], kpt.C_nM.shape[0]),
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
            kpt.eps_n[:] = np.sort(h_mm.diagonal().real.copy())
        wfs.timer.stop('Sort WFS')

        return

    def sort_wavefunctions_mom(self, wfs):
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

        return changedocc

    def todict(self):
        return {'name': self.name,
                'searchdir_algo': self.sda,
                'linesearch_algo': self.lsa,
                'localizationtype': self.localizationtype,
                'update_ref_orbs_counter': self.update_ref_orbs_counter,
                'update_precond_counter': self.update_precond_counter,
                'use_prec': self.use_prec,
                'matrix_exp': self.matrix_exp,
                'representation': self.representation,
                'functional': self.functional,
                'orthonormalization': self.orthonormalization
                }

    def get_numerical_gradients(self, n_dim, ham, wfs, dens,
                                c_nm_ref, eps=1.0e-7):

        """
           calculate gradient with respect to skew-hermitian
           matrix using finite differences with random noise
           this is just to test the implementation of anal. gradient

        :param n_dim:
        :param ham:
        :param wfs:
        :param dens:
        :param c_nm_ref:
        :param eps:
        :return:
        """

        assert not self.representation['name'] in ['sparse', 'u_invar']
        a_m = {}
        g_n = {}
        if self.matrix_exp == 'pade_approx':
            c_nm_ref = {}
        for kpt in wfs.kpt_u:
            u = self.n_kps * kpt.s + kpt.q
            a = np.random.random_sample(self.a_mat_u[u].shape) + \
                1.0j * np.random.random_sample(self.a_mat_u[u].shape)
            a = a - a.T.conj()
            u_nn = expm(a)
            g_n[u] = np.zeros_like(self.a_mat_u[u])

            if self.matrix_exp == 'pade_approx':
                a_m[u] = np.zeros_like(self.a_mat_u[u])
                c_nm_ref[u] = np.dot(u_nn.T, kpt.C_nM[:u_nn.shape[0]])
            elif self.matrix_exp == 'egdecomp':
                a_m[u] = a

        g_a = self.get_energy_and_gradients(a_m, n_dim, ham, wfs,
                                            dens, c_nm_ref)[1]

        h = [eps, -eps]
        coeif = [1.0, -1.0]

        if self.dtype == complex:
            range_z = 2
            complex_gr = [1.0, 1.0j]
        else:
            range_z = 1
            complex_gr = [1.0]

        for kpt in wfs.kpt_u:
            u = self.n_kps * kpt.s + kpt.q
            dim = a_m[u].shape[0]
            for z in range(range_z):
                for i in range(dim):
                    for j in range(dim):
                        print(u, z, i, j)
                        a = a_m[u][i][j]
                        g = 0.0
                        for l in range(2):
                            if z == 0:
                                if i != j:
                                    a_m[u][i][j] = a + h[l]
                                    a_m[u][j][i] = -np.conjugate(a + h[l])
                            else:
                                a_m[u][i][j] = a + 1.0j * h[l]
                                if i != j:
                                    a_m[u][j][i] = \
                                        -np.conjugate(a + 1.0j * h[l])

                            E = self.get_energy_and_gradients(
                                a_m, n_dim, ham, wfs, dens,
                                c_nm_ref)[0]

                            g += E * coeif[l]

                        g *= 1.0 / (2.0 * eps)

                        g_n[u][i][j] += g * complex_gr[z]
                        a_m[u][i][j] = a
                        if i != j:
                            a_m[u][j][i] = -np.conjugate(a)

        return g_a, g_n

    def get_numerical_hessian(self, n_dim, ham, wfs, dens,
                              c_nm_ref, eps=1.0e-5):
        """
        calculate hessian with respect to skew-hermitian elements
        using finite differences

        :param n_dim:
        :param ham:
        :param wfs:
        :param dens:
        :param c_nm_ref:
        :param eps:
        :return:
        """

        occ_name = getattr(wfs.occupations, "name", None)
        if occ_name == 'mom':
            self.sort_wavefunctions_mom(wfs)
            for kpt in wfs.kpt_u:
                u = self.n_kps * kpt.s + kpt.q
                c_nm_ref[u] = kpt.C_nM.copy()

        assert self.matrix_exp == 'egdecomp'

        h = [eps, -eps]
        coeif = [1.0, -1.0]

        if self.dtype == complex:
            range_z = 2
            complex_gr = [1.0, 1.0j]
        else:
            range_z = 1
            complex_gr = [1.0]

        a_m = {}
        for kpt in wfs.kpt_u:
            u = self.n_kps * kpt.s + kpt.q
            a_m[u] = np.zeros_like(self.a_mat_u[u])

        hess_a = {}
        hess_n = {}
        for kpt in wfs.kpt_u:
            u = self.n_kps * kpt.s + kpt.q
            hess_a[u] = self.get_hessian(kpt)

            m2 = self.a_mat_u[u].shape[0]
            hess_n[u] = np.zeros((m2, m2))

            for z in range(range_z):
                r = 0
                for i in range(m2):
                    a = a_m[u][i]
                    hess = np.zeros(m2)
                    for l in range(2):
                        # Does this work when self.dtype == complex?
                        if z == 0:
                            a_m[u][i] = a + h[l]
                        else:
                            a_m[u][i] = a + 1.0j * h[l]

                        g = self.get_energy_and_gradients(
                            a_m, n_dim, ham, wfs, dens, c_nm_ref)[1]

                        hess += g[u] * coeif[l]

                    hess *= 1.0 / (2.0 * eps)

                    hess_n[u][r] += hess * complex_gr[z]
                    a_m[u][i] = a

                    r += 1

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

        wfs.timer.start('Unitary rotation')
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            if n_dim[k] == 0:
                continue

            if self.gd.comm.rank == 0:
                if self.representation['name'] in ['sparse', 'u_invar']:
                    if self.matrix_exp == 'egdecomp2' and \
                            self.representation['name'] == 'u_invar':
                        n_occ = get_n_occ(kpt)
                        n_v = self.nbands - n_occ
                        a = a_mat_u[k].reshape(n_occ, n_v)
                    else:
                        a = np.zeros(shape=(n_dim[k], n_dim[k]),
                                     dtype=self.dtype)
                        a[self.ind_up[k]] = a_mat_u[k]
                        a += -a.T.conj()
                else:
                    a = a_mat_u[k]

                if self.matrix_exp == 'pade_approx':
                    # this function takes a lot of memory
                    # for large matrices... what can we do?
                    wfs.timer.start('Pade Approximants')
                    u_nn = expm(a)
                    wfs.timer.stop('Pade Approximants')
                elif self.matrix_exp == 'egdecomp':
                    # this method is based on diagonalisation
                    wfs.timer.start('Eigendecomposition')
                    u_nn, evecs, evals = \
                        expm_ed(a, evalevec=True)
                    wfs.timer.stop('Eigendecomposition')
                elif self.matrix_exp == 'egdecomp2':
                    assert self.representation['name'] == 'u_invar'
                    wfs.timer.start('Eigendecomposition')
                    u_nn = expm_ed_unit_inv(a, oo_vo_blockonly=False)
                    wfs.timer.stop('Eigendecomposition')

                else:
                    raise ValueError('Check the keyword '
                                     'for matrix_exp. \n'
                                     'Must be '
                                     '\'pade_approx\' or '
                                     '\'egdecomp\'')

                dimens1 = u_nn.shape[0]
                dimens2 = u_nn.shape[1]
                kpt.C_nM[:dimens2] = np.dot(
                    u_nn.T, c_nm_ref[k][:dimens1])

                del u_nn
                del a

            wfs.timer.start('Broadcast coefficients')
            self.gd.comm.broadcast(kpt.C_nM, 0)
            wfs.timer.stop('Broadcast coefficients')

            if self.matrix_exp == 'egdecomp':
                wfs.timer.start('Broadcast evecs and evals')
                if self.gd.comm.rank != 0:
                    evecs = np.zeros(shape=(n_dim[k], n_dim[k]),
                                     dtype=complex)
                    evals = np.zeros(shape=n_dim[k],
                                     dtype=float)

                self.gd.comm.broadcast(evecs, 0)
                self.gd.comm.broadcast(evals, 0)
                self.evecs[k], self.evals[k] = evecs, evals
                wfs.timer.stop('Broadcast evecs and evals')

            with wfs.timer('Calculate projections'):
                wfs.atomic_correction.calculate_projections(wfs, kpt)

        wfs.timer.stop('Unitary rotation')

    def initialize_orbitals(self, wfs, ham, dens, log):
        """
        if it is the first use of the scf then initialize
        coefficient matrix using eigensolver
        and then localise orbitals

        :param wfs:
        :param ham:
        :param dens:
        :param log:
        :return:
        """

        # if it is the first use of the scf then initialize
        # coefficient matrix using eigensolver
        orthname = self.orthonormalization['name']
        need_canon_coef = \
            (not wfs.coefficients_read_from_file and self.need_init_orbs)
        if need_canon_coef or orthname == 'diag':
            super(DirectMinLCAO, self).iterate(ham, wfs)
        else:
            wfs.orthonormalize(type=orthname)
        wfs.coefficients_read_from_file = False
        self.need_init_orbs = False

    def localize_wfs(self, wfs, dens, ham, log):
        """
        initial orbitals can be localised using Pipek-Mezey,
        Foster-Boys or Edmiston-Ruedenberg functions.

        :param wfs:
        :param dens:
        :param ham:
        :param log:
        :return:
        """
        pass

        # if not self.need_localization:
        #     return
        # localize_orbitals(wfs, dens, ham, log, self.localizationtype)
        # self.need_localization = False

    def check_assertions(self, wfs, dens):

        assert dens.mixer.driver.basemixerclass.name == 'no-mixing', \
            'Please, use: mixer={\'backend\': \'no-mixing\'}'
        assert wfs.bd.nbands == wfs.basis_functions.Mmax, \
            'Please, use: nbands=\'nao\''
        assert wfs.bd.comm.size == 1, \
            'Band parallelization is not supported'
        if wfs.occupations.name != 'mom':
            errormsg = \
                'Please, use occupations={\'name\': \'fixed-uniform\'}'
            assert wfs.occupations.name == 'fixed-uniform', errormsg

    def check_mom(self, wfs, dens):
        occ_name = getattr(wfs.occupations, "name", None)
        if occ_name == 'mom':
            self._e_entropy = wfs.calculate_occupation_numbers(dens.fixed)
            self.restart = self.sort_wavefunctions_mom(wfs)
            self._e_entropy = wfs.calculate_occupation_numbers(dens.fixed)

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

    :param kpt:
    :return:
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

    :param h_mm:
    :param c_nm:
    :return:
    """
    l_nn = np.dot(np.dot(c_nm, h_mm), c_nm.conj().T).conj()
    # check if diagonal then don't rotate? it could save a bit of time
    eps, w = np.linalg.eigh(l_nn)
    return w.T.conj() @ c_nm, eps
