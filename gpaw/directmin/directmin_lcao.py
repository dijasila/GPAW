from ase.units import Hartree
import numpy as np
from gpaw.utilities.blas import mmm
from gpaw.directmin.tools import expm_ed, D_matrix
from gpaw.directmin.sd_lcao import SteepestDescent, FRcg, HZcg, \
    QuickMin, LBFGS
from gpaw.directmin.ls_lcao import UnitStepLength, \
    StrongWolfeConditions


class DirectMinLCAO(object):

    def __init__(self, search_direction_algorithm='FRcg',
                 line_search_algorithm='swc_awc',
                 initial_orbitals='KS',
                 initial_rotation='zero',
                 occupied_only=False,
                 g_tol=1.0e-3,
                 n_counter=1000,
                 memory_lbfgs=3,
                 max_iter_line_search=5,
                 turn_off_swc=False,
                 prec='prec_3', save_orbitals=False,
                 update_refs=10,
                 residual=None):

        self.sda = search_direction_algorithm
        self.lsa = line_search_algorithm
        self.initial_rotation = initial_rotation
        self.initial_orbitals = initial_orbitals
        self.occupied_only = occupied_only
        self.g_tol = g_tol / Hartree
        self.n_counter = n_counter
        self.get_en_and_grad_iters = 0
        self.update_refs_counter = 0
        self.memory_lbfgs = memory_lbfgs
        self.max_iter_line_search = max_iter_line_search
        self.turn_off_swc = turn_off_swc
        self.prec = prec
        self.update_refs = update_refs
        self.residual = residual
        self.initialized = False
        self.save_orbitals = save_orbitals

    def initialize(self, wfs, gd, dtype, nao, diagonalizer=None):

        self.gd = gd
        self.nao = nao
        if diagonalizer is not None:
            self.diagonalizer = diagonalizer
        assert self.diagonalizer is not None
        self.has_initialized = True
        self._error = np.inf

        self.timer = wfs.timer
        self.world = wfs.world
        self.kpt_comm = wfs.kd.comm
        self.band_comm = wfs.bd.comm
        self.gd_comm = wfs.gd.comm
        self.dtype = wfs.dtype
        self.bd = wfs.bd
        self.nbands = wfs.bd.nbands
        self.mynbands = wfs.bd.mynbands
        self.n_kps = wfs.kd.nks // wfs.kd.nspins

        self.n_dim = {}  # dimensionality of the problem
        self.iters = 0

        for kpt in wfs.kpt_u:
            u = kpt.s * self.n_kps + kpt.q
            self.n_dim[u] = self.nbands

        self.initialize_sd_and_ls(wfs, self.sda, self.lsa)

        self.a_mat_u = {}  # skew-hermitian matrix
        self.g_mat_u = {}  # gradient matrix
        self.c_nm_ref = {} # reference orbitals to be rotated

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            self.a_mat_u[k] = np.zeros(shape=(self.n_dim[k],
                                              self.n_dim[k]),
                                       dtype=self.dtype)
            self.g_mat_u[k] = np.zeros(shape=(self.n_dim[k],
                                              self.n_dim[k]),
                                       dtype=self.dtype)
            if kpt.C_nM is None:
                self.c_nm_ref[k] = None

            # self.c_nm_ref[k] = np.copy(kpt.C_nM[:self.n_dim[k]])

        self.alpha = 1.0  # step length
        self.phi = [None, None]  # energy at alpha and alpha old
        self.der_phi = [None, None] # gradients at alpha and alpha old
        self.precond = None

        self.evecs = {}
        self.evals = {}

        self.initialized = True

    def __str__(self):

        return 'direct_minimisation'

    def initialize_sd_and_ls(self, wfs, method, ls_method):

        if method == 'SD':
            self.search_direction = SteepestDescent(wfs)
        elif method == 'HZcg':
            self.search_direction = HZcg(wfs)
        elif method == 'FRcg':
            self.search_direction = FRcg(wfs)
        elif method == 'QuickMin':
            self.search_direction = QuickMin(wfs)
        elif method == 'LBFGS':
            self.search_direction = LBFGS(wfs, self.memory_lbfgs)
        else:
            raise NotImplementedError('Check keyword for '
                                      'search direction!')

        if ls_method == 'UnitStep':
            self.line_search = \
                UnitStepLength(self.evaluate_phi_and_der_phi)
        # elif ls_method == 'Parabola':
        #     self.line_search = Parabola(self.evaluate_phi_and_der_phi,
        #                                 self.log)
        # elif ls_method == 'TwoStepParabola':
        #     self.line_search = \
        #         TwoStepParabola(self.evaluate_phi_and_der_phi,
        #                         self.log)
        # elif ls_method == 'TwoStepParabolaAwc':
        #     self.line_search = \
        #         TwoStepParabolaAwc(self.evaluate_phi_and_der_phi,
        #                            self.log)
        # elif ls_method == 'TwoStepParabolaCubicAwc':
        #     self.line_search = \
        #         TwoStepParabolaCubicAwc(self.evaluate_phi_and_der_phi,
        #                                 self.log)
        elif ls_method == 'swc_awc':
            self.line_search = \
                StrongWolfeConditions(
                    self.evaluate_phi_and_der_phi,
                    method=method,
                    awc=True,
                    max_iter=5
                )
        else:
            raise NotImplementedError('Check keyword for '
                                      'line search!')

    def reset(self):
        pass

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, e):
        self._error = e

    def iterate(self, ham, wfs, dens, occ):

        if self.iters == 0:
            self.iterate_2(ham, wfs)
            self.iters += 1
            return

        a = self.a_mat_u
        g = self.g_mat_u
        n_dim = self.n_dim
        alpha = self.alpha
        phi = self.phi
        c_ref = self.c_nm_ref
        der_phi = self.der_phi
        precond = self.precond

        if self.iters == 1:
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                self.c_nm_ref[k] = kpt.C_nM.copy()
            c_ref = self.c_nm_ref
            dens.direct_min = True

            phi[0], g = \
                self.get_energy_and_gradients(a, n_dim, ham,
                                              wfs, dens, occ,
                                              c_ref)

        p = self.get_search_direction(a, g, precond, wfs)
        der_phi_c = 0.0
        for k in g.keys():
            if self.dtype is complex:
                il1 = np.tril_indices(g[k].shape[0])
            else:
                il1 = np.tril_indices(g[k].shape[0], -1)
            der_phi_c += np.dot(g[k][il1].conj(),
                                p[k][il1]).real
        der_phi_c = wfs.kd.comm.sum(der_phi_c)
        der_phi[0] = der_phi_c

        phi_c, der_phi_c = phi[0], der_phi[0]

        alpha, phi[0], der_phi[0], self.g_mat_u = \
            self.line_search.step_length_update(a, p, n_dim,
                                                ham, wfs, dens, occ,
                                                c_ref,
                                                phi_0=phi[0],
                                                der_phi_0=der_phi[0],
                                                phi_old=phi[1],
                                                der_phi_old=der_phi[1],
                                                alpha_max=3.0,
                                                alpha_old=alpha)
        phi[1], der_phi[1] = phi_c, der_phi_c

        # calculate new matrices for optimal step length
        for k in a.keys():
            self.a_mat_u[k] += alpha * p[k]
        # self.g_mat_u = g
        print(self.iters, phi[0]*Hartree, der_phi[0])
        self.iters += 1


    def get_energy_and_gradients(self, a_mat_u, n_dim, ham, wfs, dens,
                                 occ, c_nm_ref):

        """
        Energy E = E[C exp(A)]. Gradients G_ij[C, A] = dE/dA_ij

        :param a_mat: A
        :param c_nm_ref: C
        :param n_dim:
        :return:
        """

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            if n_dim[k] == 0:
                continue
            self.timer.start('Unitary rotation')
            u_nn, self.evecs[k], self.evals[k] =\
                expm_ed(a_mat_u[k], evalevec=True)
            kpt.C_nM[:n_dim[k]] = np.dot(u_nn.T,
                                         c_nm_ref[k][:n_dim[k]])
            del u_nn
            self.timer.stop('Unitary rotation')
            wfs.atomic_correction.calculate_projections(wfs, kpt)

        self.timer.start('Update Kohn-Sham energy')
        e_total = self.update_ks_energy(ham, wfs, dens, occ)
        self.timer.stop('Update Kohn-Sham energy')

        g_mat_u = {}
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            if n_dim[k] == 0:
                g_mat_u[k] = np.zeros_like(a_mat_u[k])
                continue
            self.timer.start('Calculate gradients')
            h_mm = self.calculate_hamiltonian_matrix(ham, wfs, kpt)
            # make matrix hermitian
            ind_l = np.tril_indices(h_mm.shape[0], -1)
            h_mm[(ind_l[1], ind_l[0])] = h_mm[ind_l].conj()
            g_mat_u[k] = self.get_gradients(h_mm, kpt.C_nM, kpt.f_n,
                                            a_mat_u[k], self.evecs[k],
                                            self.evals[k])
            self.timer.stop('Calculate gradients')

        self.get_en_and_grad_iters += 1

        return e_total, g_mat_u

    def update_ks_energy(self, ham, wfs, dens, occ):

        # using new states update KS
        dens.update(wfs)
        ham.update(dens)
        # occ.calculate(self.wfs)
        e_ks = ham.get_energy(occ)
        e_kin = calculate_kinetic_energy(dens, wfs, wfs.setups)
        e_ks = e_ks - ham.e_kinetic + e_kin
        ham.e_kinetic = e_kin
        ham.e_total_free = e_ks

        # FIXME: new extrapolation of energy?
        # self.ham.e_total_extrapolated = self.e_ks
        return e_ks

    def get_gradients(self, h_mm, c_nm, f_n, a_mat, evec, evals):

        hc_mn = np.zeros_like(h_mm)
        mmm(1.0, h_mm.conj(), 'n', c_nm, 't', 0.0, hc_mn)

        if self.dtype is complex:
            mmm(1.0, c_nm.conj(), 'n', hc_mn, 'n', 0.0, h_mm)
        else:
            mmm(1.0, c_nm, 'n', hc_mn, 'n', 0.0, h_mm)

        h_mm = f_n[:, np.newaxis] * h_mm - f_n * h_mm

        grad = evec.T.conj() @ h_mm.T.conj() @ evec
        grad = grad * D_matrix(evals)
        grad = evec @ grad @ evec.T.conj()
        for i in range(grad.shape[0]):
            grad[i][i] *= 0.5

        if a_mat.dtype == float:
            return 2.0 * grad.real
        else:
            return 2.0 * grad

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

        # self.update_preconditioning()
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
        der_phi = wfs.kd.comm.sum(der_phi)

        return phi, der_phi, g_mat_u

    def calculate_hamiltonian_matrix(self, hamiltonian, wfs, kpt,
                                     Vt_xMM=None, root=-1,
                                     add_kinetic=True):
        # XXX document parallel stuff, particularly root parameter
        assert self.initialized

        bfs = wfs.basis_functions

        # distributed_atomic_correction works with ScaLAPACK/BLACS in general.
        # If SL is not enabled, it will not work with band parallelization.
        # But no one would want that for a practical calculation anyway.
        # dH_asp = wfs.atomic_correction.redistribute(wfs, hamiltonian.dH_asp)
        # XXXXX fix atomic corrections
        dH_asp = hamiltonian.dH_asp

        if Vt_xMM is None:
            wfs.timer.start('Potential matrix')
            vt_G = hamiltonian.vt_sG[kpt.s]
            Vt_xMM = bfs.calculate_potential_matrices(vt_G)
            wfs.timer.stop('Potential matrix')

        if bfs.gamma and wfs.dtype == float:
            yy = 1.0
            H_MM = Vt_xMM[0]
        else:
            wfs.timer.start('Sum over cells')
            yy = 0.5
            k_c = wfs.kd.ibzk_qc[kpt.q]
            H_MM = (0.5 + 0.0j) * Vt_xMM[0]
            for sdisp_c, Vt_MM in zip(bfs.sdisp_xc[1:], Vt_xMM[1:]):
                H_MM += np.exp(2j * np.pi * np.dot(sdisp_c, k_c)) * Vt_MM
            wfs.timer.stop('Sum over cells')

        # Add atomic contribution
        #
        #           --   a     a  a*
        # H      += >   P    dH  P
        #  mu nu    --   mu i  ij nu j
        #           aij
        #
        name = wfs.atomic_correction.__class__.__name__
        wfs.timer.start(name)
        wfs.atomic_correction.calculate_hamiltonian(wfs, kpt, dH_asp, H_MM, yy)
        wfs.timer.stop(name)

        wfs.timer.start('Distribute overlap matrix')
        H_MM = wfs.ksl.distribute_overlap_matrix(
            H_MM, root, add_hermitian_conjugate=(yy == 0.5))
        wfs.timer.stop('Distribute overlap matrix')

        if add_kinetic:
            H_MM += wfs.T_qMM[kpt.q]
        return H_MM

    def iterate_2(self, hamiltonian, wfs):

        wfs.timer.start('LCAO eigensolver')
        s = -1
        for kpt in wfs.kpt_u:
            if kpt.s != s:
                s = kpt.s
                wfs.timer.start('Potential matrix')
                Vt_xMM = wfs.basis_functions.calculate_potential_matrices(
                    hamiltonian.vt_sG[s])
                wfs.timer.stop('Potential matrix')
            self.iterate_one_k_point(hamiltonian, wfs, kpt, Vt_xMM)

        wfs.timer.stop('LCAO eigensolver')

    def iterate_one_k_point(self, hamiltonian, wfs, kpt, Vt_xMM):
        if wfs.bd.comm.size > 1 and wfs.bd.strided:
            raise NotImplementedError

        H_MM = self.calculate_hamiltonian_matrix(hamiltonian, wfs, kpt, Vt_xMM,
                                                 root=0)
        S_MM = wfs.S_qMM[kpt.q]

        if kpt.eps_n is None:
            kpt.eps_n = np.empty(wfs.bd.mynbands)

        diagonalization_string = repr(self.diagonalizer)
        wfs.timer.start(diagonalization_string)
        self.diagonalizer.diagonalize(H_MM, kpt.C_nM, kpt.eps_n, S_MM)
        wfs.timer.stop(diagonalization_string)

        wfs.timer.start('Calculate projections')
        # P_ani are not strictly necessary as required quantities can be
        # evaluated directly using P_aMi/Paaqim.  We should perhaps get rid
        # of the places in the LCAO code using P_ani directly
        wfs.atomic_correction.calculate_projections(wfs, kpt)
        wfs.timer.stop('Calculate projections')


def calculate_kinetic_energy(density, wfs, setups):
    # pseudo-part
    e_kinetic = 0.0
    e_kin_paw = 0.0

    for kpt in wfs.kpt_u:
        rho_MM = \
            wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM)
        e_kinetic += np.einsum('ij,ji->', kpt.T_MM, rho_MM)

    e_kinetic = wfs.kd.comm.sum(e_kinetic)
    # paw corrections
    for a, D_sp in density.D_asp.items():
        setup = setups[a]
        D_p = D_sp.sum(0)
        e_kin_paw += np.dot(setup.K_p, D_p) + setup.Kc

    e_kin_paw = density.gd.comm.sum(e_kin_paw)

    return e_kinetic.real + e_kin_paw
