"""
A class for finding optimal
orbitals of the KS-DFT or PZ-SIC
functionals. Alternative approach to
a density mixing and eigensolovers.

Can be used for excited state calculations as well:
arXiv:2102.06542 [physics.comp-ph]
"""

import numpy as np
from ase.units import Hartree
from ase.utils import basestring
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.xc import xc_string_to_dict
from gpaw.xc.hybrid import HybridXC
from gpaw.utilities import unpack
from gpaw.directmin import search_direction, line_search_algorithm
from gpaw.directmin.functional.fdpw import get_functional
from gpaw.directmin.tools import get_n_occ, sort_orbitals_according_to_occ, sort_orbitals_according_to_occ_kpt
from gpaw.directmin.fdpw.pz_localization import PZLocalization
from gpaw.directmin.fdpw.etdm_inner_loop import ETDMInnerLoop
import time
from ase.parallel import parprint
from gpaw.directmin.locfunc.localize_orbitals import localize_orbitals


class FDPWETDM(Eigensolver):

    def __init__(self,
                 searchdir_algo=None,
                 linesearch_algo='max-step',
                 use_prec=True,
                 functional_settings='ks',
                 need_init_orbs=True,
                 localizationtype=None,
                 localizationseed=None,
                 need_localization=True,
                 localization_tol=None,
                 maxiter=50,
                 maxiterxst=333,
                 maxstepxst=0.2,
                 kappa_tol=5.0e-4,
                 g_tol=5.0e-4,
                 g_tolxst=5.0e-4,
                 momevery=3,
                 printinnerloop=False,
                 blocksize=1,
                 converge_unocc=True,
                 maxiter_unocc=333,
                 exstopt=False):

        super(FDPWETDM, self).__init__(keep_htpsit=False,
                                       blocksize=blocksize)

        self.sda = searchdir_algo
        self.lsa = linesearch_algo
        self.name = 'etdm-fdpw'
        self.use_prec = use_prec
        self.func_settings = functional_settings
        self.localizationtype = localizationtype
        self.localizationseed = localizationseed
        self.localization_tol = localization_tol
        self.maxiter = maxiter
        self.maxiterxst = maxiterxst
        self.maxstepxst = maxstepxst
        self.kappa_tol = kappa_tol
        self.g_tol = g_tol
        self.g_tolxst = g_tolxst
        self.printinnerloop = printinnerloop
        self.converge_unocc = converge_unocc
        self.maxiter_unocc = maxiter_unocc
        self.momevery = momevery

        self.total_eg_count_iloop = 0
        self.total_eg_count_outer_iloop = 0
        self.initial_random = True

        if self.sda is None:
            self.sda = 'LBFGS'
        if isinstance(self.sda, basestring):
            self.sda = xc_string_to_dict(self.sda)
        if isinstance(self.lsa, basestring):
            self.lsa = xc_string_to_dict(self.lsa)
        if isinstance(self.func_settings, basestring):
            self.func_settings = xc_string_to_dict(self.func_settings)

        self.need_init_odd = True
        self.initialized = False
        self.need_init_orbs = need_init_orbs
        self.need_localization = need_localization
        if localizationtype is None:
            self.need_localization = False
        self.eg_count = 0
        self.exstopt = exstopt
        self.etotal = 0.0
        self.globaliters = 0

    def __repr__(self):

        sda_name = self.sda['name'].replace('-', '').lower()
        lsa_name = self.lsa['name'].replace('-', '').lower()

        if isinstance(self.func_settings, basestring):
            func_name = self.func_settings
        else:
            func_name = self.func_settings['name']

        sds = {'sd': 'Steepest Descent',
               'frcg': 'Fletcher-Reeves conj. grad. method',
               'lbfgs': 'L-BFGS algorithm',
               'lbfgsp': 'L-BFGS algorithm with preconditioning',
               'lsr1p': 'Limited-memory SR1P algorithm'}

        lss = {'maxstep': 'step size equals one',
               'swcawc': 'Inexact line search based on cubic interpolation,\n'
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
                       'Orbital-density self-interaction ' \
                       'corrections: {}\n'.format(func_name)
        repr_string += '       ' \
                       'WARNING: do not use it for metals as ' \
                       'occupation numbers are\n' \
                       '                ' \
                       'not found variationally\n'

        return repr_string

    def reset(self, need_init_odd=True):
        self.initialized = False
        self.need_init_odd = need_init_odd

    def todict(self):
        """
        Convert to dictionary, needs for saving and loading gpw
        :return:
        """
        return {'name': 'etdm-fdpw',
                'searchdir_algo': self.sda,
                'linesearch_algo': self.lsa,
                'use_prec': self.use_prec,
                'functional_settings': self.func_settings,
                'need_init_orbs': self.need_init_orbs,
                'localizationtype': self.localizationtype,
                'localizationseed': self.localizationseed,
                'need_localization': self.need_localization,
                'localization_tol': self.localization_tol,
                'maxiter': self.maxiter,
                'maxiterxst': self.maxiterxst,
                'maxstepxst': self.maxstepxst,
                'kappa_tol': self.kappa_tol,
                'g_tol': self.g_tol,
                'g_tolxst': self.g_tolxst,
                'momevery': self.momevery,
                'printinnerloop': self.printinnerloop,
                'blocksize': self.blocksize,
                'converge_unocc': self.converge_unocc,
                'maxiter_unocc': self.maxiter_unocc,
                'exstopt': self.exstopt
                }

    def initialize_dm_helper(self, wfs, ham, dens, log):
        self.initialize_super(wfs, ham)
        self.initialize_orbitals(wfs, ham)

        wfs.calculate_occupation_numbers(dens.fixed)

        self.localize(wfs, dens, ham, log)
        self.initialize_dm(wfs, dens, ham)
        self.init_mom(wfs, dens, log)

    def initialize_super(self, wfs, ham):
        """
        Initialize super class

        :param wfs:
        :param ham:
        :return:
        """
        if isinstance(ham.xc, HybridXC):
            self.blocksize = wfs.bd.mynbands

        if self.blocksize is None:
            if wfs.mode == 'pw':
                S = wfs.pd.comm.size
                # Use a multiple of S for maximum efficiency
                self.blocksize = int(np.ceil(10 / S)) * S
            else:
                self.blocksize = 10

        super(FDPWETDM, self).initialize(wfs)

    def initialize_dm(
            self, wfs, dens, ham, obj_func=None, converge_unocc=False):

        """
        initialize search direction algorithm,
        line search method, SIC corrections

        :param wfs:
        :param dens:
        :param ham:
        :param obj_func:
        :param converge_unocc:
        :return:
        """

        if obj_func is None:
            obj_func = self.evaluate_phi_and_der_phi
        self.dtype = wfs.dtype
        self.n_kps = wfs.kd.nibzkpts
        # dimensionality, number of state to be converged
        self.dimensions = {}
        for kpt in wfs.kpt_u:
            nocc = get_n_occ(kpt)[0]
            if converge_unocc:
                assert nocc < self.bd.nbands, \
                    'Please add empty bands in order to converge the' \
                    ' unoccupied orbitals'
                dim = self.bd.nbands - nocc
            elif self.exstopt:
                dim = self.bd.nbands
            else:
                dim = nocc

            k = self.n_kps * kpt.s + kpt.q
            self.dimensions[k] = dim

        # choose search direction and line search algorithm
        if isinstance(self.sda, (basestring, dict)):
            sda = self.sda

            self.search_direction = search_direction(sda, wfs, self.dimensions)
        else:
            raise Exception('Check Search Direction Parameters')
        if isinstance(self.lsa, (basestring, dict)):
            self.line_search = line_search_algorithm(
                self.lsa, obj_func, self.search_direction)
        else:
            raise Exception('Check Search Direction Parameters')

        if self.use_prec:
            self.prec = wfs.make_preconditioner(1)
        else:
            self.prec = None

        self.iters = 0
        self.alpha = 1.0  # step length
        self.phi_2i = [None, None]  # energy at last two iterations
        self.der_phi_2i = [None, None]  # energy gradient w.r.t. alpha
        self.grad_knG = None

        # odd corrections
        if self.need_init_odd:
            self.odd = get_functional(self.func_settings, wfs, dens, ham)
            self.e_sic = 0.0

            if 'SIC' in self.func_settings['name']:
                self.iloop = PZLocalization(
                    self.odd, wfs, self.kappa_tol, self.maxiter,
                    g_tol=self.g_tol)
            else:
                self.iloop = None

            if self.exstopt:
                self.outer_iloop = ETDMInnerLoop(
                    self.odd, wfs, 'all', self.kappa_tol, self.maxiterxst,
                    self.maxstepxst, g_tol=self.g_tolxst, useprec=True)
                # if you have inner-outer loop then need to have
                # U matrix of the same dimensionality in both loops
                if 'SIC' in self.func_settings['name']:
                    for kpt in wfs.kpt_u:
                        k = self.n_kps * kpt.s + kpt.q
                        self.iloop.U_k[k] = self.outer_iloop.U_k[k].copy()
            else:
                self.outer_iloop = None

        self.total_eg_count_iloop = 0
        self.total_eg_count_outer_iloop = 0

        self.initialized = True

    def iterate(self, ham, wfs, dens, log, converge_unocc=False):
        """
        One iteration of outer loop direct minimization

        :param ham:
        :param wfs:
        :param dens:
        :param log:
        :return:
        """

        n_kps = self.n_kps
        psi_copy = {}
        alpha = self.alpha
        phi_2i = self.phi_2i
        der_phi_2i = self.der_phi_2i

        wfs.timer.start('Direct Minimisation step')

        if self.iters == 0:
            # calculate gradients
            if not converge_unocc:
                phi_2i[0], grad_knG = \
                    self.get_energy_and_tangent_gradients(ham, wfs, dens)
            else:
                phi_2i[0], grad_knG = \
                    self.get_energy_and_tangent_gradients_unocc(ham, wfs)
        else:
            grad_knG = self.grad_knG

        wfs.timer.start('Get Search Direction')
        for kpt in wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            if not converge_unocc:
                psi_copy[k] = kpt.psit_nG.copy()
            else:
                n_occ = get_n_occ(kpt)[0]
                dim = self.dimensions[k]
                psi_copy[k] = kpt.psit_nG[n_occ:n_occ + dim].copy()

        p_knG = self.search_direction.update_data(
            wfs, psi_copy, grad_knG, precond=self.prec,
            dimensions=self.dimensions)
        self.project_search_direction(wfs, p_knG)
        wfs.timer.stop('Get Search Direction')

        # recalculate derivative with new search direction
        # as we used preconditiner before
        # here we project search direction on prec. gradients,
        # but should be just grad. But, it seems also works fine
        der_phi_2i[0] = 0.0
        for kpt in wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            for i, g in enumerate(grad_knG[k]):
                if kpt.f_n[i] > 1.0e-10:
                    der_phi_2i[0] += self.dot(
                        wfs, g, p_knG[k][i], kpt, addpaw=False).item().real
        der_phi_2i[0] = wfs.kd.comm.sum(der_phi_2i[0])

        alpha, phi_alpha, der_phi_alpha, grad_knG = \
            self.line_search.step_length_update(
                psi_copy, p_knG, wfs, ham, dens, converge_unocc,
                phi_0=phi_2i[0], der_phi_0=der_phi_2i[0],
                phi_old=phi_2i[1], der_phi_old=der_phi_2i[1],
                alpha_max=3.0, alpha_old=alpha, kpdescr=wfs.kd)
        self.alpha = alpha
        self.grad_knG = grad_knG

        # and 'shift' phi, der_phi for the next iteration
        phi_2i[1], der_phi_2i[1] = phi_2i[0], der_phi_2i[0]
        phi_2i[0], der_phi_2i[0] = phi_alpha, der_phi_alpha,

        self.iters += 1
        if not converge_unocc:
            self.globaliters += 1
        wfs.timer.stop('Direct Minimisation step')
        return phi_2i[0], self.error

    def update_ks_energy(self, ham, wfs, dens, updateproj=True):

        """
        Update Kohn-Shame energy
        It assumes the temperature is zero K.

        :param ham:
        :param wfs:
        :param dens:
        :return:
        """

        if updateproj:
            # calc projectors
            with wfs.timer('projections'):
                for kpt in wfs.kpt_u:
                    wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

        dens.update(wfs)
        ham.update(dens, wfs, False)

        return ham.get_energy(0.0, wfs, False)

    def evaluate_phi_and_der_phi(self, psit_k, search_dir, alpha, wfs, ham,
                                 dens, converge_unocc, phi=None, grad_k=None):
        """
        phi = E(x_k + alpha_k*p_k)
        der_phi = grad_alpha E(x_k + alpha_k*p_k) cdot p_k
        :return:  phi, der_phi # floats
        """

        if phi is None or grad_k is None:
            alpha1 = np.array([alpha])
            wfs.world.broadcast(alpha1, 0)
            alpha = alpha1[0]

            x_knG = \
                {k: psit_k[k] +
                    alpha * search_dir[k] for k in psit_k.keys()}
            if not converge_unocc:
                phi, grad_k = self.get_energy_and_tangent_gradients(
                    ham, wfs, dens, psit_knG=x_knG)
            else:
                phi, grad_k = self.get_energy_and_tangent_gradients_unocc(
                    ham, wfs, x_knG)

        der_phi = 0.0
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            for i, g in enumerate(grad_k[k]):
                if not converge_unocc and kpt.f_n[i] > 1.0e-10:
                    der_phi += self.dot(
                        wfs, g, search_dir[k][i], kpt,
                        addpaw=False).item().real
                else:
                    der_phi += self.dot(
                        wfs, g, search_dir[k][i], kpt,
                        addpaw=False).item().real
        der_phi = wfs.kd.comm.sum(der_phi)

        return phi, der_phi, grad_k

    def get_energy_and_tangent_gradients(
            self, ham, wfs, dens, psit_knG=None, updateproj=True):

        """
        calculate energy for a given wfs, gradient dE/dpsi
        and then project gradient on tangent space to psi

        :param ham:
        :param wfs:
        :param dens:
        :param psit_knG:
        :return:
        """

        n_kps = self.n_kps
        if psit_knG is not None:
            for kpt in wfs.kpt_u:
                k = n_kps * kpt.s + kpt.q
                kpt.psit_nG[:] = psit_knG[k].copy()
                wfs.orthonormalize(kpt)
        elif not wfs.orthonormalized:
            wfs.orthonormalize()

        if not self.exstopt:
            energy = self.update_ks_energy(
                ham, wfs, dens, updateproj=updateproj)
            grad = self.get_gradients_2(ham, wfs)

            if 'SIC' in self.func_settings['name']:
                e_sic = 0.0
                if self.iters > 0:
                    self.run_inner_loop(ham, wfs, dens, grad_knG=grad)
                else:
                    for kpt in wfs.kpt_u:
                        e_sic += self.odd.get_energy_and_gradients_kpt(
                            wfs, kpt, grad, self.iloop.U_k, add_grad=True)
                    self.e_sic = wfs.kd.comm.sum(e_sic)
                    ham.get_energy(0.0, wfs, kin_en_using_band=False,
                                   e_sic=self.e_sic)
                energy += self.e_sic
        else:
            grad = {}
            n_kps = self.n_kps
            for kpt in wfs.kpt_u:
                grad[n_kps * kpt.s + kpt.q] = np.zeros_like(kpt.psit_nG[:])
            self.run_inner_loop(ham, wfs, dens, grad_knG=grad)
            energy = self.etotal

        self.project_gradient(wfs, grad)
        self.error = self.error_eigv(wfs, grad)
        self.eg_count += 1
        return energy, grad

    def get_gradients_2(self, ham, wfs, scalewithocc=True):

        """
        calculate gradient dE/dpsi
        :return: H |psi_i>
        """

        grad_knG = {}
        n_kps = self.n_kps

        for kpt in wfs.kpt_u:
            grad_knG[n_kps * kpt.s + kpt.q] = \
                self.get_gradients_from_one_k_point_2(
                    ham, wfs, kpt, scalewithocc)

        return grad_knG

    def get_gradients_from_one_k_point_2(
            self, ham, wfs, kpt, scalewithocc=True):
        """
        calculate gradient dE/dpsi for one k-point
        :return: H |psi_i>
        """

        nbands = wfs.bd.mynbands
        Hpsi_nG = wfs.empty(nbands, q=kpt.q)
        wfs.apply_pseudo_hamiltonian(kpt, ham, kpt.psit_nG, Hpsi_nG)

        c_axi = {}
        for a, P_xi in kpt.P_ani.items():
            dH_ii = unpack(ham.dH_asp[a][kpt.s])
            c_xi = np.dot(P_xi, dH_ii)
            c_axi[a] = c_xi

        # not sure about this:
        ham.xc.add_correction(
            kpt, kpt.psit_nG, Hpsi_nG, kpt.P_ani, c_axi, n_x=None,
            calculate_change=False)
        # add projectors to the H|psi_i>
        wfs.pt.add(Hpsi_nG, c_axi, kpt.q)
        # scale with occupation numbers
        if scalewithocc:
            for i, f in enumerate(kpt.f_n):
                Hpsi_nG[i] *= f
        return Hpsi_nG

    def project_gradient(self, wfs, p_knG):
        """
        project gradient dE/dpsi on tangent space at psi
        See Eq.(22) and minimization algorithm p. 12 in
        arXiv:2102.06542v1 [physics.comp-ph]
        :return: H |psi_i>
        """

        n_kps = self.n_kps
        for kpt in wfs.kpt_u:
            kpoint = n_kps * kpt.s + kpt.q
            self.project_gradient_for_one_k_point(wfs, p_knG[kpoint], kpt)

    def project_gradient_for_one_k_point(self, wfs, p_nG, kpt):
        """
        project gradient dE/dpsi on tangent space at psi
        for one k-point.
        See Eq.(22) and minimization algorithm p. 12 in
        arXiv:2102.06542v1 [physics.comp-ph]
        :return: H |psi_i>
        """

        k = self.n_kps * kpt.s + kpt.q
        n_occ = self.dimensions[k]
        psc = wfs.integrate(p_nG[:n_occ], kpt.psit_nG[:n_occ], True)
        psc = 0.5 * (psc.conj() + psc.T)
        s_psit_nG = self.apply_S(wfs, kpt.psit_nG, kpt, kpt.P_ani)
        p_nG[:n_occ] -= np.tensordot(psc, s_psit_nG[:n_occ], axes=1)

    def project_search_direction(self, wfs, p_knG):

        """
        Project search direction on tangent space at psi
        it is slighlt different from project grad
        as it doesn't apply overlap matrix because of S^{-1}

        :param wfs:
        :param p_knG:
        :return:
        """

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            n_occ = self.dimensions[k]
            psc = self.dot(
                wfs, p_knG[k][:n_occ], kpt.psit_nG[:n_occ], kpt, addpaw=False)
            psc = 0.5 * (psc.conj() + psc.T)
            p_knG[k][:n_occ] -= np.tensordot(psc, kpt.psit_nG[:n_occ], axes=1)

    def apply_S(self, wfs, psit_nG, kpt, proj_psi=None):
        """
        apply overlap matrix

        :param wfs:
        :param psit_nG:
        :param kpt:
        :param proj_psi:
        :return:
        """

        if proj_psi is None:
            proj_psi = wfs.pt.dict(shape=wfs.bd.mynbands)
            wfs.pt.integrate(psit_nG, proj_psi, kpt.q)

        s_axi = {}
        for a, P_xi in proj_psi.items():
            dO_ii = wfs.setups[a].dO_ii
            s_xi = np.dot(P_xi, dO_ii)
            s_axi[a] = s_xi

        new_psi_nG = psit_nG.copy()
        wfs.pt.add(new_psi_nG, s_axi, kpt.q)

        return new_psi_nG

    def dot(self, wfs, psi_1, psi_2, kpt, addpaw=True):
        """
        dor product between two arrays psi_1 and psi_2

        :param wfs:
        :param psi_1:
        :param psi_2:
        :param kpt:
        :param addpaw:
        :return:
        """

        dot_prod = wfs.integrate(psi_1, psi_2, global_integral=True)
        if not addpaw:
            if len(psi_1.shape) == 4 or len(psi_1.shape) == 2:
                sum_dot = dot_prod
            else:
                sum_dot = np.asarray([[dot_prod]])

            return sum_dot

        def dS(a, P_ni):
            """
            apply PAW
            :param a:
            :param P_ni:
            :return:
            """
            return np.dot(P_ni, wfs.setups[a].dO_ii)

        if len(psi_1.shape) == 3 or len(psi_1.shape) == 1:
            ndim = 1
        else:
            ndim = psi_1.shape[0]

        P1_ai = wfs.pt.dict(shape=ndim)
        P2_ai = wfs.pt.dict(shape=ndim)
        wfs.pt.integrate(psi_1, P1_ai, kpt.q)
        wfs.pt.integrate(psi_2, P2_ai, kpt.q)
        if ndim == 1:
            if self.dtype is complex:
                paw_dot_prod = np.array([[0.0 + 0.0j]])
            else:
                paw_dot_prod = np.array([[0.0]])

            for a in P1_ai.keys():
                paw_dot_prod += np.dot(dS(a, P2_ai[a]), P1_ai[a].T.conj())
        else:
            paw_dot_prod = np.zeros_like(dot_prod)
            for a in P1_ai.keys():
                paw_dot_prod += np.dot(dS(a, P2_ai[a]), P1_ai[a].T.conj()).T
        paw_dot_prod = np.ascontiguousarray(paw_dot_prod)
        wfs.gd.comm.sum(paw_dot_prod)
        if len(psi_1.shape) == 4 or len(psi_1.shape) == 2:
            sum_dot = dot_prod + paw_dot_prod
        else:
            sum_dot = [[dot_prod]] + paw_dot_prod

        return sum_dot

    def error_eigv(self, wfs, grad_knG):
        """
        calcualte norm of the gradient vector
        (residual)

        :param wfs:
        :param grad_knG:
        :return:
        """

        n_kps = wfs.kd.nibzkpts
        norm = [0.0]
        for kpt in wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            for i, f in enumerate(kpt.f_n):
                if f > 1.0e-10:
                    a = self.dot(
                        wfs, grad_knG[k][i] / f, grad_knG[k][i] / f, kpt,
                        addpaw=False).item() * f
                    a = a.real
                    norm.append(a)
        # error = sum(norm) * Hartree**2 / wfs.nvalence
        error = sum(norm)
        error = wfs.kd.comm.sum(error)

        return error.real

    def get_canonical_representation(self, ham, wfs, rewrite_psi=True):
        """
        choose orbitals which diagonalize the hamiltonain matrix

        <psi_i| H |psi_j>

        For SIC, one diagonalizes L_{ij} = <psi_i| H + V_{j} |psi_j>
        for occupied subspace and
         <psi_i| H |psi_j> for unoccupied.

        :param ham:
        :param wfs:
        :param rewrite_psi:
        :return:
        """
        self.choose_optimal_orbitals(wfs)

        scalewithocc = not self.exstopt

        grad_knG = self.get_gradients_2(ham, wfs, scalewithocc=scalewithocc)
        if 'SIC' in self.func_settings['name']:
            for kpt in wfs.kpt_u:
                self.odd.get_energy_and_gradients_kpt(
                    wfs, kpt, grad_knG, self.iloop.U_k,
                    add_grad=True, scalewithocc=scalewithocc)
        for kpt in wfs.kpt_u:
            if self.exstopt:
                typediag = 'separate'
                k = self.n_kps * kpt.s + kpt.q
                lamb = wfs.integrate(kpt.psit_nG[:], grad_knG[k][:], True)
                if typediag == 'upptr':
                    iu1 = np.triu_indices(lamb.shape[0], 1)
                    il1 = np.tril_indices(lamb.shape[0], -1)
                    lamb[il1] = lamb[iu1]
                    if 'SIC' in self.func_settings['name']:
                        self.odd.lagr_diag_s[k] = np.append(
                            np.diagonal(lamb).real)
                    evals, lamb = np.linalg.eigh(lamb)
                    wfs.gd.comm.broadcast(evals, 0)
                    wfs.gd.comm.broadcast(lamb, 0)
                    kpt.eps_n[:] = evals.copy()
                    if rewrite_psi:
                        lamb = lamb.conj().T
                        kpt.psit_nG[:] = \
                            np.tensordot(lamb, kpt.psit_nG[:],
                                         axes=1)
                        for a in kpt.P_ani.keys():
                            kpt.P_ani[a][:] = \
                                np.dot(lamb, kpt.P_ani[a][:])
                elif typediag == 'separate':
                    n_occ = get_n_occ(kpt)[0]
                    dim = self.bd.nbands - n_occ
                    lamb1 = (lamb[:n_occ, :n_occ] +
                             lamb[:n_occ, :n_occ].T.conj()) / 2.0
                    lumo = (lamb[n_occ:, n_occ:] +
                            lamb[n_occ:, n_occ:].T.conj()) / 2.0
                    if 'SIC' in self.func_settings['name']:
                        self.odd.lagr_diag_s[k] = np.append(
                            np.diagonal(lamb1).real,
                            np.diagonal(lumo).real)
                    if n_occ != 0:
                        evals, lamb1 = np.linalg.eigh(lamb1)
                        wfs.gd.comm.broadcast(evals, 0)
                        wfs.gd.comm.broadcast(lamb1, 0)
                        lamb1 = lamb1.T
                        kpt.eps_n[:n_occ] = evals[:n_occ]
                    evals_lumo, lumo = np.linalg.eigh(lumo)
                    wfs.gd.comm.broadcast(evals_lumo, 0)
                    wfs.gd.comm.broadcast(lumo, 0)
                    kpt.eps_n[n_occ:n_occ + dim] = evals_lumo.real
                    kpt.eps_n[n_occ + dim:] *= 0.0
                    kpt.eps_n[n_occ + dim:] += \
                        np.absolute(
                            5.0 * kpt.eps_n[n_occ + dim - 1])
                    if rewrite_psi:
                        kpt.psit_nG[:n_occ] = \
                            np.tensordot(lamb1.conj(),
                                         kpt.psit_nG[:n_occ],
                                         axes=1)

                        kpt.psit_nG[n_occ:n_occ + dim] = np.tensordot(
                            lumo.conj(), kpt.psit_nG[n_occ:n_occ + dim],
                            axes=1)
                        for a in kpt.P_ani.keys():
                            kpt.P_ani[a][:n_occ] = np.dot(
                                lamb1.conj(), kpt.P_ani[a][:n_occ])
                elif typediag == 'full':
                    lamb = (lamb + lamb.T.conj()) / 2.0
                    evals, lamb = np.linalg.eigh(lamb)
                    wfs.gd.comm.broadcast(evals, 0)
                    wfs.gd.comm.broadcast(lamb, 0)
                    kpt.eps_n[:] = evals.copy()
                    if rewrite_psi:
                        lamb = lamb.conj().T
                        kpt.psit_nG[:] = \
                            np.tensordot(lamb, kpt.psit_nG[:], axes=1)
                        for a in kpt.P_ani.keys():
                            kpt.P_ani[a][:] = np.dot(lamb, kpt.P_ani[a][:])
                else:
                    raise KeyError
            else:
                # TODO: if homo-lumo is around zero then
                #  it is not good to do diagonalization
                #  of occupied and unoccupied states
                # separete diagonaliztion of two subspaces:
                k = self.n_kps * kpt.s + kpt.q
                n_occ = get_n_occ(kpt)[0]
                dim = self.bd.nbands - n_occ
                grad_knG[k][n_occ:n_occ + dim] = \
                    self.get_gradients_unocc(ham, wfs, kpt)
                lamb = wfs.integrate(kpt.psit_nG[:n_occ],
                                     grad_knG[k][:n_occ],
                                     True)
                lamb = (lamb + lamb.T.conj()) / 2.0
                lumo = wfs.integrate(kpt.psit_nG[n_occ:n_occ + dim],
                                     grad_knG[k][n_occ:n_occ + dim],
                                     True)
                lumo = (lumo + lumo.T.conj()) / 2.0

                lo_nn = np.diagonal(lamb).real / kpt.f_n[:n_occ]
                lu_nn = np.diagonal(lumo).real / 1.0
                if n_occ != 0:
                    evals, lamb = np.linalg.eigh(lamb)
                    wfs.gd.comm.broadcast(evals, 0)
                    wfs.gd.comm.broadcast(lamb, 0)
                    lamb = lamb.T
                    kpt.eps_n[:n_occ] = evals[:n_occ] / kpt.f_n[:n_occ]

                evals_lumo, lumo = np.linalg.eigh(lumo)
                wfs.gd.comm.broadcast(evals_lumo, 0)
                wfs.gd.comm.broadcast(lumo, 0)
                lumo = lumo.T

                kpt.eps_n[n_occ:n_occ + dim] = evals_lumo.real
                kpt.eps_n[n_occ + dim:] *= 0.0
                kpt.eps_n[n_occ + dim:] += \
                    np.absolute(5.0 * kpt.eps_n[n_occ + dim - 1])
                if rewrite_psi:
                    kpt.psit_nG[:n_occ] = \
                        np.tensordot(lamb.conj(), kpt.psit_nG[:n_occ], axes=1)
                    kpt.psit_nG[n_occ:n_occ + dim] = np.tensordot(
                        lumo.conj(), kpt.psit_nG[n_occ:n_occ + dim], axes=1)
                orb_en = [lo_nn, lu_nn]
                for i in [0, 1]:
                    ind = np.argsort(orb_en[i])
                    orb_en[i][:] = orb_en[i][ind]
                    if not rewrite_psi:
                        # we need to sort wfs
                        kpt.psit_nG[n_occ * i + np.arange(len(ind)), :] = \
                            kpt.psit_nG[n_occ * i + ind, :]
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
                if 'SIC' in self.func_settings['name']:
                    self.odd.lagr_diag_s[k] = np.append(lo_nn, lu_nn)

        del grad_knG

    def get_gradients_unocc(self, ham, wfs, kpt):

        """
        calculate gradient vector for unoccupied orbitals

        :param ham:
        :param wfs:
        :param kpt:
        :return:
        """

        n_occ = get_n_occ(kpt)[0]
        dim = self.bd.nbands - n_occ
        # calculate gradients:
        psi = kpt.psit_nG[n_occ:n_occ + dim].copy()
        P1_ai = wfs.pt.dict(shape=dim)
        wfs.pt.integrate(psi, P1_ai, kpt.q)
        Hpsi_nG = wfs.empty(dim, q=kpt.q)
        wfs.apply_pseudo_hamiltonian(kpt, ham, psi, Hpsi_nG)
        c_axi = {}
        for a, P_xi in P1_ai.items():
            dH_ii = unpack(ham.dH_asp[a][kpt.s])
            c_xi = np.dot(P_xi, dH_ii)
            c_axi[a] = c_xi
        # not sure about this:
        ham.xc.add_correction(kpt, psi, Hpsi_nG,
                              P1_ai, c_axi, n_x=None,
                              calculate_change=False)
        # add projectors to the H|psi_i>
        wfs.pt.add(Hpsi_nG, c_axi, kpt.q)

        return Hpsi_nG

    def get_energy_and_tangent_gradients_unocc(self, ham, wfs, psit_knG=None):
        """
        calculate energy and trangent gradients of
        unooccupied orbitals

        :param ham:
        :param wfs:
        :param psit_knG:
        :return:
        """
        wfs.timer.start('Gradient unoccupied orbitals')
        n_kps = self.n_kps
        if psit_knG is not None:
            for kpt in wfs.kpt_u:
                k = n_kps * kpt.s + kpt.q
                # find lumo
                n_occ = get_n_occ(kpt)[0]
                dim = self.dimensions[k]
                kpt.psit_nG[n_occ:n_occ + dim] = psit_knG[k].copy()
                wfs.orthonormalize(kpt)
        elif not wfs.orthonormalized:
            wfs.orthonormalize()

        grad = {}
        energy_t = 0.0
        error_t = 0.0

        for kpt in wfs.kpt_u:
            n_occ = get_n_occ(kpt)[0]
            k = n_kps * kpt.s + kpt.q
            dim = self.dimensions[k]
            # calculate gradients:
            psi = kpt.psit_nG[n_occ:n_occ + dim].copy()
            P1_ani = wfs.pt.dict(shape=dim)
            wfs.pt.integrate(psi, P1_ani, kpt.q)
            Hpsi_nG = wfs.empty(dim, q=kpt.q)
            wfs.apply_pseudo_hamiltonian(kpt, ham, psi, Hpsi_nG)
            c_axi = {}
            for a, P_xi in P1_ani.items():
                dH_ii = unpack(ham.dH_asp[a][kpt.s])
                c_xi = np.dot(P_xi, dH_ii)
                c_axi[a] = c_xi
            # not sure about this:
            ham.xc.add_correction(kpt, psi, Hpsi_nG,
                                  P1_ani, c_axi, n_x=None,
                                  calculate_change=False)
            # add projectors to the H|psi_i>
            wfs.pt.add(Hpsi_nG, c_axi, kpt.q)
            grad[k] = Hpsi_nG.copy()

            # calculate energy
            psi = kpt.psit_nG[:n_occ + dim].copy()
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
            lamb = wfs.integrate(psi, Hpsi_nG, global_integral=True)
            s_axi = {}
            for a, P_xi in kpt.P_ani.items():
                dO_ii = wfs.setups[a].dO_ii
                s_xi = np.dot(P_xi, dO_ii)
                s_axi[a] = s_xi
            wfs.pt.add(psi, s_axi, kpt.q)

            grad[k] -= np.tensordot(lamb.T, psi, axes=1)

            minstate = np.argmin(np.diagonal(lamb, offset=-n_occ).real)
            energy = np.diagonal(lamb, offset=-n_occ)[minstate].real

            norm = []
            for i in [minstate]:
                norm.append(self.dot(
                    wfs, grad[k][i], grad[k][i], kpt, addpaw=False).item())
            error = sum(norm).real * Hartree ** 2 / len(norm)
            error_t += error
            energy_t += energy

        error_t = wfs.kd.comm.sum(error_t)
        energy_t = wfs.kd.comm.sum(energy_t)
        self.error = error_t

        wfs.timer.stop('Gradient unoccupied orbitals')

        return energy_t, grad

    def run_unocc(self, ham, wfs, dens, max_err, log):

        """
        Converge unoccupied orbitals

        :param ham:
        :param wfs:
        :param dens:
        :param max_err:
        :param log:
        :return:
        """

        self.need_init_odd = False
        self.initialize_dm(
            wfs, dens, ham,
            obj_func=self.evaluate_phi_and_der_phi, converge_unocc=True)

        self.maxiter_unocc = 100
        while self.iters < self.maxiter_unocc:
            en, er = self.iterate(ham, wfs, dens, log, converge_unocc=True)
            log_f(self.iters, en, er, log)
            # it is quite difficult to converge unoccupied orbitals
            # with the same accuracy as occupied orbitals
            if er < max(max_err, 5.0e-4):
                log('\nUnoccupied orbitals converged after'
                    ' {:d} iterations'.format(self.iters))
                break
            if self.iters >= self.maxiter_unocc:
                log('\nUnoccupied orbitals did not converge after'
                    ' {:d} iterations'.format(self.iters))

        self.initialized = False

    def run_inner_loop(self, ham, wfs, dens, grad_knG, niter=0):

        """
        calculate optimal orbitals among occupied subspace
        which minimizes SIC.

        :param ham:
        :param wfs:
        :param dens:
        :param grad_knG:
        :param niter:
        :return:
        """

        if self.iloop is None and self.outer_iloop is None:
            return niter, False

        wfs.timer.start('Inner loop')

        if self.printinnerloop:
            log = parprint
        else:
            log = None

        if self.iloop is not None:
            if self.exstopt and self.iters == 0:
                eks = self.update_ks_energy(ham, wfs, dens)
            else:
                etotal = ham.get_energy(
                    0.0, wfs, kin_en_using_band=False, e_sic=self.e_sic)
                eks = etotal - self.e_sic
            if self.initial_random and self.iters == 1:
                small_random = True
            else:
                small_random = False
            self.e_sic, counter = self.iloop.run(
                eks, wfs, dens, log, niter, small_random=small_random,
                seed=self.localizationseed)
            self.total_eg_count_iloop += self.iloop.eg_count

            if self.outer_iloop is None:
                if grad_knG is not None:
                    for kpt in wfs.kpt_u:
                        k = self.n_kps * kpt.s + kpt.q
                        n_occ = get_n_occ(kpt)[0]
                        grad_knG[k][:n_occ] += np.tensordot(
                            self.iloop.U_k[k].conj(),
                            self.iloop.odd_pot.grad[k], axes=1)
                wfs.timer.stop('Inner loop')

                ham.get_energy(0.0, wfs, kin_en_using_band=False,
                               e_sic=self.e_sic)
                return counter, True

            for kpt in wfs.kpt_u:
                k = self.iloop.n_kps * kpt.s + kpt.q
                U = self.iloop.U_k[k]
                n_occ = U.shape[0]
                kpt.psit_nG[:n_occ] = np.tensordot(
                    U.T, kpt.psit_nG[:n_occ], axes=1)
                # calc projectors
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

        self.etotal, counter = self.outer_iloop.run(
            wfs, dens, log, niter, ham=ham)
        self.total_eg_count_outer_iloop += self.outer_iloop.eg_count
        self.e_sic = self.outer_iloop.esic
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            grad_knG[k] += np.tensordot(self.outer_iloop.U_k[k].conj(),
                                        self.outer_iloop.odd_pot.grad[k],
                                        axes=1)
            if self.iloop is not None:
                U = self.iloop.U_k[k]
                n_occ = U.shape[0]
                kpt.psit_nG[:n_occ] = \
                    np.tensordot(U.conj(),
                                 kpt.psit_nG[:n_occ], axes=1)
                # calc projectors
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
                grad_knG[k][:n_occ] = \
                    np.tensordot(U.conj(),
                                 grad_knG[k][:n_occ], axes=1)
                self.iloop.U_k[k] = \
                    self.iloop.U_k[k] @ self.outer_iloop.U_k[k]
                self.outer_iloop.U_k[k] = np.eye(n_occ, dtype=self.dtype)

        wfs.timer.stop('Inner loop')

        ham.get_energy(0.0, wfs, kin_en_using_band=False,
                       e_sic=self.e_sic)

        return counter, True

    def initialize_orbitals(self, wfs, ham):
        """
        :param wfs:
        :param ham:
        :return:
        """
        if self.need_init_orbs and not wfs.read_from_file_init_wfs_dm:
            for kpt in wfs.kpt_u:
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
                super(FDPWETDM, self).subspace_diagonalize(
                    ham, wfs, kpt, True)
                wfs.gd.comm.broadcast(kpt.eps_n, 0)
            self.need_init_orbs = False
        if wfs.read_from_file_init_wfs_dm:
            self.initial_random = False

    def localize(self, wfs, dens, ham, log):
        if not self.need_localization:
            return
        localize_orbitals(wfs, dens, ham, log, self.localizationtype,
                          tol=self.localization_tol,
                          seed=self.localizationseed,
                          func_settings=self.func_settings)
        self.need_localization = False

    def choose_optimal_orbitals(self, wfs):
        """
        choose optimal orbitals and store them in wfs.kpt_u.
        Optimal orbitals are those which minimize the energy
        functional and might not coincide with canonical orbitals

        :param wfs:
        :return:
        """
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            if self.iloop is not None:
                dim = self.iloop.U_k[k].shape[0]
                kpt.psit_nG[:dim] = \
                    np.tensordot(
                        self.iloop.U_k[k].T, kpt.psit_nG[:dim],
                        axes=1)
                self.iloop.U_k[k] = np.eye(self.iloop.U_k[k].shape[0])
                self.iloop.Unew_k[k] = np.eye(
                    self.iloop.Unew_k[k].shape[0])
            if self.outer_iloop is not None:
                dim = self.outer_iloop.U_k[k].shape[0]
                kpt.psit_nG[:dim] = \
                    np.tensordot(
                        self.outer_iloop.U_k[k].T,
                        kpt.psit_nG[:dim], axes=1)
                self.outer_iloop.U_k[k] = np.eye(
                    self.outer_iloop.U_k[k].shape[0])
                self.outer_iloop.Unew_k[k] = np.eye(
                    self.outer_iloop.Unew_k[k].shape[0])
            if self.iloop is not None or \
                    self.outer_iloop is not None:
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

    def check_assertions(self, wfs, dens):

        assert dens.mixer.driver.basemixerclass.name == 'no-mixing', \
            'Please, use: mixer={\'backend\': \'no-mixing\'}'
        assert wfs.bd.comm.size == 1, \
            'Band parallelization is not supported'
        if wfs.occupations.name != 'mom':
            errormsg = \
                'Please, use occupations={\'name\': \'fixed-uniform\'}'
            assert wfs.occupations.name == 'fixed-uniform', errormsg

    def check_mom(self, wfs, ham, dens):

        occ_name = getattr(wfs.occupations, 'name', None)
        if occ_name != 'mom':
            return

        sic_calc = 'SIC' in self.func_settings['name']
        iloop = self.outer_iloop is not None
        update = False

        if iloop:
            astmnt = self.outer_iloop.odd_pot.restart
            bstmnt = (self.iters + 1) % self.momevery == 0 and \
                not self.outer_iloop.converged
            if astmnt or bstmnt:
                update = True
        if update and not wfs.occupations.use_fixed_occupations:
            self.choose_optimal_orbitals(wfs)
            if not sic_calc:
                for kpt in wfs.kpt_u:
                    wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
                    super(FDPWETDM, self).subspace_diagonalize(
                        ham, wfs, kpt, True)
                    wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
                    wfs.gd.comm.broadcast(kpt.eps_n, 0)
            wfs.calculate_occupation_numbers(dens.fixed)
            sort_orbitals_according_to_occ(wfs, update_mom=True)
            wfs.calculate_occupation_numbers(dens.fixed)
            self.update_ks_energy(ham, wfs, dens, updateproj=True)
            self.iters = 0
            self.initialized = False
            self.need_init_odd = True

    def init_mom(self, wfs, dens, log):
        occ_name = getattr(wfs.occupations, 'name', None)
        if occ_name != 'mom':
            return
        # we need to do it in order to initialize mom..
        # it will take occupied orbitals from previous step
        if self.globaliters == 0:
            for kpt in wfs.kpt_u:
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
            wfs.orthonormalize()
            wfs.occupations.initialize_reference_orbitals()
            log(" MOM reference orbitals initialized.\n", flush=True)
            # fill occ numbers
            wfs.calculate_occupation_numbers(dens.fixed)
            sort_orbitals_according_to_occ(wfs, update_mom=True)
            wfs.calculate_occupation_numbers(dens.fixed)
        return


def log_f(niter, e_total, eig_error, log):
    """
    log function for convergence of unoccupied states.

    :param niter:
    :param e_total:
    :param eig_error:
    :param log:
    :return:
    """

    T = time.localtime()
    if niter == 1:
        header = '                              ' \
                 '     wfs    \n' \
                 '           time        Energy:' \
                 '     error(ev^2):'
        log(header)

    log('iter: %3d  %02d:%02d:%02d ' %
        (niter,
         T[3], T[4], T[5]
         ), end='')

    log('%11.6f %11.1e' %
        (Hartree * e_total, eig_error), end='')

    log(flush=True)