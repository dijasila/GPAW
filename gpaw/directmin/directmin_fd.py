import numpy as np
from ase.units import Hartree
from ase.utils import basestring
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.xc import xc_string_to_dict
from gpaw.xc.hybrid import HybridXC
from gpaw.utilities import unpack
from gpaw.directmin.fd import sd_outer, ls_outer
from gpaw.utilities.lapack import diagonalize
from gpaw.directmin.odd import odd_corrections
from gpaw.directmin.tools import get_n_occ
from gpaw.directmin.fd.inner_loop import InnerLoop
from gpaw.pipekmezey.pipek_mezey_wannier import PipekMezey
from gpaw.pipekmezey.wannier_basic import WannierLocalization
import time
from ase.parallel import parprint
# from gpaw.utilities.memory import maxrss


class DirectMinFD(Eigensolver):

    def __init__(self,
                 searchdir_algo=None,
                 linesearch_algo='TSPCAWC',
                 use_prec=True,
                 odd_parameters='Zero',
                 inner_loop=None,
                 initial_orbitals=None,
                 blocksize=1):

        super(DirectMinFD, self).__init__(keep_htpsit=False,
                                          blocksize=blocksize)

        self.sda = searchdir_algo
        self.lsa = linesearch_algo
        self.name = 'direct_min'
        self.use_prec = use_prec
        self.odd_parameters = odd_parameters
        self.inner_loop = inner_loop
        self.initial_orbitals = initial_orbitals

        if isinstance(self.odd_parameters, basestring):
            self.odd_parameters = \
                xc_string_to_dict(self.odd_parameters)

        if 'SIC' in self.odd_parameters['name']:
            if self.sda is None:
                self.sda = 'FRcg'
            if self.initial_orbitals is None:
                self.initial_orbitals = 'W'
        else:
            if self.sda is None:
                self.sda = 'LBFGS'

        if isinstance(self.sda, basestring):
            self.sda = xc_string_to_dict(self.sda)
        if isinstance(self.lsa, basestring):
            self.lsa = xc_string_to_dict(self.lsa)
            self.lsa['method'] = self.sda['name']

        self.need_init_odd = True
        self.initialized = False
        self.need_init_orbs = True
        self.U_k = {}

    def __repr__(self):

        sds = {'SD': 'Steepest Descent',
               'FRcg': 'Fletcher-Reeves conj. grad. method',
               'HZcg': 'Hager-Zhang conj. grad. method',
               'PRcg': 'Polak-Ribiere conj. grad. method',
               'PRpcg': 'Polak-Ribiere+ conj. grad. method',
               'QuickMin': 'Velocity projection algorithm',
               'LBFGS': 'LBFGS algorithm',
               'LBFGS_P': 'LBFGS algorithm with preconditioning'}

        lss = {'UnitStep': 'step size equals one',
               'Parabola': 'Parabolic line search',
               'TSP': 'Parabolic two-step line search ',
               'TSPAWC': 'Parabolic two-step line search with\n'
                         '                  '
                         ' approximate Wolfe conditions',
               'TSPCAWC': 'Parabolic and Cubic two-step '
                          'line search with\n'
                          '                   '
                         ' approximate Wolfe conditions',
               'TSPCD': 'Parabolic and Cubic two-step '
                          'line search with\n'
                          '                   '
                          'descent condition',
               'SwcAwc': 'Inexact line search based '
                         'on cubic interpolation,\n'
                         '                    strong'
                         ' and approximate Wolfe conditions'}

        repr_string = 'Direct minimisation\n' \

        repr_string += '       ' \
                       'Search ' \
                       'direction: {}\n'.format(sds[self.sda['name']])
        repr_string += '       ' \
                       'Line ' \
                       'search: {}\n'.format(lss[self.lsa['name']])
        repr_string += '       ' \
                       'Preconditioning: {}\n'.format(self.use_prec)

        repr_string += '       '\
                       'Orbital-density self-interaction ' \
                       'corrections: {}\n'.format(
            self.odd_parameters['name'])

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
        return {'name': 'direct_min_fd',
                'searchdir_algo': self.sda,
                'linesearch_algo': self.lsa,
                'use_prec': self.use_prec,
                'odd_parameters': self.odd_parameters
                }

    def initialize_super(self, wfs):

        if self.blocksize is None:
            if wfs.mode == 'pw':
                S = wfs.pd.comm.size
                # Use a multiple of S for maximum efficiency
                self.blocksize = int(np.ceil(10 / S)) * S
            else:
                self.blocksize = 10

        super(DirectMinFD, self).initialize(wfs)

    def initialize_dm(self, wfs, dens, ham, log=None,
                      obj_func=None, lumo=False):

        if obj_func is None:
            obj_func = self.evaluate_phi_and_der_phi
        self.dtype = wfs.dtype
        self.n_kps = wfs.kd.nks // wfs.kd.nspins
        # dimensionality, number of state to be converged:
        self.dimensions = {}
        for kpt in wfs.kpt_u:
            if lumo:
                dim = 1
            else:
                dim = get_n_occ(kpt)
            k = self.n_kps * kpt.s + kpt.q
            self.dimensions[k] = dim
            if 'SIC' in self.odd_parameters['name']:
                self.U_k[k] = np.eye(dim, dtype=self.dtype)

            if not lumo and self.dimensions[k] == len(kpt.f_n):
                raise Exception('Please add one more empty band '
                                'in order to converge LUMO.')
        # choose search direction and line search algorithm
        if isinstance(self.sda, (basestring, dict)):
            self.search_direction = sd_outer(self.sda, wfs,
                                             self.dimensions)
        else:
            raise Exception('Check Search Direction Parameters')
        if isinstance(self.lsa, (basestring, dict)):
            self.line_search = \
                ls_outer(self.lsa, obj_func)
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
        # self.iloop = None
        # self.odd = None
        if self.need_init_odd:
            if isinstance(self.odd_parameters, (basestring, dict)):
                self.odd = odd_corrections(self.odd_parameters, wfs,
                                           dens, ham)
            else:
                raise Exception('Check ODD Parameters')
            self.e_sic = 0.0

            iloop = (self.inner_loop is None and
                     'SIC' in self.odd_parameters['name']) or \
                self.inner_loop is True

            if iloop:
                self.iloop = InnerLoop(self.odd, wfs)
            else:
                self.iloop = None

        self.initialized = True

    def iterate(self, ham, wfs, dens, occ, log):

        assert dens.mixer.driver.name == 'dummy', \
            'Please, use: mixer={\'method\': \'dummy\'}'
        assert wfs.bd.comm.size == 1, \
            'Band parallelization is not supported'
        assert occ.width < 1.0e-5, \
            'Zero Kelvin only.'

        if not self.initialized:
            if isinstance(ham.xc, HybridXC):
                self.blocksize = wfs.bd.mynbands
            self.initialize_super(wfs)
            self.init_wfs(wfs, ham, occ, log)
            self.initialize_dm(wfs, dens, ham, log)

        n_kps = self.n_kps
        psi_copy = {}
        alpha = self.alpha
        phi_2i = self.phi_2i
        der_phi_2i = self.der_phi_2i

        wfs.timer.start('Direct Minimisation step')

        if self.iters == 0:
            # calculate gradients
            phi_2i[0], grad_knG = \
                self.get_energy_and_tangent_gradients(ham, wfs, dens,
                                                      occ)
            # self.error = self.error_eigv(wfs, grad_knG)
        else:
            grad_knG = self.grad_knG

        wfs.timer.start('Get Search Direction')
        for kpt in wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            psi_copy[k] = kpt.psit_nG.copy()
        p_knG = self.search_direction.update_data(psi_copy, grad_knG,
                                                  wfs, self.prec)
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
                    der_phi_2i[0] += \
                        self.dot(wfs, g, p_knG[k][i], kpt).item().real
        der_phi_2i[0] = wfs.kd.comm.sum(der_phi_2i[0])

        alpha, phi_alpha, der_phi_alpha, grad_knG = \
            self.line_search.step_length_update(
                psi_copy, p_knG, ham, wfs, dens, occ,
                phi_0=phi_2i[0], der_phi_0=der_phi_2i[0],
                phi_old=phi_2i[1], der_phi_old=der_phi_2i[1],
                alpha_max=3.0, alpha_old=alpha)

        # calculate new wfs:
        # do we actually need to do this?
        # for kpt in wfs.kpt_u:
        #     k = n_kps * kpt.s + kpt.q
        #     kpt.psit_nG[:] = psi_copy[k] + alpha * p_knG[k]
        # wfs.orthonormalize()

        self.alpha = alpha
        self.grad_knG = grad_knG
        self.iters += 1

        # and 'shift' phi, der_phi for the next iteration
        phi_2i[1], der_phi_2i[1] = phi_2i[0], der_phi_2i[0]
        phi_2i[0], der_phi_2i[0] = phi_alpha, der_phi_alpha,

        wfs.timer.stop('Direct Minimisation step')

    def update_ks_energy(self, ham, wfs, dens, occ):

        wfs.timer.start('Update Kohn-Sham energy')

        # calc projectors
        for kpt in wfs.kpt_u:
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

        # occ.calculate(wfs)
        dens.update(wfs)
        ham.update(dens, wfs, False)
        wfs.timer.stop('Update Kohn-Sham energy')

        return ham.get_energy(occ, False)

    def evaluate_phi_and_der_phi(self, psit_k, search_dir, alpha,
                                 ham, wfs, dens, occ,
                                 phi=None, grad_k=None):
        """
        phi = E(x_k + alpha_k*p_k)
        der_phi = grad_alpha E(x_k + alpha_k*p_k) cdot p_k
        :return:  phi, der_phi # floats
        """

        if phi is None or grad_k is None:
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                kpt.psit_nG[:] = psit_k[k] + alpha * search_dir[k]
            wfs.orthonormalize()

            phi, grad_k = \
                self.get_energy_and_tangent_gradients(ham, wfs, dens,
                                                      occ)

        der_phi = 0.0
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            for i, g in enumerate(grad_k[k]):
                if kpt.f_n[i] > 1.0e-10:
                    der_phi += self.dot(wfs,
                                        g, search_dir[k][i],
                                        kpt).item().real
        der_phi = wfs.kd.comm.sum(der_phi)

        return phi, der_phi, grad_k

    def get_energy_and_tangent_gradients(self, ham, wfs, dens, occ,
                                         psit_knG=None):

        n_kps = self.n_kps
        if psit_knG is not None:
            for kpt in wfs.kpt_u:
                k = n_kps * kpt.s + kpt.q
                kpt.psit_nG[:] = psit_knG[k].copy()
                wfs.orthonormalize(kpt)
        elif not wfs.orthonormalized:
            wfs.orthonormalize()

        energy = self.update_ks_energy(ham, wfs, dens, occ)
        grad = self.get_gradients_2(ham, wfs)

        if 'SIC' in self.odd_parameters['name']:
            temp = {}
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                temp[k] = kpt.psit_nG[:].copy()
                n_occ=get_n_occ(kpt)
                self.U_k[k] = self.U_k[k] @ self.iloop.U_k[k].copy()
                kpt.psit_nG[:n_occ] = \
                    np.tensordot(
                        self.U_k[k].T, kpt.psit_nG[:n_occ], axes=1)

            self.e_sic = 0.0
            error = self.error * Hartree ** 2 / wfs.nvalence
            if error > 1.0e-8 and self.iters > 0:
                self.run_inner_loop(ham, wfs, occ, dens, log=None)
            self.e_sic = 0.0
            for kpt in wfs.kpt_u:
                k = n_kps * kpt.s + kpt.q
                self.e_sic +=\
                    self.odd.get_energy_and_gradients_kpt_2(
                        wfs, kpt, grad, dens,
                        U = self.U_k[k] @ self.iloop.U_k[k])
            self.e_sic = wfs.kd.comm.sum(self.e_sic)
            energy += self.e_sic
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                kpt.psit_nG[:] = temp[k]

        self.project_search_direction_2(wfs, grad)
        self.error = self.error_eigv(wfs, grad)

        # self.eg_counter += 1

        return energy, grad

    def get_gradients_2(self, ham, wfs):

        """
        :return: H |psi_i>
        """

        grad_knG = {}
        n_kps = self.n_kps

        for kpt in wfs.kpt_u:
            grad_knG[n_kps * kpt.s + kpt.q] = \
                self.get_gradients_from_one_k_point_2(ham, wfs, kpt)

        return grad_knG

    def get_gradients_from_one_k_point_2(self, ham, wfs, kpt):

        nbands = wfs.bd.mynbands
        Hpsi_nG = wfs.empty(nbands, q=kpt.q)
        wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
        wfs.apply_pseudo_hamiltonian(kpt, ham, kpt.psit_nG, Hpsi_nG)

        c_axi = {}
        for a, P_xi in kpt.P_ani.items():
            dH_ii = unpack(ham.dH_asp[a][kpt.s])
            c_xi = np.dot(P_xi, dH_ii)
            c_axi[a] = c_xi

        # not sure about this:
        ham.xc.add_correction(kpt, kpt.psit_nG, Hpsi_nG,
                              kpt.P_ani, c_axi, n_x=None,
                              calculate_change=False)
        # add projectors to the H|psi_i>
        wfs.pt.add(Hpsi_nG, c_axi, kpt.q)
        # scale with occupation numbers
        for i, f in enumerate(kpt.f_n):
            Hpsi_nG[i] *= f

        return Hpsi_nG

    def project_search_direction_2(self, wfs, p_knG):

        n_kps = self.n_kps
        for kpt in wfs.kpt_u:
            kpoint = n_kps * kpt.s + kpt.q
            self.project_search_direction_for_one_k_point_2(
                wfs, p_knG[kpoint], kpt)

    def project_search_direction_for_one_k_point_2(self, wfs, p_nG,
                                                   kpt):

        def dot_2(psi_1, psi_2, wfs):
            dot_prod = wfs.gd.integrate(psi_1, psi_2, False)
            # ?
            dot_prod = np.ascontiguousarray(dot_prod)
            if len(psi_1.shape) == 3:
                dot_prod = wfs.gd.comm.sum(dot_prod)
                return dot_prod
            else:
                wfs.gd.comm.sum(dot_prod)
                return dot_prod

        k = self.n_kps * kpt.s + kpt.q
        n_occ = self.dimensions[k]
        psc = dot_2(p_nG[:n_occ], kpt.psit_nG[:n_occ], wfs)
        psc = 0.5 * (psc.conj() + psc.T)
        s_psit_nG = self.apply_S(wfs, kpt.psit_nG, kpt)
        p_nG[:n_occ] -= np.tensordot(psc, s_psit_nG[:n_occ], axes=1)

    def project_search_direction(self, wfs, p_knG):

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            n_occ = self.dimensions[k]
            psc = self.dot(wfs, p_knG[k][:n_occ], kpt.psit_nG[:n_occ],
                           kpt)
            psc = 0.5 * (psc.conj() + psc.T)
            p_knG[k][:n_occ] -= np.tensordot(psc, kpt.psit_nG[:n_occ],
                                             axes=1)

    def apply_S(self, wfs, psit_nG, kpt):

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

    def dot(self, wfs, psi_1, psi_2, kpt):

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
            if self.dtype is complex:
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
            paw_dot_prod = np.zeros_like(dot_prod)
            for a in P1_ai.keys():
                paw_dot_prod += \
                    np.dot(dS(a, P2_ai[a]), P1_ai[a].T.conj()).T
            sum_dot = dot_prod + paw_dot_prod
        sum_dot = np.ascontiguousarray(sum_dot)
        wfs.gd.comm.sum(sum_dot)

        return sum_dot

    def error_eigv(self, wfs, grad_knG):

        n_kps = wfs.kd.nks // wfs.kd.nspins
        norm = []
        for kpt in wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            for i, f in enumerate(kpt.f_n):
                if f > 1.0e-10:
                    norm.append(self.dot(wfs,
                                         grad_knG[k][i] / f,
                                         grad_knG[k][i] / f,
                                         kpt).item() * f)

        # error = sum(norm) * Hartree**2 / wfs.nvalence
        error = sum(norm)
        error = wfs.kd.comm.sum(error)

        return error.real

    def get_canonical_representation(self, ham, wfs, occ, dens,
                                     rewrite_psi=True):

        grad_knG = self.get_gradients_2(ham, wfs)
        if 'SIC' in self.odd_parameters['name']:
            for kpt in wfs.kpt_u:
                self.odd.get_energy_and_gradients_kpt(
                    wfs, kpt, grad_knG, dens)
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            n_occ = get_n_occ(kpt)
            # if n_occ == 0:
            #     continue
            grad_knG[k][n_occ:n_occ + 1] = \
                self.get_gradients_lumo(ham, wfs, kpt)
            lamb = wfs.gd.integrate(kpt.psit_nG[:n_occ + 1],
                                    grad_knG[k][:n_occ + 1],
                                    False)
            lamb = (lamb + lamb.T.conj()) / 2.0
            lamb = np.ascontiguousarray(lamb)
            wfs.gd.comm.sum(lamb)
            if 'SIC' in self.odd_parameters['name']:
                n_unocc = len(kpt.f_n) - (n_occ + 1)
                self.odd.lagr_diag_s[k] = np.append(
                    np.diagonal(lamb).real,
                    np.ones(shape=n_unocc) *
                    np.absolute(lamb[n_occ, n_occ] * 5.))
                # np.ones(shape=n_unocc) * np.inf)
                # inf is not a good
                # for example for ase get gap

                self.odd.lagr_diag_s[k][:n_occ] /= kpt.f_n[:n_occ]
            evals = np.empty(lamb.shape[0])
            diagonalize(lamb, evals)
            wfs.gd.comm.broadcast(evals, 0)
            wfs.gd.comm.broadcast(lamb, 0)
            kpt.eps_n[:n_occ + 1] = evals
            kpt.eps_n[:n_occ] = kpt.eps_n[:n_occ] / kpt.f_n[:n_occ]
            # kpt.eps_n[n_occ + 1:] = +np.inf
            # inf is not a good for example for ase get gap
            kpt.eps_n[n_occ + 1:] *= 0.0
            kpt.eps_n[n_occ + 1:] +=\
                np.absolute(5.0 * kpt.eps_n[n_occ])
            if rewrite_psi:
                # TODO:
                # Do we need sort wfs according to eps_n
                # or they will be automatically sorted?
                kpt.psit_nG[:n_occ + 1] = \
                    np.tensordot(lamb.conj(), kpt.psit_nG[:n_occ + 1],
                                 axes=1)
                for a in kpt.P_ani.keys():
                    kpt.P_ani[a][:n_occ + 1] = \
                        np.dot(lamb.conj(),
                               kpt.P_ani[a][:n_occ + 1])

        # update fermi level?
        occ.calculate(wfs)

    def get_gradients_lumo(self, ham, wfs, kpt):

        n_occ = 0
        for f in kpt.f_n:
            if f > 1.0e-10:
                n_occ += 1
        # calculate gradients:
        psi = kpt.psit_nG[n_occ:n_occ+1].copy()
        P1_ai = wfs.pt.dict(shape=1)
        wfs.pt.integrate(psi, P1_ai, kpt.q)
        Hpsi_nG = wfs.empty(1, q=kpt.q)
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

    def evaluate_phi_and_der_phi_lumo(self, psit_k, search_dir,
                                      alpha, ham, wfs,
                                      phi=None, grad_k=None):

        """
        phi = E(x_k + alpha_k*p_k)
        der_phi = grad_alpha E(x_k + alpha_k*p_k) cdot p_k
        :return:  phi, der_phi # floats
        """

        # TODO: the difference is only that we use
        #  ..._lumo and skip if kpt.f_n[i] > 1.0e-10:

        if phi is None or grad_k is None:
            x_knG = \
                {k: psit_k[k] +
                    alpha * search_dir[k] for k in psit_k.keys()}
            phi, grad_k = \
                self.get_energy_and_tangent_gradients_lumo(ham,
                                                           wfs, x_knG)
        der_phi = 0.0
        n_kps = self.n_kps
        for kpt in wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            for i, g in enumerate(grad_k[k]):
                der_phi += self.dot(
                    wfs, g, search_dir[k][i], kpt).item().real
        der_phi = wfs.kd.comm.sum(der_phi)

        return phi, der_phi, grad_k

    def get_energy_and_tangent_gradients_lumo(self,
                                              ham, wfs,
                                              psit_knG=None):

        n_kps = self.n_kps
        if psit_knG is not None:
            for kpt in wfs.kpt_u:
                k = n_kps * kpt.s + kpt.q
                # find lumo
                n_occ = get_n_occ(kpt)
                kpt.psit_nG[n_occ:n_occ+1] = psit_knG[k].copy()
                wfs.orthonormalize(kpt)
        elif not wfs.orthonormalized:
            wfs.orthonormalize()

        grad = {}
        energy_t = 0.0
        error_t = 0.0

        for kpt in wfs.kpt_u:
            n_occ = 0
            for f in kpt.f_n:
                if f > 1.0e-10:
                    n_occ += 1
            k = n_kps * kpt.s + kpt.q
            # calculate gradients:
            psi = kpt.psit_nG[n_occ:n_occ+1].copy()
            P1_ai = wfs.pt.dict(shape=1)
            wfs.pt.integrate(psi, P1_ai, kpt.q)
            Hpsi_nG = wfs.empty(1, q=kpt.q)
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
            grad[k] = Hpsi_nG.copy()

            # calculate energy
            if self.odd_parameters['name'] == 'Zero':
                energy = wfs.gd.integrate(
                    psi, Hpsi_nG, global_integral=True).item().real

                kpt.eps_n[n_occ:n_occ + 1] = energy
                # project gradients:
                s_axi = {}
                for a, P_xi in P1_ai.items():
                    dO_ii = wfs.setups[a].dO_ii
                    s_xi = np.dot(P_xi, dO_ii)
                    s_axi[a] = s_xi
                wfs.pt.add(psi, s_axi, kpt.q)
                grad[k] -= energy * psi
            else:
                lamb = np.zeros(shape=n_occ+1, dtype=self.dtype)
                pic = kpt.psit_nG[:n_occ + 1].copy()
                for z, zeta in enumerate(pic):
                    lamb[z] = \
                        wfs.gd.integrate(zeta, Hpsi_nG,
                                         global_integral=True).item()
                    wfs.pt.integrate(zeta, P1_ai, kpt.q)
                    s_axi = {}
                    for a, P_xi in P1_ai.items():
                        dO_ii = wfs.setups[a].dO_ii
                        s_xi = np.dot(P_xi, dO_ii)
                        s_axi[a] = s_xi
                    wfs.pt.add(zeta, s_axi, kpt.q)
                    x = lamb[z]
                    grad[k] -= x * zeta
                energy = lamb[-1]
            # kpt.eps_n[n_occ:n_occ + 1] = energy.real

            norm = self.dot(wfs, grad[k], grad[k], kpt).item().real
            error = norm * Hartree ** 2
            error_t += error
            energy_t += energy.real

        error_t = wfs.kd.comm.sum(error_t)
        energy_t = wfs.kd.comm.sum(energy_t)
        self.error = error_t

        return energy_t, grad

    def iterate_lumo(self, ham, wfs, dens):

        n_kps = self.n_kps
        psi_copy = {}
        alpha = self.alpha
        phi_2i = self.phi_2i
        der_phi_2i = self.der_phi_2i

        wfs.timer.start('Direct Minimisation step')

        if self.iters == 0:
            # calculate gradients
            phi_2i[0], grad_knG = \
                self.get_energy_and_tangent_gradients_lumo(ham, wfs)
            # self.error = self.error_eigv(wfs, grad_knG)
        else:
            grad_knG = self.grad_knG

        wfs.timer.start('Get Search Direction')
        for kpt in wfs.kpt_u:
            n_occ = get_n_occ(kpt)
            k = n_kps * kpt.s + kpt.q
            psi_copy[k] = kpt.psit_nG[n_occ:n_occ + 1].copy()
        p_knG = self.search_direction.update_data(psi_copy, grad_knG,
                                                  wfs, self.prec)
        # self.project_search_direction(wfs, p_knG)
        wfs.timer.stop('Get Search Direction')

        # recalculate derivative with new search direction
        der_phi_2i[0] = 0.0
        for kpt in wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            der_phi_2i[0] += \
                self.dot(wfs, grad_knG[k][0], p_knG[k][0],
                         kpt).item().real
        der_phi_2i[0] = wfs.kd.comm.sum(der_phi_2i[0])

        alpha, phi_alpha, der_phi_alpha, grad_knG = \
            self.line_search.step_length_update(
                psi_copy, p_knG, ham, wfs, dens,
                phi_0=phi_2i[0], der_phi_0=der_phi_2i[0],
                phi_old=phi_2i[1], der_phi_old=der_phi_2i[1],
                alpha_max=3.0, alpha_old=alpha)
        # calculate new wfs:
        for kpt in wfs.kpt_u:
            n_occ = get_n_occ(kpt)
            k = n_kps * kpt.s + kpt.q
            kpt.psit_nG[n_occ:n_occ + 1] = \
                psi_copy[k] + alpha * p_knG[k]
            wfs.orthonormalize(kpt)

        self.alpha = alpha
        self.grad_knG = grad_knG
        self.iters += 1

        # and 'shift' phi, der_phi for the next iteration
        phi_2i[1], der_phi_2i[1] = phi_2i[0], der_phi_2i[0]
        phi_2i[0], der_phi_2i[0] = phi_alpha, der_phi_alpha,

        wfs.timer.stop('Direct Minimisation step')
        return phi_alpha, self.error

    def run_lumo(self, ham, wfs, dens, occ, max_err, log):

        self.need_init_odd = False
        self.initialize_dm(
            wfs, dens, ham,
            obj_func=self.evaluate_phi_and_der_phi_lumo, lumo=True)

        max_iter = 3000
        while self.iters < max_iter:
            en, er = self.iterate_lumo(ham, wfs, dens)
            log_f(self.iters, en, er, log)
            # it is quite difficult to converge lumo with the same
            # accuaracy as occupaied states.
            if er < max_err * 10.:
                break
        log('\nLUMO converged after'
            ' {:d} iterations'.format(self.iters))
        self.initialized = False

    def run_inner_loop(self, ham, wfs, occ, dens, log, niter=0):

        if self.iloop is None:
            return niter, False

        wfs.timer.start('Inner loop')

        psi_copy = {}
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            psi_copy[k] = kpt.psit_nG[:].copy()

        e_total = ham.get_energy(occ,
                                 kin_en_using_band=False,
                                 e_sic=self.e_sic)
        log = parprint

        counter = self.iloop.run(
            e_total - self.e_sic, psi_copy, wfs, dens, log, niter)
        del psi_copy

        wfs.timer.stop('Inner loop')

        return counter, True

    def init_wfs(self, wfs, ham, occ, log):
        # initial orbitals can be localised using Pipek-Mezey
        # or Wannier functions.

        if not self.need_init_orbs or wfs.read_from_file:
            return

        log("Initial Localization: ...", flush=True)
        wfs.timer.start('Initial Localization')

        # we need to fill also eps_n
        for kpt in wfs.kpt_u:
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
            super(DirectMinFD, self).subspace_diagonalize(
                ham, wfs, kpt, True)
            wfs.gd.comm.broadcast(kpt.eps_n, 0)
        occ.calculate(wfs)  # fill occ numbers

        for kpt in wfs.kpt_u:
            if sum(kpt.f_n) < 1.0e-3:
                continue
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

            if self.initial_orbitals == 'KS' or \
                    self.initial_orbitals is None:
                self.need_init_orbs = False
                break
            elif self.initial_orbitals == 'PM':
                lf_obj = PipekMezey(
                    wfs=wfs, spin=kpt.s, dtype=wfs.dtype)
            elif self.initial_orbitals == 'W':
                lf_obj = WannierLocalization(
                    wfs=wfs, spin=kpt.s)
            else:
                raise ValueError('Check initial orbitals.')
            lf_obj.localize(tolerance=1.0e-8)
            if self.initial_orbitals == 'PM':
                U = np.ascontiguousarray(
                    lf_obj.W_k[kpt.q].T)
            else:
                U = np.ascontiguousarray(
                    lf_obj.U_kww[kpt.q].T)
                if kpt.psit_nG.dtype == float:
                    U = U.real
            wfs.gd.comm.broadcast(U, 0)
            dim = U.shape[0]
            kpt.psit_nG[:dim] = np.einsum('ij,jkml->ikml',
                                          U, kpt.psit_nG[:dim])
            del lf_obj

        self.need_init_orbs = False
        wfs.timer.stop('Initial Localization')
        log("Done", flush=True)


def log_f(niter, e_total, eig_error, log):

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
