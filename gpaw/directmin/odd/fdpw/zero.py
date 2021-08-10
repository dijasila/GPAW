"""
Potentials for orbital density dependent energy functionals
"""

import numpy as np
from gpaw.utilities import unpack
from gpaw.directmin.fdpw.tools import d_matrix


class ZeroCorrections:

    """
    KS-DFT 
    """
    def __init__(self, wfs, dens, ham, scaling_factor=(1.0, 1.0),
                 sic_coarse_grid=True, store_potentials=False,
                 poisson_solver='FPS'):

        self.name = 'Zero'
        self.n_kps = wfs.kd.nibzkpts
        self.grad = {}
        self.total_sic = 0.0
        self.eks = 0.0
        self.changedocc = 0
        self.restart = 0
        self.momevery = 20
        self.momcounter = 0

    def get_energy_and_gradients(self, wfs, grad_knG=None,
                                 dens=None, U_k=None,
                                 add_grad=False,
                                 ham=None):

        wfs.timer.start('Update Kohn-Sham energy')
        # calc projectors
        for kpt in wfs.kpt_u:
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

        if self.momcounter % self.momevery == 0:
            f_sn = {}
            for kpt in wfs.kpt_u:
                n_kps = wfs.kd.nibzkpts
                u = n_kps * kpt.s + kpt.q
                f_sn[u] = kpt.f_n.copy()
            self._e_entropy = wfs.calculate_occupation_numbers(dens.fixed)
            self.changedocc = 0
            for kpt in wfs.kpt_u:
                n_kps = wfs.kd.nibzkpts
                u = n_kps * kpt.s + kpt.q
                self.changedocc = int(
                    not np.allclose(f_sn[u], kpt.f_n.copy()))
            self.changedocc = wfs.kd.comm.max(self.changedocc)
            occ_name = getattr(wfs.occupations, 'name', None)
            if occ_name == 'mom':
                for kpt in wfs.kpt_u:
                    wfs.eigensolver.sort_wavefunctions(wfs, kpt)
            if self.changedocc:
                self.restart = 1
        self.momcounter += 1

        dens.update(wfs)
        ham.update(dens, wfs, False)
        wfs.timer.stop('Update Kohn-Sham energy')
        energy = ham.get_energy(0.0, wfs, False)

        for kpt in wfs.kpt_u:
            self.get_energy_and_gradients_kpt(
                wfs, kpt, grad_knG, dens, U_k,
                add_grad=add_grad, ham=ham)
        # energy = wfs.kd.comm.sum(energy)
        self.eks = energy
        return energy

    def get_energy_and_gradients_kpt(self, wfs, kpt, grad_knG,
                                     dens=None, U_k=None,
                                     add_grad=False,
                                     ham=None):

        k = self.n_kps * kpt.s + kpt.q
        nbands = wfs.bd.nbands
        if U_k is not None:
            assert U_k[k].shape[0] == nbands

        wfs.timer.start('e/g grid calculations')
        self.grad[k] = wfs.empty(nbands, q=kpt.q)
        wfs.apply_pseudo_hamiltonian(kpt, ham, kpt.psit_nG, self.grad[k])

        c_axi = {}
        for a, P_xi in kpt.P_ani.items():
            dH_ii = unpack(ham.dH_asp[a][kpt.s])
            c_xi = np.dot(P_xi, dH_ii)
            c_axi[a] = c_xi

        # not sure about this:
        ham.xc.add_correction(kpt, kpt.psit_nG, self.grad[k],
                              kpt.P_ani, c_axi, n_x=None,
                              calculate_change=False)
        # add projectors to the H|psi_i>
        wfs.pt.add(self.grad[k], c_axi, kpt.q)
        # scale with occupation numbers
        for i, f in enumerate(kpt.f_n):
            self.grad[k][i] *= f

        if add_grad:
            if U_k is not None:
                grad_knG[k] += \
                    np.tensordot(U_k[k].conj(), self.grad[k], axes=1)
            else:
                grad_knG[k] += self.grad[k]
        else:
            if U_k is not None:
                self.grad[k][:] = np.tensordot(U_k[k].conj(),
                                               self.grad[k], axes=1)

        wfs.timer.stop('e/g grid calculations')

        return 0.0

    def get_energy_and_gradients_inner_loop(self, wfs, a_mat,
                                            evals, evec, dens,
                                            ham):
        nbands = wfs.bd.nbands
        e_sic = self.get_energy_and_gradients(wfs,
                                              grad_knG=None,
                                              dens=dens, U_k=None,
                                              add_grad=False,
                                              ham=ham)
        wfs.timer.start('Unitary gradients')
        g_k = {}
        kappa_tmp = 0.0
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            l_odd = wfs.integrate(kpt.psit_nG, self.grad[k], True)
            # l_odd = np.ascontiguousarray(l_odd)
            # wfs.gd.comm.sum(l_odd)
            f = np.ones(nbands)
            indz = np.absolute(l_odd) > 1.0e-4
            l_c = 2.0 * l_odd[indz]
            l_odd = f[:, np.newaxis] * l_odd.T.conj() - f * l_odd
            kappa = np.max(np.absolute(l_odd[indz]) / np.absolute(l_c))
            if kappa > kappa_tmp:
                kappa_tmp = kappa
            if a_mat[k] is None:
                g_k[k] = l_odd.T
            else:
                g_mat = evec[k].T.conj() @ l_odd.T.conj() @ evec[k]
                g_mat = g_mat * d_matrix(evals[k])
                g_mat = evec[k] @ g_mat @ evec[k].T.conj()
                for i in range(g_mat.shape[0]):
                    g_mat[i][i] *= 0.5
                if a_mat[k].dtype == float:
                    g_mat = g_mat.real
                g_k[k] = 2.0 * g_mat

        kappa = wfs.kd.comm.max(kappa_tmp)
        wfs.timer.stop('Unitary gradients')
        return g_k, e_sic, kappa
