"""Module defining an eigensolver base-class."""

import numpy as np
from ase.utils.timing import timer

from gpaw.utilities.blas import axpy
from gpaw.utilities import unpack
from gpaw.xc.hybrid import HybridXC


def reshape(a_x, shape):
    """Get an ndarray of size shape from a_x buffer."""
    return a_x.ravel()[:np.prod(shape)].reshape(shape)


class Eigensolver:
    def __init__(self, keep_htpsit=True, blocksize=1):
        self.keep_htpsit = keep_htpsit
        self.initialized = False
        self.Htpsit_nG = None
        self.error = np.inf
        self.blocksize = blocksize
        self.orthonormalization_required = True

    def initialize(self, wfs):
        self.timer = wfs.timer
        self.world = wfs.world
        self.kpt_comm = wfs.kd.comm
        self.band_comm = wfs.bd.comm
        self.dtype = wfs.dtype
        self.bd = wfs.bd
        self.nbands = wfs.bd.nbands
        self.mynbands = wfs.bd.mynbands

        if wfs.bd.comm.size > 1:
            self.keep_htpsit = False

        if self.keep_htpsit:
            self.Htpsit_nG = wfs.empty(self.nbands)

        # Preconditioner for the electronic gradients:
        self.preconditioner = wfs.make_preconditioner(self.blocksize)

        for kpt in wfs.kpt_u:
            if kpt.eps_n is None:
                kpt.eps_n = np.empty(self.mynbands)

        self.initialized = True

    def reset(self):
        self.initialized = False

    def iterate(self, ham, wfs):
        """Solves eigenvalue problem iteratively

        This method is inherited by the actual eigensolver which should
        implement *iterate_one_k_point* method for a single iteration of
        a single kpoint.
        """

        if not self.initialized:
            if isinstance(ham.xc, HybridXC):
                self.blocksize = wfs.bd.mynbands
            self.initialize(wfs)

        error = 0.0
        for kpt in wfs.kpt_u:
            if not wfs.orthonormalized:
                wfs.orthonormalize(kpt)
            e = self.iterate_one_k_point(ham, wfs, kpt)
            error += e
            if self.orthonormalization_required:
                wfs.orthonormalize(kpt)

        wfs.orthonormalized = True
        self.error = self.band_comm.sum(self.kpt_comm.sum(error))

    def iterate_one_k_point(self, ham, kpt):
        """Implemented in subclasses."""
        raise NotImplementedError

    def calculate_residuals(self, kpt, wfs, hamiltonian, psit_xG, P_axi, eps_x,
                            R_xG, n_x=None, calculate_change=False):
        """Calculate residual.

        From R=Ht*psit calculate R=H*psit-eps*S*psit."""

        for R_G, eps, psit_G in zip(R_xG, eps_x, psit_xG):
            axpy(-eps, psit_G, R_G)

        c_axi = {}
        for a, P_xi in P_axi.items():
            dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
            dO_ii = hamiltonian.setups[a].dO_ii
            c_xi = (np.dot(P_xi, dH_ii) -
                    np.dot(P_xi * eps_x[:, np.newaxis], dO_ii))
            c_axi[a] = c_xi
        hamiltonian.xc.add_correction(kpt, psit_xG, R_xG, P_axi, c_axi, n_x,
                                      calculate_change)
        wfs.pt.add(R_xG, c_axi, kpt.q)

    @timer('Subspace diag')
    def subspace_diagonalize(self, ham, wfs, kpt):
        """Diagonalize the Hamiltonian in the subspace of kpt.psit_nG

        *Htpsit_nG* is a work array of same size as psit_nG which contains
        the local part of the Hamiltonian times psit on exit

        First, the Hamiltonian (defined by *kin*, *vt_sG*, and
        *dH_asp*) is applied to the wave functions, then the *H_nn*
        matrix is calculated and diagonalized, and finally, the wave
        functions (and also Htpsit_nG are rotated.  Also the
        projections *P_ani* are rotated.

        It is assumed that the wave functions *psit_nG* are orthonormal
        and that the integrals of projector functions and wave functions
        *P_ani* are already calculated.

        Return rotated wave functions and H applied to the rotated
        wave functions if self.keep_htpsit is True.
        """

        if self.band_comm.size > 1 and wfs.bd.strided:
            raise NotImplementedError

        psit_n = kpt.psit_n
        tmp_n = psit_n.new(buf=wfs.work_array_nG)
        H_nn = wfs.M_nn
        P_nI = kpt.P_n
        dHP_nI = P_nI.new()

        def Ht(psit_n, Htpsit_n):
            wfs.apply_pseudo_hamiltonian(kpt, ham, psit_n.data, Htpsit_n.data)

        dH_II = P_nI.paw_matrix(unpack(ham.dH_asp[a][kpt.s])
                                for a in kpt.P_ani)

        with self.timer('calc_h_matrix'):
            Ht(psit_n, tmp_n)
            psit_n.matrix_elements(tmp_n, H_nn, hermitian=True)
            dHP_nI[:] = P_nI * dH_II
            H_nn += P_nI.C * dHP_nI.T
            ham.xc.correct_hamiltonian_matrix(kpt, H_nn.data)

        with wfs.timer('diagonalize'):
            H_nn.eigh(kpt.eps_n)
            # H_nn now contains the eigenvectors

        with self.timer('rotate_psi'):
            if self.keep_htpsit:
                Htpsit_n = psit_n.new(buf=self.Htpsit_nG)
                Htpsit_n[:] = H_nn.T * tmp_n
            tmp_n[:] = H_nn.T * psit_n
            psit_n[:] = tmp_n
            dHP_nI[:] = H_nn.T * P_nI
            dHP_nI.extract_to(kpt.P_ani)
            # Rotate orbital dependent XC stuff:
            ham.xc.rotate(kpt, H_nn.data)

        if self.keep_htpsit:
            return kpt.psit_nG, Htpsit_n.data
        else:
            return kpt.psit_nG, None

    def estimate_memory(self, mem, wfs):
        gridmem = wfs.bytes_per_wave_function()

        keep_htpsit = self.keep_htpsit and (wfs.bd.mynbands == wfs.bd.nbands)

        if keep_htpsit:
            mem.subnode('Htpsit', wfs.bd.nbands * gridmem)
        else:
            mem.subnode('No Htpsit', 0)

        mem.subnode('eps_n', wfs.bd.mynbands * mem.floatsize)
        mem.subnode('eps_N', wfs.bd.nbands * mem.floatsize)
        mem.subnode('Preconditioner', 4 * gridmem)
        mem.subnode('Work', gridmem)
