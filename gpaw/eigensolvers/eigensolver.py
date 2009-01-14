"""Module defining and eigensolver base-class."""

from math import ceil

import numpy as np

from gpaw.operators import Laplace
from gpaw.preconditioner import Preconditioner
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.blas import axpy, r2k, gemm
from gpaw.utilities.tools import apply_subspace_mask
from gpaw.utilities import unpack
from gpaw.mpi import run, parallel
from gpaw.utilities import scalapack
from gpaw import sl_diagonalize
from gpaw import debug


class Eigensolver:
    def __init__(self, keep_htpsit=True):
        self.keep_htpsit = keep_htpsit
        self.initialized = False
        self.Htpsit_nG = None
        self.error = np.inf
        
    def initialize(self, wfs):
        self.timer = wfs.timer
        self.kpt_comm = wfs.kpt_comm
        self.band_comm = wfs.band_comm
        self.dtype = wfs.dtype
        self.gd = wfs.gd
        self.comm = wfs.gd.comm
        self.nbands = wfs.nbands
        self.mynbands = wfs.mynbands

        if self.mynbands != self.nbands:
            self.keep_htpsit = False

        self.eps_n = np.empty(self.nbands)

        # Preconditioner for the electronic gradients:
        self.preconditioner = Preconditioner(self.gd, wfs.kin, self.dtype)

        if self.keep_htpsit:
            # Soft part of the Hamiltonian times psit:
            self.Htpsit_nG = self.gd.empty(self.nbands, self.dtype)

        # Hamiltonian matrix
        self.H_nn = np.empty((self.nbands, self.nbands), self.dtype)

        for kpt in wfs.kpt_u:
            if kpt.eps_n is None:
                kpt.eps_n = np.empty(self.mynbands)

        self.operator = wfs.overlap.operator
        
        self.initialized = True

    def iterate(self, hamiltonian, wfs):
        """Solves eigenvalue problem iteratively

        This method is inherited by the actual eigensolver which should
        implement *iterate_one_k_point* method for a single iteration of
        a single kpoint.
        """

        if not self.initialized:
            self.initialize(wfs)

        if not wfs.orthonormalized:
            wfs.orthonormalize()
            
        error = 0.0
        for kpt in wfs.kpt_u:
            error += self.iterate_one_k_point(hamiltonian, wfs, kpt)

        wfs.orthonormalize()

        self.error = self.band_comm.sum(self.kpt_comm.sum(error))

    def iterate_one_k_point(self, hamiltonian, kpt):
        """Implemented in subclasses."""
        raise NotImplementedError

    def calculate_residuals(self, wfs, hamiltonian, kpt, eps_n, R_nG, psit_nG):
        B = len(eps_n)  # block size
        wfs.kin.apply(psit_nG, R_nG, kpt.phase_cd)
        hamiltonian.apply_local_potential(psit_nG, R_nG, kpt.s)
        P_ani = dict([(a, np.zeros((B, wfs.setups[a].ni), wfs.dtype))
                      for a in kpt.P_ani])
        wfs.pt.integrate(psit_nG, P_ani, kpt.q)
        self.calculate_residuals2(wfs, hamiltonian, kpt, R_nG,
                                  eps_n, psit_nG, P_ani)
        
    def calculate_residuals2(self, wfs, hamiltonian, kpt, R_nG,
                             eps_n=None, psit_nG=None, P_ani=None):
        if eps_n is None:
            eps_n = kpt.eps_n
        if psit_nG is None:
            psit_nG = kpt.psit_nG
        if P_ani is None:
            P_ani = kpt.P_ani
        for R_G, eps, psit_G in zip(R_nG, eps_n, psit_nG):
            axpy(-eps, psit_G, R_G)
        c_ani = {}
        for a, P_ni in P_ani.items():
            dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
            dO_ii = hamiltonian.setups[a].O_ii
            c_ni = (np.dot(P_ni, dH_ii) -
                    np.dot(P_ni * eps_n[:, np.newaxis], dO_ii))
            c_ani[a] = c_ni
        wfs.pt.add(R_nG, c_ani, kpt.q)

    def calculate_hamiltonian_matrix(self, hamiltonian, wfs, kpt):
        XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        """Set up the Hamiltonian in the subspace of kpt.psit_nG

        *Htpsit_nG* is a work array of same size as psit_nG which contains
        the local part of the Hamiltonian times psit on exit

        The Hamiltonian (defined by *kin*, *vt_sG*, and
        *my_nuclei*) is applied to the wave functions, then the
        *H_nn* matrix is calculated.

        It is assumed that the wave functions *psit_n* are orthonormal
        and that the integrals of projector functions and wave functions
        *P_uni* are already calculated
        """

        if self.band_comm.size > 1:
            return self.calculate_hamiltonian_matrix2(hamiltonian, kpt)

        psit_nG = kpt.psit_nG
        H_nn = self.H_nn
        H_nn[:] = 0.0  # r2k can fail without this!

        if self.keep_htpsit:
            Htpsit_nG = self.Htpsit_nG

            wfs.kin.apply(psit_nG, Htpsit_nG, kpt.phase_cd)
            hamiltonian.apply_local_potential(psit_nG, Htpsit_nG, kpt.s)

            hamiltonian.xc.xcfunc.apply_non_local(kpt, Htpsit_nG, H_nn)

            r2k(0.5 * self.gd.dv, psit_nG, Htpsit_nG, 1.0, H_nn)
        else:
            Htpsit_nG = self.big_work_arrays['work_nG']
            n1 = 0
            while n1 < self.nbands:
                n2 = n1 + self.blocksize
                if n2 > self.nbands:
                    n2 = self.nbands
                hamiltonian.apply(psit_nG[n1:n2], Htpsit_nG[:n2 - n1], kpt,
                                  local_part_only=True)

                gemm(self.gd.dv, Htpsit_nG, psit_nG[n1:], 0.0,
                     H_nn[n1:, n1:n2], 'c')
                n1 = n2

        for a, P_ni in kpt.P_ani.items():
            dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
            H_nn += np.dot(P_ni, np.inner(dH_ii, P_ni.conj()))

        self.comm.sum(H_nn, 0)

        # Uncouple occupied and unoccupied subspaces:
        if hamiltonian.xc.xcfunc.hybrid > 0.0:
            apply_subspace_mask(H_nn, kpt.f_n)

    def subspace_diagonalize(self, hamiltonian, wfs, kpt):
        """Diagonalize the Hamiltonian in the subspace of kpt.psit_nG

        *Htpsit_nG* is a work array of same size as psit_nG which contains
        the local part of the Hamiltonian times psit on exit

        First, the Hamiltonian (defined by *kin*, *vt_sG*, and
        *my_nuclei*) is applied to the wave functions, then the
        *H_nn* matrix is calculated and diagonalized, and finally,
        the wave functions are rotated.  Also the projections
        *P_uni* (an attribute of the nuclei) are rotated.

        It is assumed that the wave functions *psit_n* are orthonormal
        and that the integrals of projector functions and wave functions
        *P_uni* are already calculated.
        """

        self.timer.start('Subspace diag.')

        psit_nG = kpt.psit_nG
        P_ani = kpt.P_ani

        if self.keep_htpsit:
            Htpsit_xG = self.Htpsit_nG
        else:
            Htpsit_xG = self.operator.work1_xG
            
        def H(psit_xG):
            wfs.kin.apply(psit_xG, Htpsit_xG, kpt.phase_cd)
            hamiltonian.apply_local_potential(psit_xG, Htpsit_xG, kpt.s)
            #hamiltonian.xc.xcfunc.apply_non_local(kpt, Htpsit_xG, H_nn)
            return Htpsit_xG
        
        dH_aii = dict([(a, unpack(dH_sp[kpt.s]))
                       for a, dH_sp in hamiltonian.dH_asp.items()])

        H_nn = self.operator.calculate_matrix_elements(psit_nG, P_ani,
                                                       H, dH_aii)

        if sl_diagonalize:
            assert parallel
            assert scalapack()
            dsyev_zheev_string = 'Subspace diag.: '+'pdsyevd/pzheevd'
        else:
            dsyev_zheev_string = 'Subspace diag.: '+'dsyev/zheev'

        self.timer.start(dsyev_zheev_string)
        if sl_diagonalize:
            print 'Should H_nn be broadcast first?'
            info = diagonalize(H_nn, self.eps_n, root=0)
            if info != 0:
                raise RuntimeError('Failed to diagonalize: info=%d' % info)
        else:
            if self.comm.rank == 0:
                if self.band_comm.rank == 0:
                    info = diagonalize(H_nn, self.eps_n)
                    if info != 0:
                        raise RuntimeError('Failed to diagonalize: info=%d' % info)
        self.timer.stop(dsyev_zheev_string)

        if self.comm.rank == 0:
            self.band_comm.scatter(self.eps_n, kpt.eps_n, 0)
            self.band_comm.broadcast(H_nn, 0)

        U_nn = H_nn
        del H_nn

        self.comm.broadcast(U_nn, 0)
        self.comm.broadcast(kpt.eps_n, 0)

        kpt.psit_nG = self.operator.matrix_multiply(U_nn, psit_nG, P_ani)
        if self.keep_htpsit:
            self.Htpsit_nG = self.operator.matrix_multiply(U_nn, Htpsit_xG)

        # Rotate EXX related stuff
        if hamiltonian.xc.xcfunc.hybrid > 0.0:
            hamiltonian.xc.xcfunc.exx.rotate(kpt.u, U_nn)

        self.timer.stop('Subspace diag.')
