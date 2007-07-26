"""Module defining  ``Eigensolver`` classes."""

import Numeric as num
from multiarray import innerproduct as inner # avoid the dotblas version!

import LinearAlgebra as linalg
from gpaw.operators import Laplace
from gpaw.preconditioner import Preconditioner
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.blas import axpy, r2k, gemm
from gpaw.utilities.complex import cc, real
from gpaw.utilities.tools import apply_subspace_mask
from gpaw.utilities import unpack


class Eigensolver:
    def __init__(self, paw, nbands=None):
        self.timer = paw.timer
        self.kpt_comm = paw.kpt_comm
        self.typecode = paw.typecode
        self.gd = paw.gd
        self.comm = paw.gd.comm
        if nbands is None:
            self.nbands = paw.nbands
        else:
            self.nbands = nbands
        self.convergeall = paw.input_parameters['convergeall']

        # Preconditioner for the electronic gradients:
        self.preconditioner = Preconditioner(self.gd, paw.hamiltonian.kin,
                                             self.typecode)

        # Soft part of the Hamiltonian times psit
        self.Htpsit_nG = self.gd.empty(self.nbands, self.typecode)

        # Work array for e.g. subspace rotations
        self.work = self.gd.empty(self.nbands, self.typecode)

        # Hamiltonian matrix
        self.H_nn = num.empty((self.nbands, self.nbands), self.typecode)

    def iterate(self, hamiltonian, kpt_u):
        """Solves eigenvalue problem iteratively

        This method is inherited by the actual eigensolver which should
        implement ``iterate_one_k_point`` method for a single iteration of
        a single kpoint.
        """

        error = 0.0
        for kpt in kpt_u:
            error += self.iterate_one_k_point(hamiltonian, kpt)
            
        self.error = self.comm.sum(self.kpt_comm.sum(error))

    def iterate_one_k_point(self, hamiltonian, kpt):
        """Implemented in subclasses."""
        return 0.0
    
    def diagonalize(self, hamiltonian, kpt):
        """Diagonalize the Hamiltonian in the subspace of kpt.psit_nG

        ``Htpsit_nG`` is working array of same size as psit_nG which contains
        the local part of the Hamiltonian times psit on exit

        First, the Hamiltonian (defined by ``kin``, ``vt_sG``, and
        ``my_nuclei``) is applied to the wave functions, then the
        ``H_nn`` matrix is calculated and diagonalized, and finally,
        the wave functions are rotated.  Also the projections
        ``P_uni`` (an attribute of the nuclei) are rotated.
        
        It is assumed that the wave functions ``psit_n`` are orthonormal
        and that the integrals of projector functions and wave functions
        ``P_uni`` are already calculated
        """        
           
        self.timer.start('Subspace diag.')

        if self.nbands != kpt.nbands:
            raise RuntimeError('Bands: %d != %d' % (self.nbands, kpt.nbands))
        
        Htpsit_nG = self.Htpsit_nG
        psit_nG = kpt.psit_nG
        eps_n = kpt.eps_n
        H_nn = self.H_nn

        hamiltonian.kin.apply(psit_nG, Htpsit_nG, kpt.phase_cd)
            
        Htpsit_nG += psit_nG * hamiltonian.vt_sG[kpt.s]

        H_nn[:] = 0.0  # r2k fails without this!
        
        self.timer.start('Non-local xc')
        hamiltonian.xc.xcfunc.apply_non_local(kpt, Htpsit_nG, H_nn)
        self.timer.stop('Non-local xc')
        
        r2k(0.5 * self.gd.dv, psit_nG, Htpsit_nG, 1.0, H_nn)
        
        for nucleus in hamiltonian.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            H_nn += num.dot(P_ni, num.dot(unpack(nucleus.H_sp[kpt.s]),
                                          cc(num.transpose(P_ni))))

        self.comm.sum(H_nn, kpt.root)

        # Uncouple occupied and unoccupied subspaces
        if hamiltonian.xc.xcfunc.hybrid > 0.0:
            apply_subspace_mask(H_nn, kpt.f_n)

        self.timer.start('dsyev/zheev')
        if self.comm.rank == kpt.root:
            info = diagonalize(H_nn, eps_n)
            if info != 0:
                raise RuntimeError, 'Very Bad!!'
        self.timer.stop('dsyev/zheev')

        self.timer.start('bcast H')
        self.comm.broadcast(H_nn, kpt.root)
        self.timer.stop('bcast H')
        self.timer.start('bcast eps')
        self.comm.broadcast(eps_n, kpt.root)
        self.timer.stop('bcast eps')

        # Rotate psit_nG:
        gemm(1.0, psit_nG, H_nn, 0.0, self.work)
        
        # Rotate Htpsit_nG:
        gemm(1.0, Htpsit_nG, H_nn, 0.0, psit_nG)

        #Switch the references
        kpt.psit_nG, self.Htpsit_nG, self.work = self.work, psit_nG, Htpsit_nG
        
        # Rotate P_uni:
        for nucleus in hamiltonian.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            gemm(1.0, P_ni.copy(), H_nn, 0.0, P_ni)

        # Rotate EXX related stuff
        if hamiltonian.xc.xcfunc.hybrid > 0.0:
            hamiltonian.xc.xcfunc.exx.rotate(kpt.u, H_nn)

        self.timer.stop('Subspace diag.')
