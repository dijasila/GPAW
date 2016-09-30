import numpy as np

from gpaw.utilities.blas import gemm
from gpaw.blacs import BlacsGrid


def reshape(a_x, shape):
    """Get an ndarray of size shape from a_x buffer."""
    return a_x.ravel()[:np.prod(shape)].reshape(shape)


class MatrixOperator:
    def __init__(self, ksl):
        self.bd = ksl.bd
        self.gd = ksl.gd
        self.bmd = ksl.new_descriptor()  # XXX take hermitian as argument?
        self.dtype = ksl.dtype
        self.nblocks = 1
        self.A_nn = None
        self.work1_xG = None
        self.grid = BlacsGrid(self.bd.comm, self.bd.comm.size, 1)
        G = self.gd.n_c.prod()
        N = self.bd.nbands
        self.md1 = self.grid.new_descriptor(N, G, self.bd.maxmynbands, G)
        self.md2 = self.grid.new_descriptor(N, N, N, N)

    def calculate_matrix_elements(self, psit1_nG, P1_ani, A, dA,
                                  psit2_nG=None, P2_ani=None):
        band_comm = self.bd.comm
        domain_comm = self.gd.comm
        #block_comm = self.block_comm

        B = band_comm.size
        J = self.nblocks
        N = self.bd.mynbands
        M = int(np.ceil(N / float(J)))

        if psit2_nG is None:
            psit2_nG = psit1_nG
            hermitian = True
        else:
            hermitian = False

        if P2_ani is None:
            P2_ani = P1_ani

        if self.A_nn is None:
            self.A_nn = np.empty((N, N), self.dtype)

        A_NN = self.A_nn


        Apsit_nG = A(psit2_nG)
        from gpaw.utilities.scalapack import pblas_gemm
        M = self.bd.mynbands
        G = self.gd.n_c.prod()
        pblas_gemm(self.gd.dv,
                   psit1_nG.reshape(M, G),
                   Apsit_nG.reshape(M, G),
                   0.0, A_NN, self.md1, self.md1, self.md2, 'N', 'T')
        print(A_NN)
        self.gd.integrate(psit1_nG, Apsit_nG, hermitian=hermitian,
                          _transposed_result=A_NN)
        print(A_NN);asdg
        for a, P1_ni in P1_ani.items():
            P2_ni = P2_ani[a]
            gemm(1.0, P1_ni, dA(a, P2_ni), 1.0, A_NN, 'c')
        domain_comm.sum(A_NN, 0)
        return self.bmd.redistribute_output(A_NN)

        dfgjkh

        domain_comm.sum(A_qnn, 0)

        if B == 1:
            return self.bmd.redistribute_output(A_NN)

        if domain_comm.rank == 0:
            self.bmd.assemble_blocks(A_qnn, A_NN, hermitian)

        # Because of the amount of communication involved, we need to
        # be syncronized up to this point.
        block_comm.barrier()
        return self.bmd.redistribute_output(A_NN)

    def matrix_multiply(self, C_NN, psit_nG, P_ani=None, out_nG=None):
        """Calculate new linear combinations of wave functions.

        Results will be put in the *P_ani* dict and a new psit_nG returned::

                     __                                __
            ~       \       ~           ~a  ~         \       ~a  ~
           psi  <--  ) C   psi    and  <p |psi >  <--  ) C   <p |psi >
              n     /__ nn'   n'         i    n       /__ nn'  i    n'
                     n'                                n'


        Parameters:

        C_NN: ndarray
            Matrix representation of the requested linear combinations. Even
            with a hermitian operator, this matrix need not be self-adjoint.
            However, unlike the results from calculate_matrix_elements, it is
            assumed that all matrix elements are filled in (use e.g. tri2full).
        psit_nG: ndarray
            Set of vectors in which the matrix elements are evaluated.
        P_ani: dict
            Dictionary of projector overlap integrals P_ni = <p_i | psit_nG>.

        """

        N = self.bd.mynbands

        if self.work1_xG is None:
            self.work1_xG = self.gd.empty(N, self.dtype)

        work_nG = reshape(self.work1_xG, psit_nG.shape)
        if out_nG is None:
            out_nG = work_nG
            out_nG[:] = 117  # gemm may not like nan's
        elif out_nG is psit_nG:
            work_nG[:] = psit_nG
            psit_nG = work_nG
        
        from gpaw.matrix import Matrix
        psit_n = Matrix(psit_nG, self.gd)
        out_n = Matrix(out_nG, self.gd)
        C_nn = Matrix(C_NN)
        P_n = Matrix(P_ani)
        P2_n = P_n.empty_like()
        
        out_n[:] = C_nn * psit_n
        P2_n[:] = C_nn * P_n
        P2_n.extract(P_ani)
        
        return out_nG
