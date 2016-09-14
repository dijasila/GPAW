import numpy as np

from gpaw.utilities.blas import gemm


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

    def calculate_matrix_elements(self, psit1_nG, P1_ani, A, dA,
                                  psit2_nG=None, P2_ani=None):
        """Calculate matrix elements for A-operator.

        Results will be put in the *A_nn* array::

                                    ___
                    ~ 2  ^  ~ 1     \     ~   ~a    a   ~a  ~
           A    = <psi  |A|psi  > +  )  <psi |p > dA   <p |psi >
            nn'       n      n'     /___    n  i    ii'  i'   n'
                                     aii'

        Fills in the lower part of *A_nn*, but only on domain and band masters.


        Parameters:

        psit_nG: ndarray
            Set of vectors in which the matrix elements are evaluated.
        P_ani: dict
            Dictionary of projector overlap integrals P_ni = <p_i | psit_nG>.
        A: function
            Functional form of the operator A which works on psit_nG.
            Must accept and return an ndarray of the same shape as psit_nG.
        dA: function
            Operator which works on | phi_i >.  Must accept atomic
            index a and P_ni and return an ndarray with the same shape
            as P_ni, thus representing P_ni multiplied by dA_ii.

        """
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

        if B == 1 and J == 1:
            # Simple case:
            Apsit_nG = A(psit2_nG)
            self.gd.integrate(psit1_nG, Apsit_nG, hermitian=hermitian,
                              _transposed_result=A_NN)
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

        band_comm = self.bd.comm
        B = band_comm.size
        J = self.nblocks

        C_NN = self.bmd.redistribute_input(C_NN)

        if B == 1 and J == 1:
            # Simple case:
            work_nG = reshape(self.work1_xG, psit_nG.shape)
            if out_nG is None:
                out_nG = work_nG
                out_nG[:] = 117  # gemm may not like nan's
            elif out_nG is psit_nG:
                work_nG[:] = psit_nG
                psit_nG = work_nG
            self.gd.gemm(1.0, psit_nG, C_NN, 0.0, out_nG)
            if P_ani:
                for P_ni in P_ani.values():
                    gemm(1.0, P_ni.copy(), C_NN, 0.0, P_ni)
            return out_nG

        asdfhjg
        # Now it gets nasty! We parallelize over B groups of bands and
        # each grid chunk is divided in J smaller slices (less memory).

        Q = B  # always non-hermitian XXX
        rank = band_comm.rank
        shape = psit_nG.shape
        psit_nG = psit_nG.reshape(N, -1)
        G = psit_nG.shape[1]  # number of grid-points
        g = int(np.ceil(G / float(J)))

        # Buffers for send/receive of pre-multiplication versions of P_ani's.
        sbuf_nI = rbuf_nI = None
        if P_ani:
            sbuf_nI = np.hstack([P_ni for P_ni in P_ani.values()])
            sbuf_nI = np.ascontiguousarray(sbuf_nI)
            if B > 1:
                rbuf_nI = np.empty_like(sbuf_nI)

        # Because of the amount of communication involved, we need to
        # be syncronized up to this point but only on the 1D band_comm
        # communication ring
        band_comm.barrier()
        while g * J >= G + g:  # remove extra slice(s)
            J -= 1
        assert 0 < g * J < G + g

        work1_xG = reshape(self.work1_xG, (self.X,) + psit_nG.shape[1:])
        work2_xG = reshape(self.work2_xG, (self.X,) + psit_nG.shape[1:])

        for j in range(J):
            G1 = j * g
            G2 = G1 + g
            if G2 > G:
                G2 = G
                g = G2 - G1
            sbuf_ng = reshape(work1_xG, (N, g))
            rbuf_ng = reshape(work2_xG, (N, g))
            sbuf_ng[:] = psit_nG[:, G1:G2]
            beta = 0.0
            cycle_P_ani = (j == J - 1 and P_ani)
            for q in range(Q):
                # Start sending currently buffered kets to rank below
                # and receiving next set of kets from rank above us.
                # If we're at the last slice, start cycling P_ani too.
                if q < Q - 1:
                    self._initialize_cycle(sbuf_ng, rbuf_ng,
                                           sbuf_nI, rbuf_nI, cycle_P_ani)

                # Calculate wave-function contributions from the current slice
                # of grid data by the current mynbands x mynbands matrix block.
                C_nn = self.bmd.extract_block(C_NN, (rank + q) % B, rank)
                self.gd.gemm(1.0, sbuf_ng, C_nn, beta, psit_nG[:, G1:G2])

                # If we're at the last slice, add contributions to P_ani's.
                if cycle_P_ani:
                    I1 = 0
                    for P_ni in P_ani.values():
                        I2 = I1 + P_ni.shape[1]
                        gemm(1.0, sbuf_nI[:, I1:I2], C_nn, beta, P_ni)
                        I1 = I2

                # Wait for all send/receives to finish before next iteration.
                # Swap send and receive buffer such that next becomes current.
                # If we're at the last slice, also finishes the P_ani cycle.
                if q < Q - 1:
                    sbuf_ng, rbuf_ng, sbuf_nI, rbuf_nI = self._finish_cycle(
                        sbuf_ng, rbuf_ng, sbuf_nI, rbuf_nI, cycle_P_ani)

                # First iteration was special because we initialized the kets
                if q == 0:
                    beta = 1.0

        psit_nG.shape = shape
        return psit_nG
