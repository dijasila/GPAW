# Copyright (C) 2008 CAMd
# Please see the accompanying LICENSE file for further information.

from __future__ import division

import numpy as np
from gpaw.utilities.blas import rk, r2k, gemm
from gpaw.matrix_descriptor import BandMatrixDescriptor, \
                                   BlacsBandMatrixDescriptor

class MatrixOperator:
    """Base class for overlap and hamiltonian operators.

    Due to optimized BLAS usage, matrices are considered
    transposed both upon input and output.

    As both the overlap and Hamiltonian matrices are Hermitian, they
    can be considered as transposed *or* conjugated as compared to
    standard definitions.
    """

    # This class has 100% parallel unittest coverage by parallel/ut_hsops.py!
    # If you add to or change any aspect of the code, please update the test.

    nblocks = 1
    async = True
    hermitian = True

    def __init__(self, bd, gd, ksl, nblocks=None, async=None, hermitian=None):
        self.bd = bd
        self.gd = gd
        self.work1_xG = None
        self.work2_xG = None
        self.A_qnn = None
        self.A_nn = None
        if nblocks is not None:
            self.nblocks = nblocks
        if async is not None:
            self.async = async
        if hermitian is not None:
            self.hermitian = hermitian
        self.bmd = ksl.new_descriptor() #XXX take hermitian as argument?
        self.M = 1
        self.Q = bd.comm.size

    def allocate_work_arrays(self, dtype):
        """This is a little complicated, but let's look at the facts.

        Given::

          N = mynbands            The number of bands on this core.
          J = nblocks             The number of blocks to divide bands into.
          M = N // J              The number of bands in each block.
          G = gd.n_c.prod()       The number of grid points on this core.
          g = int(np.ceil(G/J))   The number of grid points in a block.

        We allocate work arrays as gd.empty(X) where X is to be determined.

        Conditions::

          N % J == 0        Bands must be exactly divisible into blocks.
          g * J >= G        The grid point blocks can contain all the points.
          X * G >= N * g    Blocking over grid points must have enough space.

        Cases::

          When G % J == 0, the expression for g simplifies to g = G//J,
          and g * J >= G is obviously fulfilled. The condition for X
          is equivalent to X >= N//J, hence this is the minimal requirement.

          When G % J != 0, the expression for g becomes g = G//J+1 instead,
          and g * J >= G is again fulfilled. The condition for X is basically
          X * J * G >= N * (G+J) hence X = N//J + int(np.ceil(N/G)) is best.

        """
        ngroups = self.bd.comm.size
        mynbands = self.bd.mynbands
        nbands = self.bd.nbands
        if ngroups == 1 and self.nblocks == 1:
            self.work1_xG = self.gd.zeros(mynbands, dtype)
        else:
            assert mynbands % self.nblocks == 0
            M = mynbands // self.nblocks
            self.M = M
            X = M
            if self.gd.n_c.prod() % self.nblocks != 0:
                X += int(np.ceil(mynbands/self.gd.n_c.prod()))
            self.work1_xG = self.gd.zeros(X, dtype)
            self.work2_xG = self.gd.zeros(X, dtype)
            if ngroups > 1:
                if self.hermitian:
                    self.Q = ngroups // 2 + 1
                else:
                    self.Q = ngroups
                self.A_qnn = np.zeros((self.Q, mynbands, mynbands), dtype)
        self.A_nn = self.bmd.zeros(dtype=dtype)

    def estimate_memory(self, mem, dtype):
        ngroups = self.bd.comm.size
        mynbands = self.bd.mynbands
        nbands = self.bd.nbands
        gdbytes = self.gd.bytecount(dtype)
        # Code semipasted from allocate_work_arrays
        if ngroups == 1 and self.nblocks == 1:
            mem.subnode('work_xG', mynbands * gdbytes)
        else:
            X = mynbands // self.nblocks
            if self.gd.n_c.prod() % self.nblocks != 0:
                X += int(np.ceil(mynbands/self.gd.n_c.prod()))
            mem.subnode('2 work_xG', 2 * X * gdbytes)
            if ngroups > 1:
                if self.hermitian:
                    Q = ngroups // 2 + 1
                else:
                    Q = ngroups
                count = Q * mynbands**2
                mem.subnode('A_qnn', count * mem.itemsize[dtype])
        mem.subnode('A_nn', np.prod(self.bmd.shape) * mem.itemsize[dtype])

    def _pseudo_braket(self, bra_xG, ket_yG, A_yx, square=None):
        """Calculate matrix elements of braket pairs of pseudo wave functions.
        Low-level helper function. Results will be put in the *A_yx* array::
        
                   /     ~ *     ~   
           A    =  | dG bra (G) ket  (G)
            nn'    /       n       n'


        Parameters:

        bra_xG: ndarray
            Set of bra-like vectors in which the matrix elements are evaluated.
        key_yG: ndarray
            Set of ket-like vectors in which the matrix elements are evaluated.
        A_yx: ndarray
            Matrix in which to put calculated elements. Take care: Due to the
            difference in Fortran/C array order and the inherent BLAS nature,
            the matrix has to be filled in transposed (conjugated in future?).

        """
        assert bra_xG.shape[1:] == ket_yG.shape[1:]
        assert (ket_yG.shape[0], bra_xG.shape[0]) == A_yx.shape

        if square is None:
            square = (bra_xG.shape[0]==ket_yG.shape[0])

        dv = self.gd.dv
        if ket_yG is bra_xG:
            rk(dv, bra_xG, 0.0, A_yx)
        elif self.hermitian and square:
            r2k(0.5 * dv, bra_xG, ket_yG, 0.0, A_yx)
        else:
            gemm(dv, bra_xG, ket_yG, 0.0, A_yx, 'c')

    def _initialize_cycle(self, sbuf_mG, rbuf_mG, sbuf_In, rbuf_In, auxiliary):
        """Initializes send/receive cycle of pseudo wave functions, as well as
        an optional auxiliary send/receive cycle of corresponding projections.
        Low-level helper function. Results in the following communications::

                        Rank below            This rank            Rank above
          Asynchronous: ... o/i  <-- sbuf_mG --  o/i  <-- rbuf_mG --  o/i ...
          Synchronous:     blank                blank                blank

          Auxiliary:    ... o/i  <-- sbuf_In --  o/i  <-- rbuf_In --  o/i ...

        A letter 'o' signifies a non-blocking send and 'i' a matching receive.


        Parameters:

        sbuf_mG: ndarray
            Send buffer for the outgoing set of pseudo wave functions.
        rbuf_mG: ndarray
            Receive buffer for the incoming set of pseudo wave functions.
        sbuf_In: ndarray, ignored if not auxiliary
            Send buffer for the outgoing set of atomic projector overlaps.
        rbuf_In: ndarray, ignored if not auxiliary
            Receive buffer for the incoming set of atomic projector overlaps.
        auxiliary: bool
            Determines whether to initiate the auxiliary send/receive cycle.

        """
        band_comm = self.bd.comm
        rankm = (band_comm.rank - 1) % band_comm.size
        rankp = (band_comm.rank + 1) % band_comm.size
        self.req, self.req2 = [], []

        # If asyncronous, non-blocking send/receives of psit_nG's start here.
        if self.async:
            self.req.append(band_comm.send(sbuf_mG, rankm, 11, False))
            self.req.append(band_comm.receive(rbuf_mG, rankp, 11, False))

        # Auxiliary asyncronous cycle, also send/receive of P_ani's.
        if auxiliary:
            self.req2.append(band_comm.send(sbuf_In, rankm, 31, False))
            self.req2.append(band_comm.receive(rbuf_In, rankp, 31, False))

    def _finish_cycle(self, sbuf_mG, rbuf_mG, sbuf_In, rbuf_In, auxiliary):
        """Completes a send/receive cycle of pseudo wave functions, as well as
        an optional auxiliary send/receive cycle of corresponding projections.
        Low-level helper function. Results in the following communications::

                        Rank below            This rank            Rank above
          Asynchronous: ... w/w  <-- sbuf_mG --  w/w  <-- rbuf_mG --  w/w ...
          Synchronous:  ... O/I  <-- sbuf_mG --  O/I  <-- rbuf_mG --  O/I ...

          Auxiliary:    ... w/w  <-- sbuf_In --  w/w  <-- rbuf_In --  w/w ...

        A letter 'w' signifies wait for initialized non-blocking communication.
        The letter 'O' signifies a blocking send and 'I' a matching receive.


        Parameters:

        Same as _initialize_cycle.

        Returns:

        sbuf_mG: ndarray
            New send buffer with the received set of pseudo wave functions.
        rbuf_mG: ndarray
            New receive buffer (has the sent set of pseudo wave functions).
        sbuf_In: ndarray, same as input if not auxiliary
            New send buffer with the received set of atomic projector overlaps.
        rbuf_In: ndarray, same as input if not auxiliary
            New receive buffer (has the sent set of atomic projector overlaps).

        """
        band_comm = self.bd.comm
        rankm = (band_comm.rank - 1) % band_comm.size
        rankp = (band_comm.rank + 1) % band_comm.size

        # If syncronous, blocking send/receives of psit_nG's carried out here.
        if self.async:
            assert len(self.req) == 2, 'Expected asynchronous request pairs.'
            band_comm.waitall(self.req)
        else:
            assert len(self.req) == 0, 'Got unexpected asynchronous requests.'
            band_comm.sendreceive(sbuf_mG, rankm, rbuf_mG, rankp, 11, 11)
        sbuf_mG, rbuf_mG = rbuf_mG, sbuf_mG

        # Auxiliary asyncronous cycle, also wait for P_ani's.
        if auxiliary:
            assert len(self.req2) == 2, 'Expected asynchronous request pairs.'
            band_comm.waitall(self.req2)
            sbuf_In, rbuf_In = rbuf_In, sbuf_In

        return sbuf_mG, rbuf_mG, sbuf_In, rbuf_In

    def suggest_temporary_buffer(self, dtype):
        """Return a *suggested* buffer for calculating A(psit_nG) during
        a call to calculate_matrix_elements. Work arrays will be allocated
        if they are not already available.

        Note that the temporary buffer is merely a reference to (part of) a
        work array, hence data race conditions occur if you're not careful.
        """
        if self.work1_xG is None:
            self.allocate_work_arrays(dtype)
        else:
            assert self.work1_xG.dtype == dtype

        J = self.nblocks
        N = self.bd.mynbands
        B = self.bd.comm.size

        if B == 1 and J == 1:
            return self.work1_xG
        else:
            assert N % J == 0, "Can't divide %d bands in %d blocks." % (N,J)
            M = self.M
            return self.work1_xG[:M]

    def calculate_matrix_elements(self, psit_nG, P_ani, A, dA):
        """Calculate matrix elements for A-operator.

        Results will be put in the *A_nn* array::

                                  ___
                    ~   ^  ~     \     ~   ~a    a   ~a  ~
           A    = <psi |A|psi > + )  <psi |p > dA   <p |psi >
            nn'       n      n'  /___    n  i    ii'  i'   n'
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
        dA: dict or function
            Dictionary of atomic matrix elements dA_ii = d<phi_i | A | phi_i >
            or functional form of the operator which works on | phi_i >.
            Must accept atomic index a and P_ni and return an ndarray with the
            same shape as P_ni, thus representing P_ni multiplied by dA_ii.

        """
        band_comm = self.bd.comm
        domain_comm = self.gd.comm
        B = band_comm.size
        J = self.nblocks
        N = self.bd.mynbands
        
        if self.work1_xG is None:
            self.allocate_work_arrays(psit_nG.dtype)
        else:
            assert self.work1_xG.dtype == psit_nG.dtype

        A_NN = self.A_nn

        dAP_ani = {}
        for a, P_ni in P_ani.items():
            if hasattr(dA, '__call__'):
                dAP_ani[a] = dA(a, P_ni)
            else:
                # dA denotes dA_aii as usual
                dAP_ani[a] = np.dot(P_ni, dA[a])
        
        if B == 1 and J == 1:
            # Simple case:
            Apsit_nG = A(psit_nG)
            self._pseudo_braket(psit_nG, Apsit_nG, A_NN)
            for a, P_ni in P_ani.items():
                # A_NN += np.dot(dAP_ani[a], P_ni.T.conj())
                gemm(1.0, P_ni, dAP_ani[a], 1.0, A_NN, 'c')
            domain_comm.sum(A_NN, 0)
            return self.bmd.redistribute_output(A_NN)
        
        # Now it gets nasty! We parallelize over B groups of bands and
        # each band group is blocked in J smaller slices (less memory).
        Q = self.Q
        M = self.M

        # Buffer for storage of blocks of calculated matrix elements.
        if B == 1:
            A_qnn = A_NN.reshape((1, N, N))
        else:
            A_qnn = self.A_qnn

        # Buffers for send/receive of operated-on versions of P_ani's.
        sbuf_In = rbuf_In = None
        if P_ani:
            sbuf_In = np.concatenate([dAP_ani[a].T
                                      for a, P_ni in P_ani.items()])
            if B > 1:
                rbuf_In = np.empty_like(sbuf_In)

        # Because of the amount of communication involved, we need to
        # be syncronized up to this point but only on the 1D band_comm
        # communication ring
        band_comm.barrier()
        for j in range(J):
            n1 = j * M
            n2 = n1 + M
            psit_mG = psit_nG[n1:n2]
            sbuf_mG = A(psit_mG)
            rbuf_mG = self.work2_xG[:M]
            cycle_P_ani = (j == J - 1 and P_ani)

            for q in range(Q):
                A_nn = A_qnn[q]
                A_mn = A_nn[n1:n2]

                # Start sending currently buffered kets to rank below
                # and receiving next set of kets from rank above us.
                # If we're at the last slice, start cycling P_ani too.
                if q < Q - 1:
                    self._initialize_cycle(sbuf_mG, rbuf_mG,
                                           sbuf_In, rbuf_In, cycle_P_ani)

                # Calculate pseudo-braket contributions for the current slice
                # of bands in the current mynbands x mynbands matrix block.
                if q == 0 and self.hermitian and not self.bd.strided:
                    # Special case, we only need the lower part:
                    self._pseudo_braket(psit_nG[:n2], sbuf_mG, A_mn[:, :n2])
                else:
                    self._pseudo_braket(psit_nG, sbuf_mG, A_mn, square=False)

                # If we're at the last slice, add contributions from P_ani's.
                if cycle_P_ani:
                    I1 = 0
                    for P_ni in P_ani.values():
                        I2 = I1 + P_ni.shape[1]
                        gemm(1.0, P_ni, sbuf_In[I1:I2].T.copy(),
                             1.0, A_nn, 'c')
                        I1 = I2

                # Wait for all send/receives to finish before next iteration.
                # Swap send and receive buffer such that next becomes current.
                # If we're at the last slice, also finishes the P_ani cycle.
                if q < Q - 1:
                    sbuf_mG, rbuf_mG, sbuf_In, rbuf_In = self._finish_cycle(
                        sbuf_mG, rbuf_mG, sbuf_In, rbuf_In, cycle_P_ani)

                # First iteration was special because we had the ket to ourself
                if q == 0:
                    rbuf_mG = self.work1_xG[:M]

        domain_comm.sum(A_qnn, 0)

        if B == 1:
            return self.bmd.redistribute_output(A_NN)

        if domain_comm.rank == 0:
            self.bmd.assemble_blocks(A_qnn, A_NN, self.hermitian)

        # Because of the amount of communication involved, we need to
        # be syncronized up to this point.           
        band_comm.barrier()
        domain_comm.barrier()
        return self.bmd.redistribute_output(A_NN)
        
    def matrix_multiply(self, C_NN, psit_nG, P_ani=None):
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

        band_comm = self.bd.comm
        domain_comm = self.gd.comm
        B = band_comm.size
        J = self.nblocks
        N = self.bd.mynbands

        if self.work1_xG is None:
            self.allocate_work_arrays(psit_nG.dtype)
        else:
            assert self.work1_xG.dtype == psit_nG.dtype

        C_NN = self.bmd.redistribute_input(C_NN)

        if B == 1 and J == 1:
            # Simple case:
            newpsit_nG = self.work1_xG
            gemm(1.0, psit_nG, C_NN, 0.0, newpsit_nG)
            self.work1_xG = psit_nG
            if P_ani:
                for P_ni in P_ani.values():
                    gemm(1.0, P_ni.copy(), C_NN, 0.0, P_ni)
            return newpsit_nG
        
        # Now it gets nasty! We parallelize over B groups of bands and
        # each grid chunk is divided in J smaller slices (less memory).

        Q = B # always non-hermitian XXX
        rank = band_comm.rank
        shape = psit_nG.shape
        psit_nG = psit_nG.reshape(N, -1)
        G = psit_nG.shape[1]   # number of grid-points
        g = int(np.ceil(G/J))

        # Buffers for send/receive of pre-multiplication versions of P_ani's.
        sbuf_In = rbuf_In = None
        if P_ani:
            sbuf_In = np.concatenate([P_ni.T for P_ni in P_ani.values()])
            if B > 1:
                rbuf_In = np.empty_like(sbuf_In)

        # Because of the amount of communication involved, we need to
        # be syncronized up to this point but only on the 1D band_comm
        # communication ring
        band_comm.barrier()
        for j in range(J):
            G1 = j * g
            G2 = G1 + g
            if G2 > G:
                G2 = G
                g = G2 - G1
            sbuf_ng = self.work1_xG.reshape(-1)[:N * g].reshape(N, g)
            rbuf_ng = self.work2_xG.reshape(-1)[:N * g].reshape(N, g)
            sbuf_ng[:] = psit_nG[:, G1:G2]
            beta = 0.0
            cycle_P_ani = (j == J - 1 and P_ani)
            for q in range(Q):
                # Start sending currently buffered kets to rank below
                # and receiving next set of kets from rank above us.
                # If we're at the last slice, start cycling P_ani too.
                if q < Q - 1:
                    self._initialize_cycle(sbuf_ng, rbuf_ng,
                                           sbuf_In, rbuf_In, cycle_P_ani)

                # Calculate wave-function contributions from the current slice
                # of grid data by the current mynbands x mynbands matrix block.
                C_nn = self.bmd.extract_block(C_NN, (rank + q) % B, rank)
                gemm(1.0, sbuf_ng, C_nn, beta, psit_nG[:, G1:G2])

                # If we're at the last slice, add contributions to P_ani's.
                if cycle_P_ani:
                    I1 = 0
                    for P_ni in P_ani.values():
                        I2 = I1 + P_ni.shape[1]
                        gemm(1.0, sbuf_In[I1:I2].T.copy(), C_nn, beta, P_ni)
                        I1 = I2

                # Wait for all send/receives to finish before next iteration.
                # Swap send and receive buffer such that next becomes current.
                # If we're at the last slice, also finishes the P_ani cycle.
                if q < Q - 1:
                    sbuf_ng, rbuf_ng, sbuf_In, rbuf_In = self._finish_cycle(
                        sbuf_ng, rbuf_ng, sbuf_In, rbuf_In, cycle_P_ani)

                # First iteration was special because we initialized the kets
                if q == 0:
                    beta = 1.0

        psit_nG.shape = shape
        return psit_nG

