import numpy as npy
from gpaw.utilities.blas import rk, r2k
from gpaw.utilities import unpack
from gpaw.utilities.lapack import diagonalize
from gpaw import debug


class LCAO:
    """Eigensolver for LCAO-basis calculation"""

    def __init__(self):
        self.lcao = True
        self.initialized = False
        if debug:
            self.eig_lcao_iteration = 0
        self.linear_kpts = None

    def initialize(self, paw, wfs):
        self.timer = paw.timer
        self.nuclei = paw.nuclei
        self.my_nuclei = paw.my_nuclei
        self.comm = paw.gd.comm
        self.error = 0.0
        self.nmybands = paw.nmybands
        self.band_comm = paw.band_comm
        self.dtype = paw.dtype

        self.nao = wfs.nao
        self.eps_M = npy.empty(self.nao)
        self.S_MM = npy.empty((self.nao, self.nao), self.dtype)
        self.H_MM = npy.empty((self.nao, self.nao), self.dtype)
        self.linear_dependence_check(wfs)
        self.initialized = True

    def linear_dependence_check(self, wfs):
        # Near-linear dependence check. This is done by checking the
        # eigenvalues of the overlap matrix S_kmm. Eigenvalues close
        # to zero mean near-linear dependence in the basis-set.
        self.linear_kpts = {}
        for k, S_MM in enumerate(wfs.S_kMM):
            P_MM = S_MM.copy()
            #P_mm = wfs.S_kMM[k].copy()
            p_M = npy.empty(self.nao)

            dsyev_zheev_string = 'LCAO: '+'diagonalize-test'

            self.timer.start(dsyev_zheev_string)
            if debug:
                self.timer.start(dsyev_zheev_string +
                                 ' %03d' % self.eig_lcao_iteration)

            if self.comm.rank == 0:
                p_M[0] = 42
                info = diagonalize(P_MM, p_M)
                assert p_M[0] != 42
                if info != 0:
                    raise RuntimeError('Failed to diagonalize: info=%d' % info)

            if debug:
                self.timer.stop(dsyev_zheev_string +
                                ' %03d' % self.eig_lcao_iteration)
                self.eig_lcao_iteration += 1
            self.timer.stop(dsyev_zheev_string)

            self.comm.broadcast(P_MM, 0)
            self.comm.broadcast(p_M, 0)

            self.thres = 1e-6
            if (p_M <= self.thres).any():
                self.linear_kpts[k] = (P_MM, p_M)

        # Debug stuff
        if 0:
            print 'Hamiltonian S_kMM[0] diag'
            print self.S_kMM[0].diagonal()
            print 'Hamiltonian S_kMM[0]'
            for row in self.S_kMM[0]:
                print ' '.join(['%02.03f' % f for f in row])
            print 'Eigenvalues:'
            print npy.linalg.eig(self.S_kMM[0])[0]

    def get_hamiltonian_matrix(self, hamiltonian, wfs, kpt=None, k=0, s=0):
        if kpt is not None:
            k = kpt.k
            s = kpt.s

        self.timer.start('LCAO: potential matrix')
        wfs.basis_functions.calculate_potential_matrix(
            hamiltonian.vt_sG[s], self.H_MM, k)
        self.timer.stop('LCAO: potential matrix')
        
        for nucleus in self.my_nuclei:
            dH_ii = unpack(nucleus.H_sp[s])
            P_Mi = nucleus.P_kmi[k]
            self.H_MM += npy.dot(P_Mi, npy.inner(dH_ii, P_Mi).conj())

        self.comm.sum(self.H_MM)

        self.H_MM += wfs.T_kMM[k]

        return self.H_MM

    def iterate(self, hamiltonian, wfs):
        for kpt in wfs.kpt_u:
            self.iterate_one_k_point(hamiltonian, wfs, kpt)

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        k = kpt.k
        s = kpt.s
        u = kpt.u

        H_MM = self.get_hamiltonian_matrix(hamiltonian, wfs, kpt)
        self.S_MM[:] = wfs.S_kMM[k]

        rank = self.band_comm.rank
        size = self.band_comm.size

        n1 = rank * self.nmybands
        n2 = n1 + self.nmybands

        # Check and remove linear dependence for the current k-point
        if k in self.linear_kpts:
            print '*Warning*: near linear dependence detected for k=%s' % k
            P_MM, p_M = wfs.lcao_hamiltonian.linear_kpts[k]
            eps_q, C2_nM = self.remove_linear_dependence(P_MM, p_M, H_MM)
            kpt.C_nM[:] = C2_nM[n1:n2]
            kpt.eps_n[:] = eps_q[n1:n2]
        else:
            dsyev_zheev_string = 'LCAO: ' + 'dsygv/zhegv'

            self.timer.start(dsyev_zheev_string)
            if debug:
                self.timer.start(dsyev_zheev_string +
                                 ' %03d' % self.eig_lcao_iteration)

            if self.comm.rank == 0:
                self.eps_M[0] = 42
                info = diagonalize(H_MM, self.eps_M, self.S_MM)
                assert self.eps_M[0] != 42
                if info != 0:
                    raise RuntimeError('Failed to diagonalize: info=%d' % info)

            if debug:
                self.timer.stop(dsyev_zheev_string + ' %03d'
                                % self.eig_lcao_iteration)
                self.eig_lcao_iteration += 1
            self.timer.stop(dsyev_zheev_string)

            self.comm.broadcast(self.eps_M, 0)
            self.comm.broadcast(H_MM, 0)

            kpt.C_nM[:] = H_MM[n1:n2]
            kpt.eps_n[:] = self.eps_M[n1:n2]

        for nucleus in self.my_nuclei:
            nucleus.P_uni[u] = npy.dot(kpt.C_nM, nucleus.P_kmi[k])

    def remove_linear_dependence(self, P_MM, p_M, H_MM):
        """Diagonalize H_MM with a reduced overlap matrix from which the
        linear dependent eigenvectors have been removed.

        The eigenvectors P_MM of the overlap matrix S_mm which correspond
        to eigenvalues p_M < thres are removed, thus producing a
        q-dimensional subspace. The hamiltonian H_MM is also transformed into
        H_qq and diagonalized. The transformation operator P_Mq looks like::

                ------------m--------- ...
                ---p---  ------q------ ...
               +---------------------------
               |
           |   |
           |   |
           m   |
           |   |
           |   |
             . |
             .


        """

        s_q = npy.extract(p_M > self.thres, p_M)
        S_qq = npy.diag(s_q)
        S_qq = npy.array(S_qq, self.dtype)
        q = len(s_q)
        p = self.nao - q
        P_Mq = P_MM[p:, :].T.conj()

        # Filling up the upper triangle
        for M in range(self.nao - 1):
            H_MM[M, m:] = H_MM[M:, M].conj()

        H_qq = npy.dot(P_Mq.T.conj(), npy.dot(H_MM, P_Mq))

        eps_q = npy.zeros(q)

        dsyev_zheev_string = 'LCAO: ' + 'dsygv/zhegv remove'

        self.timer.start(dsyev_zheev_string)
        if debug:
            self.timer.start(dsyev_zheev_string +
                             ' %03d' % self.eig_lcao_iteration)

        if self.comm.rank == 0:
            eps_q[0] = 42
            info = diagonalize(H_qq, eps_q, S_qq)
            assert eps_q[0] != 42
            if info != 0:
                raise RuntimeError('Failed to diagonalize: info=%d' % info)

        if debug:
            self.timer.stop(dsyev_zheev_string +
                            ' %03d' % self.eig_lcao_iteration)
            self.eig_lcao_iteration += 1
        self.timer.stop(dsyev_zheev_string)

        self.comm.broadcast(eps_q, 0)
        self.comm.broadcast(H_qq, 0)

        C_nq = H_qq
        C_nM = npy.dot(C_nq, P_Mq.T.conj())
        return eps_q, C_nM
