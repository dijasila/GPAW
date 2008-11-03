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

    def initialize(self, paw, wfs):
        self.timer = paw.timer
        self.nuclei = paw.nuclei
        self.my_nuclei = paw.my_nuclei
        self.comm = paw.gd.comm
        self.error = 0.0
        #self.nspins = paw.nspins
        #self.nkpts = paw.nkpts
        #self.nbands = paw.nbands
        self.nmybands = paw.nmybands
        self.band_comm = paw.band_comm
        self.dtype = paw.dtype

        self.nao = wfs.nao
        self.eps_m = npy.empty(self.nao)
        self.S_mm = npy.empty((self.nao, self.nao), self.dtype)
        self.Vt_skmm = npy.empty((paw.nspins, paw.nkpts,
                                  self.nao, self.nao), self.dtype)
        self.initialized = True

    def get_hamiltonian_matrix(self, hamiltonian, wfs, kpt=None,k=0,s=0):
        if kpt != None:
            k = kpt.k
            s = kpt.s

        H_mm = self.Vt_skmm[s,k]
        
        for nucleus in self.my_nuclei:
            dH_ii = unpack(nucleus.H_sp[s])
            P_mi = nucleus.P_kmi[k]
            H_mm += npy.dot(P_mi, npy.inner(dH_ii, P_mi).conj())

        self.comm.sum(H_mm)

        H_mm += wfs.T_kMM[k]

        return H_mm

    def iterate(self, hamiltonian, wfs):
        self.timer.start('LCAO: potential matrix')
        wfs.basis_functions.calculate_potential_matrix(hamiltonian.vt_sG,
                                                       self.Vt_skmm)
        self.timer.stop('LCAO: potential matrix')

        for kpt in wfs.kpoints.kpt_u:
            self.iterate_one_k_point(hamiltonian, wfs, kpt)


    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        k = kpt.k
        s = kpt.s
        u = kpt.u

        H_mm = self.get_hamiltonian_matrix(hamiltonian, wfs, kpt)
        self.S_mm[:] = wfs.S_kMM[k]

        rank = self.band_comm.rank
        size = self.band_comm.size

        n1 = rank * self.nmybands
        n2 = n1 + self.nmybands

        # Check and remove linear dependence for the current k-point
        if k in wfs.lcao_hamiltonian.linear_kpts:
            print '*Warning*: near linear dependence detected for k=%s' % k
            P_mm, p_m = wfs.lcao_hamiltonian.linear_kpts[k]
            thres = wfs.lcao_hamiltonian.thres
            eps_q, C_nm = self.remove_linear_dependence(P_mm, p_m, H_mm, thres)
            kpt.C_nm[:] = C_nm[n1:n2]
            kpt.eps_n[:] = eps_q[n1:n2]
        else:
            dsyev_zheev_string = 'LCAO: ' + 'dsygv/zhegv'

            self.timer.start(dsyev_zheev_string)
            if debug:
                self.timer.start(dsyev_zheev_string +
                                 ' %03d' % self.eig_lcao_iteration)

            if self.comm.rank == 0:
                self.eps_m[0] = 42
                info = diagonalize(H_mm, self.eps_m, self.S_mm)
                assert self.eps_m[0] != 42
                if info != 0:
                    raise RuntimeError('Failed to diagonalize: info=%d' % info)

            if debug:
                self.timer.stop(dsyev_zheev_string + ' %03d'
                                % self.eig_lcao_iteration)
                self.eig_lcao_iteration += 1
            self.timer.stop(dsyev_zheev_string)

            self.comm.broadcast(self.eps_m, 0)
            self.comm.broadcast(H_mm, 0)

            kpt.C_nm[:] = H_mm[n1:n2]
            kpt.eps_n[:] = self.eps_m[n1:n2]

        for nucleus in self.my_nuclei:
            nucleus.P_uni[u] = npy.dot(kpt.C_nm, nucleus.P_kmi[k])

    def remove_linear_dependence(self, P_mm, p_m, H_mm, thres):
        """Diagonalize H_mm with a reduced overlap matrix from which the
        linear dependent eigenvectors have been removed.

        The eigenvectors P_mm of the overlap matrix S_mm which correspond
        to eigenvalues p_m < thres are removed, thus producing a
        q-dimensional subspace. The hamiltonian H_mm is also transformed into
        H_qq and diagonalized. The transformation operator P_mq looks like::

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

        s_q = npy.extract(p_m > thres, p_m)
        S_qq = npy.diag(s_q)
        S_qq = npy.array(S_qq, self.dtype)
        q = len(s_q)
        p = self.nao - q
        P_mq = P_mm[p:, :].T.conj()

        # Filling up the upper triangle
        for m in range(self.nao - 1):
            H_mm[m, m:] = H_mm[m:, m].conj()

        H_qq = npy.dot(P_mq.T.conj(), npy.dot(H_mm, P_mq))

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
        C_nm = npy.dot(C_nq, P_mq.T.conj())
        return eps_q, C_nm
