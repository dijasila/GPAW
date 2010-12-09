import numpy as np

from gpaw.xc.lda import LDA
from gpaw.xc.libxc import LibXC
from gpaw.lcao.eigensolver import LCAO
from gpaw.wavefunctions.lcao import LCAOWaveFunctions
from gpaw.utilities import unpack
from gpaw.utilities.blas import gemm
from gpaw.mixer import BaseMixer


class NonColinearLDA(LDA):
    def __init__(self):
        LDA.__init__(self, LibXC('LDA'))
        
    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        n_g = n_sg[0]
        m_vg = n_sg[1:4]
        m_g = (m_vg**2).sum(0)**0.5
        nnew_sg = gd.empty(2)
        nnew_sg[:] = n_g
        nnew_sg[0] += m_g
        nnew_sg[1] -= m_g
        nnew_sg *= 0.5
        vnew_sg = gd.zeros(2)
        e = LDA.calculate(self, gd, nnew_sg, vnew_sg, e_g)
        v_sg[0] += 0.5 * vnew_sg.sum(0)
        dir_vg = m_vg / m_g
        v_sg[1:4] += 0.5 * vnew_sg[0] * dir_vg
        v_sg[1:4] -= 0.5 * vnew_sg[1] * dir_vg
        return e


class NonColinearLCAOEigensolver(LCAO):
    def calculate_hamiltonian_matrix(self, hamiltonian, wfs, kpt, root=-1):
        assert self.has_initialized
        vt_sG = hamiltonian.vt_sG
        H_sMM = np.empty((4, wfs.ksl.mynao, wfs.ksl.nao), wfs.dtype)

        wfs.timer.start('Potential matrix')
        for s in range(4):
            wfs.basis_functions.calculate_potential_matrix(vt_sG[s],
                                                           H_sMM[s], kpt.q)
        wfs.timer.stop('Potential matrix')

        H_sMsM = np.empty((2, wfs.ksl.mynao, 2, wfs.ksl.nao), wfs.dtype)
        H_sMsM[0, :, 0] = H_sMM[0] + H_sMM[3]
        H_sMsM[0, :, 1] = H_sMM[1] + 1j * H_sMM[2]
        H_sMsM[1, :, 0] = H_sMM[1] - 1j * H_sMM[2]
        H_sMsM[1, :, 1] = H_sMM[0] - H_sMM[3]
        # Add atomic contribution
        #
        #           --   a     a  a*
        # H      += >   P    dH  P
        #  mu nu    --   mu i  ij nu j
        #           aij
        #
        wfs.timer.start('Atomic Hamiltonian')
        Mstart = wfs.basis_functions.Mstart
        Mstop = wfs.basis_functions.Mstop
        for a, P_Mi in kpt.P_aMi.items():
            dH_ii = np.asarray(unpack(hamiltonian.dH_asp[a][kpt.s]), wfs.dtype)
            dHP_iM = np.zeros((dH_ii.shape[1], P_Mi.shape[0]), wfs.dtype)
            # (ATLAS can't handle uninitialized output array)
            gemm(1.0, P_Mi, dH_ii, 0.0, dHP_iM, 'c')
            print dHP_iM, P_Mi[Mstart:Mstop]
            gemm(1.0, dHP_iM, P_Mi[Mstart:Mstop], 1.0, H_sMsM[0, :, 0])
            gemm(1.0, dHP_iM, P_Mi[Mstart:Mstop], 1.0, H_sMsM[1, :, 1])
        wfs.timer.stop('Atomic Hamiltonian')
        wfs.timer.start('Distribute overlap matrix')
        H_sMsM[0, :, 0] += wfs.T_qMM[kpt.q]
        H_sMsM[1, :, 1] += wfs.T_qMM[kpt.q]
        H_sMsM.shape = (2 * wfs.ksl.mynao, 2 * wfs.ksl.nao)
        #H_MM = wfs.ksl.distribute_overlap_matrix(H_MM, root)
        wfs.timer.stop('Distribute overlap matrix')
        return H_sMsM
    
    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        if wfs.bd.comm.size > 1 and wfs.bd.strided:
            raise NotImplementedError

        H_MM = self.calculate_hamiltonian_matrix(hamiltonian, wfs, kpt, root=0)
        S_MM = np.zeros_like(H_MM)
        nao = wfs.ksl.nao
        S_MM.shape = (2, nao, 2, nao)
        for s1 in range(2):
            S_MM[s1, :, s1] = wfs.S_qMM[kpt.q]  # XXX conj?
        S_MM.shape = (2 * nao, 2 * nao)

        if kpt.eps_n is None:
            kpt.eps_n = np.empty(2 * wfs.bd.mynbands)
            
        diagonalization_string = repr(self.diagonalizer)
        wfs.timer.start(diagonalization_string)
        self.diagonalizer._diagonalize(H_MM, S_MM, kpt.eps_n)
        print kpt.eps_n
        print H_MM;sdfg

        wfs.timer.stop(diagonalization_string)

        wfs.timer.start('Calculate projections')
        # P_ani are not strictly necessary as required quantities can be
        # evaluated directly using P_aMi.  We should probably get rid
        # of the places in the LCAO code using P_ani directly
        for a, P_ni in kpt.P_ani.items():
            # ATLAS can't handle uninitialized output array:
            P_ni.fill(117)
            gemm(1.0, kpt.P_aMi[a], kpt.C_nM, 0.0, P_ni, 'n')
        wfs.timer.stop('Calculate projections')


class NonColinearLCAOWaveFunctions(LCAOWaveFunctions):
    pass


class NonColinearMixer(BaseMixer):
    def mix(self, density):
        nt_sG = density.nt_sG
        D_asp = density.D_asp.values()

        # Mix density
        BaseMixer.mix(self, nt_sG[0], D_asp)
