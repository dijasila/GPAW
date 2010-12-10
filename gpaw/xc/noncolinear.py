import numpy as np

from gpaw.xc.lda import LDA
from gpaw.xc.libxc import LibXC
from gpaw.lcao.eigensolver import LCAO
from gpaw.wavefunctions.lcao import LCAOWaveFunctions
from gpaw.utilities import unpack
from gpaw.utilities.blas import gemm
from gpaw.mixer import BaseMixer


class NonColinearLDAKernel(LibXC):
    def __init__(self):
        LibXC.__init__(self, 'LDA')
        
    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):
        n_g = n_sg[0]
        m_vg = n_sg[1:4]
        m_g = (m_vg**2).sum(0)**0.5
        nnew_sg = np.empty((2,) + n_g.shape)
        nnew_sg[:] = n_g
        nnew_sg[0] += m_g
        nnew_sg[1] -= m_g
        nnew_sg *= 0.5
        vnew_sg = np.zeros_like(nnew_sg)
        LibXC.calculate(self, e_g, nnew_sg, vnew_sg)
        dedn_sg[0] += 0.5 * vnew_sg.sum(0)
        dir_vg = m_vg / m_g
        dedn_sg[1:4] += 0.5 * vnew_sg[0] * dir_vg
        dedn_sg[1:4] -= 0.5 * vnew_sg[1] * dir_vg


class NonColinearLCAOEigensolver(LCAO):
    def calculate_hamiltonian_matrix(self, ham, wfs, kpt, root=-1):

        assert self.has_initialized

        vt_sG = ham.vt_sG
        dH_asp = ham.dH_asp
        H_MM = np.empty((wfs.ksl.mynao, wfs.ksl.nao), wfs.dtype)
        H_sMsM = np.empty((2, wfs.ksl.mynao, 2, wfs.ksl.nao), complex)
        
        wfs.timer.start('Potential matrix')
        self.get_component(wfs, 0, vt_sG, dH_asp, kpt, H_MM)
        H_sMsM[0, :, 0] = H_MM
        H_sMsM[1, :, 1] = H_MM
        self.get_component(wfs, 1, vt_sG, dH_asp, kpt, H_MM)
        H_sMsM[0, :, 1] = H_MM
        H_sMsM[1, :, 0] = H_MM.conj().T
        self.get_component(wfs, 2, vt_sG, dH_asp, kpt, H_MM)
        H_sMsM[0, :, 1] += 1j * H_MM
        H_sMsM[1, :, 0] -= 1j * H_MM.conj().T
        self.get_component(wfs, 3, vt_sG, dH_asp, kpt, H_MM)
        H_sMsM[0, :, 0] += H_MM
        H_sMsM[1, :, 0] -= H_MM
        wfs.timer.stop('Potential matrix')

        H_sMsM[0, :, 0] += wfs.T_qMM[kpt.q]
        H_sMsM[1, :, 1] += wfs.T_qMM[kpt.q]

        wfs.timer.start('Distribute overlap matrix')
        #H_MM = wfs.ksl.distribute_overlap_matrix(H_MM, root)
        wfs.timer.stop('Distribute overlap matrix')
        H_sMsM.shape = (2 * wfs.ksl.mynao, 2 * wfs.ksl.nao)
        return H_sMsM

    def get_component(self, wfs, s, vt_sG, dH_asp, kpt, H_MM):
        wfs.basis_functions.calculate_potential_matrix(vt_sG[s], H_MM, kpt.q)

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
        wfs.timer.stop('Atomic Hamiltonian')
        for a, P_Mi in kpt.P_aMi.items():
            dH_ii = np.asarray(unpack(dH_asp[a][s]), wfs.dtype)
            dHP_iM = np.zeros((dH_ii.shape[1], P_Mi.shape[0]), wfs.dtype)
            # (ATLAS can't handle uninitialized output array)
            gemm(1.0, P_Mi, dH_ii, 0.0, dHP_iM, 'c')
            gemm(1.0, dHP_iM, P_Mi[Mstart:Mstop], 1.0, H_MM)

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        if wfs.bd.comm.size > 1 and wfs.bd.strided:
            raise NotImplementedError

        H_MM = self.calculate_hamiltonian_matrix(hamiltonian, wfs, kpt, root=0)
        S_MM = np.zeros_like(H_MM)
        nao = wfs.ksl.nao
        S_MM.shape = (2, nao, 2, nao)
        for s in range(2):
            S_MM[s, :, s] = wfs.S_qMM[kpt.q]
        S_MM.shape = (2 * nao, 2 * nao)

        if kpt.eps_n is None:
            kpt.eps_n = np.empty(wfs.bd.mynbands)

        diagonalization_string = repr(self.diagonalizer)
        wfs.timer.start(diagonalization_string)
        kpt.C_nsM.shape = (wfs.bd.mynbands, 2 * nao)
        self.diagonalizer.diagonalize(H_MM, kpt.C_nsM, kpt.eps_n, S_MM)
        kpt.C_nsM.shape = (wfs.bd.mynbands, 2, nao)
        wfs.timer.stop(diagonalization_string)


class NonColinearLCAOWaveFunctions(LCAOWaveFunctions):
    def set_positions(self, spos_ac):
        LCAOWaveFunctions.set_positions(self, spos_ac)
        for kpt in self.kpt_u:
            kpt.C_nM = None
            kpt.C_nsM = np.empty((self.bd.mynbands, 2, self.ksl.nao), complex)
            
    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        rho_MM = self.ksl.calculate_density_matrix(f_n, kpt.C_nsM[:, 0])
        self.basis_functions.construct_density(rho_MM, nt_sG[0], kpt.q)
        rho_MM = self.ksl.calculate_density_matrix(f_n, kpt.C_nsM[:, 1])
        self.basis_functions.construct_density(rho_MM, nt_sG[3], kpt.q)
        nt_sG[0] += nt_sG[3]
        nt_sG[0] *= 0.5
        nt_sG[3] -= nt_sG[0]
        nt_sG[3] = -nt_sG[3]
        rho_MM = self.ksl.calculate_density_matrix(f_n, kpt.C_nsM[:, 1]
                                                   C2nM=kpt.C_nsM[:, 0])
        self.basis_functions.construct_density(rho_MM.real, nt_sG[1], kpt.q)
        self.basis_functions.construct_density(rho_MM.imag, nt_sG[2], kpt.q)

    def calculate_atomic_density_matrices_k_point(self, D_sii, kpt, a, f_n):
        ...
        P_Mi = kpt.P_aMi[a]
        #P_Mi = kpt.P_aMi_sparse[a]
        #ind = get_matrix_index(kpt.P_aMi_index[a])
        #D_sii[kpt.s] += np.dot(np.dot(P_Mi.T.conj(), kpt.rho_MM),
        #                       P_Mi).real
        rhoP_Mi = np.zeros_like(P_Mi)
        D_ii = np.zeros(D_sii[kpt.s].shape, kpt.rho_MM.dtype)
        #gemm(1.0, P_Mi, kpt.rho_MM[ind.T, ind], 0.0, tmp)
        gemm(1.0, P_Mi, kpt.rho_MM, 0.0, rhoP_Mi)
        gemm(1.0, rhoP_Mi, P_Mi.T.conj().copy(), 0.0, D_ii)
        D_sii[kpt.s] += D_ii.real
        #D_sii[kpt.s] += dot(dot(P_Mi.T.conj().copy(),
        #                        kpt.rho_MM[ind.T, ind]), P_Mi).real
        

class NonColinearMixer(BaseMixer):
    def mix(self, density):
        nt_sG = density.nt_sG
        D_asp = density.D_asp.values()

        # Mix density
        BaseMixer.mix(self, nt_sG[0], D_asp)
