import numpy as np

from gpaw.lcao.eigensolver import DirectLCAO


class NonCollinearLDAKernel:
    name = 'LDA'
    type = 'LDA'

    def __init__(self, kernel):
        self.kernel = kernel

    def calculate(self, e_g, n_sg, v_sg):
        n_g = n_sg[0]
        m_vg = n_sg[1:4]
        m_g = (m_vg**2).sum(0)**0.5
        nnew_sg = np.empty((2,) + n_g.shape)
        nnew_sg[:] = n_g
        nnew_sg[0] += m_g
        nnew_sg[1] -= m_g
        nnew_sg *= 0.5
        vnew_sg = np.zeros_like(nnew_sg)
        self.kernel.calculate(e_g, nnew_sg, vnew_sg)
        v_sg[0] += 0.5 * vnew_sg.sum(0)
        vnew_sg /= np.where(m_g < 1e-15, 1, m_g)
        v_sg[1:4] += 0.5 * vnew_sg[0] * m_vg
        v_sg[1:4] -= 0.5 * vnew_sg[1] * m_vg


class NonCollinearLCAOEigensolver(DirectLCAO):
    def iterate(self, ham, wfs):
        wfs.timer.start('LCAO eigensolver')

        wfs.timer.start('Potential matrix')
        Vt_xdMM = [wfs.basis_functions.calculate_potential_matrices(vt_G)
                   for vt_G in ham.vt_xG]
        wfs.timer.stop('Potential matrix')

        for kpt in wfs.mykpts:
            self.iterate_one_k_point(ham, wfs, kpt, Vt_xdMM)

        wfs.timer.stop('LCAO eigensolver')

    def iterate_one_k_point(self, ham, wfs, kpt, Vt_xdMM):
        if wfs.bd.comm.size > 1 and wfs.bd.strided:
            raise NotImplementedError

        H_xMM = []
        for x in range(4):
            kpt.s = x
            H_MM = self.calculate_hamiltonian_matrix(ham, wfs, kpt, Vt_xdMM[x],
                                                     root=0,
                                                     add_kinetic=(x == 0))
            H_xMM.append(H_MM)

        kpt.s = None

        S_MM = wfs.S_qMM[kpt.q]
        print(H_xMM,S_MM);asdf

        kpt.eps_n = np.empty(wfs.bd.mynbands * 2)

        diagonalization_string = repr(self.diagonalizer)
        wfs.timer.start(diagonalization_string)
        self.diagonalizer.diagonalize(H_MM, kpt.C_nM, kpt.eps_n, S_MM)
        wfs.timer.stop(diagonalization_string)
