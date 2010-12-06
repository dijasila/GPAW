from gpaw.xc.lda import LDA
from gpaw.xc.libxc import LibXC


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
