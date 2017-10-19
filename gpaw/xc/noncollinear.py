import numpy as np


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
