import numpy as np
from gpaw.response.g0w0 import G0W0


class GW0(G0W0):
    def __init__(self, calc, filename, **kwargs):
        G0W0.__init__(self, calc, filename, **kwargs)
        try:
            qp_xsin = np.load(filename + '-gw0.npy')
        except IOError:
            qp_xsin = []
            
        self.iteration = len(qp_xsin)
        self.qp_xsin = np.empty((self.iteration + 1,) + self.shape)
        self.qp_xsin[:-1] = qp_xsin
    
    def get_k_point(self, s, K, n1, n2):
        kpt = G0W0.get_k_point(self, s, K, n1, n2)
        if self.iteration > 0:
            b1, b2 = self.bands
            m1 = max(b1, n1)
            m2 = min(b2, n2)
            if m2 > m1:
                k = self.calc.wfs.kd.bzk2ibzk_k[K]
                i = self.kpts.find(k)
                qp_n = self.qp_xsin[-1, s, i, m1 - b1:m2 - b1]
                kpt.eps_n[m1 - n1:m2 - n1] = qp_n
                
        return kpt
    
    def calculate(self):
        G0W0.calculate(self)
        self.qp_xsin[-1] = self.qp_sin
        if self.world.rank == 0:
            with open(self.filename + '-gw0.npy', 'w') as fd:
                np.save(fd, self.qp_xsin)
