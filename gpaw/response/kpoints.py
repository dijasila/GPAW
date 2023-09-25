import numpy as np


class ResponseKPointGrid:
    def __init__(self, icell_cv, bzk_kc):
        self.icell_cv = icell_cv
        self.bzk_kc = bzk_kc
        self.bzk_kv = np.dot(bzk_kc, icell_cv * 2 * np.pi)
