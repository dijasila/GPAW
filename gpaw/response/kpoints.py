import numpy as np


class ResponseKPointGrid:
    def __init__(self, icell_cv, bzk_kc):
        self.icell_cv = icell_cv
        self.bzk_kc = bzk_kc
        self.bzk_kv = bzk_kc @ (2 * np.pi * icell_cv)
