import numpy as np


class Chi0Data:
    def __init__(self, wd, blockdist, pd, optical_limit):
        self.wd = wd
        self.blockdist = blockdist
        self.pd = pd
        self.optical_limit = optical_limit

        nG = blockdist.blocks1d.N
        nw = len(self.wd)
        wGG_shape = (nw, blockdist.blocks1d.nlocal, nG)

        self.chi0_wGG = np.zeros(wGG_shape, complex)

        if self.optical_limit:
            self.chi0_wxvG = np.zeros((nw, 2, 3, nG), complex)
            self.chi0_wvv = np.zeros((nw, 3, 3), complex)
        else:
            self.chi0_wxvG = None
            self.chi0_wvv = None
