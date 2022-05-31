import numpy as np


class Chi0Data:
    """Data object containing the chi0 data arrays for a single q-point,
    while holding also the corresponding basis descriptors and block
    distributor."""
    def __init__(self, wd, blockdist, pd, optical_limit, extend_head):
        """Construct the Chi0Data object

        Parameters
        ----------
        wd : FrequencyDescriptor
            Descriptor for the temporal (frequency) degrees of freedom
        blockdist : PlaneWaveBlockDistributor
            Distributor for the block parallelization
        pd : PWDescriptor
            Descriptor for the spatial (plane wave) degrees of freedom
        optical_limit : bool
            Are we in the q=0 limit?
        extend_head : bool
            If True: Extend the wings and head of chi in the optical limit to
            take into account the non-analytic nature of chi. Effectively
            means that chi has dimension (nw, nG + 2, nG + 2) in the optical
            limit.
        """
        self.wd = wd
        self.blockdist = blockdist
        self.pd = pd
        self.optical_limit = optical_limit
        self.extend_head = extend_head

        nG = blockdist.blocks1d.N
        nw = len(self.wd)
        wGG_shape = (nw, blockdist.blocks1d.nlocal, nG)

        self.chi0_wGG = np.zeros(wGG_shape, complex)

        if self.optical_limit and not self.extend_head:
            self.chi0_wxvG = np.zeros((nw, 2, 3, nG), complex)
            self.chi0_wvv = np.zeros((nw, 3, 3), complex)
        else:
            self.chi0_wxvG = None
            self.chi0_wvv = None
