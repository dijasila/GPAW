import numpy as np

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.response.pw_parallelization import (Blocks1D,
                                              PlaneWaveBlockDistributor)


def create_pd(q_c, ecut, gd):
    """Get the planewave descriptor of q_c."""
    qd = KPointDescriptor([q_c])
    pd = PWDescriptor(ecut, gd, complex, qd)
    return pd


class Chi0Data:
    """Data object containing the chi0 data arrays for a single q-point,
    while holding also the corresponding basis descriptors and block
    distributor."""
    def __init__(self,
                 *,
                 pd,
                 wd,
                 extend_head,
                 world,
                 blockcomm,
                 kncomm):
        """Construct the Chi0Data object

        Parameters
        ----------
        pd: PWDescriptor
            Descriptor for the spatial (plane wave) degrees of freedom
        wd: FrequencyDescriptor
            Descriptor for the temporal (frequency) degrees of freedom
        extend_head: bool
            If True: Extend the wings and head of chi in the optical limit to
            take into account the non-analytic nature of chi. Effectively
            means that chi has dimension (nw, nG + 2, nG + 2) in the optical
            limit.
        """
        q_c, = pd.kd.ibzk_kc
        optical_limit = np.allclose(q_c, 0.0)

        # Initialize block distibution of plane wave basis
        nG = pd.ngmax
        if optical_limit and extend_head:
            nG += 2
        self.blocks1d = Blocks1D(blockcomm, nG)
        self.blockdist = PlaneWaveBlockDistributor(world,
                                                   blockcomm,
                                                   kncomm)
        self.wd = wd
        self.pd = pd
        self.optical_limit = optical_limit
        self.extend_head = extend_head

        # Data arrays
        self.chi0_wGG = None
        self.chi0_wxvG = None
        self.chi0_wvv = None

        self.allocate_arrays()

    def allocate_arrays(self):
        """Allocate data arrays."""
        self.chi0_wGG = np.zeros(self.wGG_shape, complex)

        if self.optical_limit and not self.extend_head:
            self.chi0_wxvG = np.zeros(self.wxvG_shape, complex)
            self.chi0_wvv = np.zeros(self.wvv_shape, complex)

    @property
    def nw(self):
        return len(self.wd)

    @property
    def nG(self):
        return self.blocks1d.N

    @property
    def mynG(self):
        return self.blocks1d.nlocal

    @property
    def wGG_shape(self):
        return (self.nw, self.mynG, self.nG)

    @property
    def wxvG_shape(self):
        if self.optical_limit:
            return (self.nw, 2, 3, self.nG)
        else:
            return None

    @property
    def wvv_shape(self):
        if self.optical_limit:
            return (self.nw, 3, 3)
        else:
            None

    def redistribute(self, out_x=None):
        """Return redistributed chi0_wGG array."""
        return self.blockdist.redistribute(self.chi0_wGG, self.nw, out_x=out_x)

    def distribute_frequencies(self):
        """Return chi0_wGG array with frequencies distributed to all cores."""
        return self.blockdist.distribute_frequencies(self.chi0_wGG, self.nw)
