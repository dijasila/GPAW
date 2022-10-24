import numpy as np

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.response.pw_parallelization import (Blocks1D,
                                              PlaneWaveBlockDistributor)
from gpaw.response.frequencies import FrequencyDescriptor


class SingleQPWDescriptor(PWDescriptor):

    @staticmethod
    def from_q(q_c, ecut, gd):
        """Construct a plane wave descriptor for q_c with a given cutoff."""
        qd = KPointDescriptor([q_c])
        return PWDescriptor(ecut, gd, complex, qd)


class Chi0Data:
    """Data object containing the chi0 data arrays for a single q-point,
    while holding also the corresponding basis descriptors and block
    distributor."""
    def __init__(self, wd, pd, blockdist, xp=np):
        """Construct the Chi0Data object

        Parameters
        ----------
        wd: FrequencyDescriptor
            Descriptor for the temporal (frequency) degrees of freedom
        pd: PWDescriptor
            Descriptor for the spatial (plane wave) degrees of freedom
        blockdist : PlaneWaveBlockDistributor
            Distributor for the block parallelization
        """
        self.wd = wd
        self.pd = pd
        self.blockdist = blockdist
        self.xp = xp

        # Check if in optical limit
        q_c, = pd.kd.ibzk_kc
        optical_limit = np.allclose(q_c, 0.0)
        self.optical_limit = optical_limit

        # Initialize block distibution of plane wave basis
        nG = pd.ngmax
        self.blocks1d = Blocks1D(blockdist.blockcomm, nG)

        # Data arrays
        self.chi0_wGG = None
        self.chi0_wxvG = None
        self.chi0_wvv = None

        self.allocate_arrays(xp=xp)

    @staticmethod
    def from_descriptor_arguments(frequencies, plane_waves, parallelization):
        """Contruct the necesarry descriptors and initialize the Chi0Data
        object."""
        # Construct wd
        if isinstance(frequencies, FrequencyDescriptor):
            wd = frequencies
        else:
            wd = frequencies.from_array_or_dict(frequencies)

        # Construct pd
        if isinstance(plane_waves, SingleQPWDescriptor):
            pd = plane_waves
        else:
            assert isinstance(plane_waves, tuple)
            assert len(plane_waves) == 3
            pd = SingleQPWDescriptor.from_q(*plane_waves)

        # Construct blockdist
        if isinstance(parallelization, PlaneWaveBlockDistributor):
            blockdist = parallelization
        else:
            assert isinstance(parallelization, tuple)
            assert len(parallelization) == 3
            blockdist = PlaneWaveBlockDistributor(*parallelization)

        return Chi0Data(wd, pd, blockdist)

    def allocate_arrays(self, xp=np):
        """Allocate data arrays."""
        self.chi0_wGG = xp.zeros(self.wGG_shape, complex)

        if self.optical_limit:
            self.chi0_wxvG = xp.zeros(self.wxvG_shape, complex)
            self.chi0_wvv = xp.zeros(self.wvv_shape, complex)

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
            return None

    def redistribute(self):
        """Return redistributed chi0_wGG array."""
        return self.blockdist.redistribute(self.chi0_wGG, self.nw)

    def distribute_frequencies(self):
        """Return chi0_wGG array with frequencies distributed to all cores."""
        return self.blockdist.distribute_frequencies(self.chi0_wGG, self.nw)
