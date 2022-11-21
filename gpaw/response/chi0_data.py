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

    
class ResponseDescriptors:
    """ Data object holding all combined response descriptors needed for
    chi0Data
        Parameters
        ----------
        wd: FrequencyDescriptor
            Descriptor for the temporal (frequency) degrees of freedom
        pd: PWDescriptor
            Descriptor for the spatial (plane wave) degrees of freedom
"""
    
    def __init__(self, wd, pd):
        self.wd = wd
        self.pd = pd
        # Check if in optical limit
        q_c, = pd.kd.ibzk_kc
        optical_limit = np.allclose(q_c, 0.0)
        self.optical_limit = optical_limit
        self.nG = pd.ngmax
        
    @staticmethod
    def from_descriptor_arguments(frequencies, plane_waves):
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

        return ResponseDescriptors(wd, pd)

    @staticmethod
    def make_blockdist(parallelization):
        # Construct blockdist
        if isinstance(parallelization, PlaneWaveBlockDistributor):
            blockdist = parallelization
        else:
            assert isinstance(parallelization, tuple)
            assert len(parallelization) == 3
            blockdist = PlaneWaveBlockDistributor(*parallelization)
        return blockdist
        
class BodyData:
    """Data object containing the response body data arrays for a single q-point,
    while holding also the corresponding basis descriptors and block
    distributor."""

    
    def __init__(self, descriptors, blockdist):
        """Construct the Chi0Data object from ResponseDescriptors object.

        Parameters
        ----------
        descriptors: ResponseDescriptors 
        blockdist : PlaneWaveBlockDistributor
            Distributor for the block parallelization
        """
        self.descriptors = descriptors
        self.wd = descriptors.wd
        self.pd = descriptors.pd
        self.blockdist = blockdist
        self.optical_limit = descriptors.optical_limit

        # Initialize block distibution of plane wave basis
        nG = self.pd.ngmax
        self.blocks1d = Blocks1D(blockdist.blockcomm, nG)

        # Data arrays
        self.chi0_wGG = None
        self.allocate_arrays()

    @staticmethod
    def from_descriptor_arguments(frequencies, plane_waves, parallelization):
        """Contruct the necesarry descriptors and initialize the BodyData
        object."""
        descriptors = ResponseDescriptors.from_descriptor_arguments(frequencies, plane_waves)
        blockdist = ResponseDescriptors.make_blockdist(parallelization)
        return BodyData(descriptors, blockdist)

    def allocate_arrays(self):
        """Allocate data arrays."""
        self.chi0_wGG = np.zeros(self.wGG_shape, complex)
        
    @property
    def nw(self):
        return len(self.wd)

    @property
    def nG(self):
        return self.pd.ngmax
        #return self.blocks1d.N

    @property
    def mynG(self):
        return self.blocks1d.nlocal
    
    @property
    def wGG_shape(self):
        return (self.nw, self.mynG, self.nG)

    def distribute_frequencies(self):
        """Return chi0_wGG array with frequencies distributed to all cores."""
        return self.blockdist.distribute_frequencies(self.chi0_wGG, self.nw)

    def distribute_as(self, out_dist):
        """Distribute self.chi0_wGG as given in out_dist.
        out_dist: str 'wGG' for parallell over w and
        'WgG' for parallel over G"""
        return self.blockdist.distribute_as(self.chi0_wGG, self.nw, out_dist)

    def check_distribution(self, test_dist):
        """Checks if self.chi0_wGG is distributed according to test_dist"""
        _, __, same_dist = self.blockdist.check_distribution(self.chi0_wGG,
                                                             self.nw,
                                                             test_dist)
        return same_dist

    def make_HeadWings(self):
        return HeadWingsData(self.descriptors)
    
class HeadWingsData:
    def __init__(self, descriptors):
        assert(descriptors.optical_limit)  # only head and wings in optical limit
        self.wd = descriptors.wd
        self.pd = descriptors.pd
        self.chi0_wxvG = None  # Wings
        self.chi0_wvv = None  # Head
        self.allocate_arrays()
        
    def allocate_arrays(self):
        self.chi0_wxvG = np.zeros(self.wxvG_shape, complex)
        self.chi0_wvv = np.zeros(self.wvv_shape, complex)

    @staticmethod
    def from_descriptor_arguments(frequencies, plane_waves, parallelization):
        """Contruct the necesarry descriptors and initialize the BodyData
        object."""
        descriptors = ResponseDescriptors.from_descriptor_arguments(frequencies, plane_waves)

        return HeadWingsData(descriptors, blockdist)
        
    @property
    def nw(self):
        return len(self.wd)

    @property
    def nG(self):
        return self.pd.ngmax

    @property
    def wxvG_shape(self):
        return (self.nw, 2, 3, self.nG)

    @property
    def wvv_shape(self):
        return (self.nw, 3, 3)

        
class Chi0Data(BodyData):
    """Data object containing the chi0 data arrays for a single q-point,
    while holding also the corresponding basis descriptors and block
    distributor."""
    def __init__(self, wd, pd, blockdist):
        descriptors = ResponseDescriptors(wd, pd)
        super().__init__(descriptors, blockdist)

        if self.optical_limit:
            self.head_and_wings = self.make_HeadWings()
            self.chi0_wxvG = self.head_and_wings.chi0_wxvG
            self.chi0_wvv = self.head_and_wings.chi0_wvv
            self.wxvG_shape = self.head_and_wings.wxvG_shape 
            self.wvv_shape = self.head_and_wings.wvv_shape
        else:
            self.chi0_wxvG = None
            self.chi0_wvv = None
            self.wxvG_shape = None
            self.wvv_shape = None

    @staticmethod
    def from_descriptor_arguments(frequencies, plane_waves, parallelization):
        """Contruct the necesarry descriptors and initialize the BodyData
        object."""
        descriptors = ResponseDescriptors.from_descriptor_arguments(frequencies, plane_waves)
        blockdist = ResponseDescriptors.make_blockdist(parallelization)
        return Chi0Data(descriptors.wd, descriptors.pd, blockdist)
