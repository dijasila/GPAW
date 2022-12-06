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


class LatticePeriodicPairFunctionDescriptorCollection:
    """Descriptor collection for lattice periodic pair functions."""

    def __init__(self, wd, pd):
        """Construct the descriptor collection

        Parameters
        ----------
        wd : FrequencyDescriptor
        pd : PWDescriptor
        """
        self.wd = wd
        self.pd = pd

        # Extract q_c
        q_c = pd.kd.bzk_kc[0]
        self.q_c = q_c
        optical_limit = np.allclose(q_c, 0.0)
        self.optical_limit = optical_limit

        # Basis set size
        self.nw = len(wd)
        self.nG = pd.ngmax


class LatticePeriodicPairFunction:
    r"""Data object for lattice periodic pair functions.

    A pair function is considered to be lattice periodic, if it is invariant
    under translations of Bravais lattice vectors R:

    pf(r, r', ω) = pf(r + R, r' + R, ω).

    The Bloch lattice Fourier transform of a lattice periodic pair function,
                        __
                        \
    pf(r, r', q, ω)  =  /  e^(-iq.[r-r'-R']) pf(r, r' + R', ω)
                        ‾‾
                        R'

    is then periodic in both r and r' independently and can be expressed in an
    arbitrary lattice periodic basis.

    In the GPAW response code, lattice periodic pair functions are expanded in
    plane waves:

                   1   //
    pf_GG'(q, ω) = ‾‾ || drdr' e^(-iG.r) pf(r, r', q, ω) e^(iG'.r')
                   V0 //
                        V0

    Hence, the collection consists of a frequency descriptor and a plane-wave
    descriptor, where the latter is specific to the q-point in question.
    """

    def __init__(self, descriptors, blockdist, distribution='WgG'):
        """
        Some documentation here!                                               XXX
        """
        self.descriptors = descriptors
        self.q_c = descriptors.q_c

        self.blockdist = blockdist
        self.distribution = distribution

        if distribution == 'WgG':
            blocks1d = Blocks1D(blockdist.blockcomm, descriptors.nG)
            shape = (descriptors.nw, blocks1d.nlocal, descriptors.nG)
        elif distribution == 'GWg':
            blocks1d = Blocks1D(blockdist.blockcomm, descriptors.nG)
            shape = (descriptors.nG, descriptors.nw, blocks1d.nlocal)
        elif distribution == 'wGG':
            blocks1d = Blocks1D(blockdist.blockcomm, descriptors.nw)
            shape = (blocks1d.nlocal, descriptors.nG, descriptors.nG)
        else:
            raise NotImplementedError(f'Distribution: {distribution}')
        self.blocks1d = blocks1d
        self.shape = shape

        # Allocate array
        self.array = np.zeros(shape, complex)

    
class Chi0Descriptors(LatticePeriodicPairFunctionDescriptorCollection):
    """Descriptor collection for Chi0Data

    Attributes
    ----------
    self.wd : FrequencyDescriptor
        Descriptor for the temporal (frequency) degrees of freedom
    self.pd : PWDescriptor
        Descriptor for the spatial (plane wave) degrees of freedom
    """

    @staticmethod
    def from_descriptor_arguments(frequencies, plane_waves):
        """Contruct the necesarry descriptors and initialize the
        Chi0Descriptors object."""
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

        return Chi0Descriptors(wd, pd)


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
    """Data object containing the response body data arrays
    for a single q-point, while holding also the corresponding
    basis descriptors and block distributor."""

    def __init__(self, descriptors, blockdist):
        """Construct the BodyData object from Chi0Descriptors object.

        Parameters
        ----------
        descriptors: Chi0Descriptors
        blockdist : PlaneWaveBlockDistributor
            Distributor for the block parallelization
        """
        self.descriptors = descriptors
        self.wd = descriptors.wd
        self.pd = descriptors.pd
        self.blockdist = blockdist

        # Initialize block distibution of plane wave basis
        nG = self.pd.ngmax
        self.blocks1d = Blocks1D(blockdist.blockcomm, nG)

        # Data arrays
        self.chi0_wGG = None
        self.allocate_arrays()

    @classmethod
    def from_descriptor_arguments(cls, frequencies, plane_waves,
                                  parallelization):
        """Contruct the necesarry descriptors and initialize the BodyData
        object."""
        descriptors = Chi0Descriptors.from_descriptor_arguments(
            frequencies, plane_waves)
        blockdist = make_blockdist(parallelization)
        return cls(descriptors, blockdist)

    def allocate_arrays(self):
        """Allocate data arrays."""
        self.chi0_wGG = np.zeros(self.wGG_shape, complex)
        
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


class HeadAndWingsData:
    def __init__(self, descriptors):
        assert descriptors.optical_limit
        self.wd = descriptors.wd
        self.pd = descriptors.pd
        self.chi0_wxvG = None  # Wings
        self.chi0_wvv = None  # Head
        self.allocate_arrays()
        
    def allocate_arrays(self):
        self.chi0_wxvG = np.zeros(self.wxvG_shape, complex)
        self.chi0_wvv = np.zeros(self.wvv_shape, complex)

    @staticmethod
    def from_descriptor_arguments(frequencies, plane_waves):
        """Contruct the necesarry descriptors and initialize the
        HeadAndWingsData object"""

        descriptors = Chi0Descriptors.from_descriptor_arguments(
            frequencies, plane_waves)

        return HeadAndWingsData(descriptors)
        
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
    def __init__(self, descriptors, blockdist):
        super().__init__(descriptors, blockdist)
        self.optical_limit = descriptors.optical_limit

        if self.optical_limit:
            self.head_and_wings = HeadAndWingsData(self.descriptors)
            self.chi0_wxvG = self.head_and_wings.chi0_wxvG
            self.chi0_wvv = self.head_and_wings.chi0_wvv
            self.wxvG_shape = self.head_and_wings.wxvG_shape
            self.wvv_shape = self.head_and_wings.wvv_shape
        else:
            self.chi0_wxvG = None
            self.chi0_wvv = None
            self.wxvG_shape = None
            self.wvv_shape = None
