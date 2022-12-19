import numpy as np

from gpaw.pw.descriptor import PWMapping

from gpaw.response.pw_parallelization import (Blocks1D,
                                              PlaneWaveBlockDistributor)
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.pair_functions import (SingleQPWDescriptor,
                                          map_WgG_array_to_reduced_pd)


class Chi0Descriptors:
    """Descriptor collection for Chi0Data."""

    def __init__(self, wd, pd):
        """Construct the descriptor collection

        Parameters
        ----------
        wd : FrequencyDescriptor
        pd : SingleQPWDescriptor
        """
        self.wd = wd
        self.pd = pd

        # Extract optical limit
        self.q_c = pd.q_c
        self.optical_limit = np.allclose(pd.q_c, 0.0)

        # Basis set size
        self.nG = pd.ngmax

    @staticmethod
    def from_descriptor_arguments(frequencies, plane_waves):
        """Contruct a Chi0Descriptors, with wd and pd constructed on the fly.
        """
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
        self.data_WgG = self.zeros()

    @classmethod
    def from_descriptor_arguments(cls, frequencies, plane_waves,
                                  parallelization):
        """Contruct the necesarry descriptors and initialize the BodyData
        object."""
        descriptors = Chi0Descriptors.from_descriptor_arguments(
            frequencies, plane_waves)
        blockdist = make_blockdist(parallelization)
        return cls(descriptors, blockdist)

    def zeros(self):
        return np.zeros(self.WgG_shape, complex)
        
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
    def WgG_shape(self):
        return (self.nw, self.mynG, self.nG)

    def get_distributed_frequencies_array(self):
        """Copy data to a 'wGG'-like array, distributed over the entire world.

        This differs from copy_array_with_distribution('wGG'), in that the
        frequencies are distributed over world, instead of among the block
        communicator."""
        return self.blockdist.distribute_frequencies(self.data_WgG, self.nw)

    def copy_array_with_distribution(self, distribution):
        """Copy data to a new array of a desired distribution.

        Parameters
        ----------
        distribution: str
            Array distribution. Choices: 'wGG' and 'WgG'
        """
        data_x = self.blockdist.distribute_as(self.data_WgG, self.nw,
                                              distribution)

        if data_x is self.data_WgG:
            # When asking for 'WgG' distribution or when there is no block
            # distribution at all, we may still be pointing to the original
            # array, but we want strictly to return a copy
            assert distribution == 'WgG' or \
                self.blockdist.blockcomm.size == 1
            data_x = self.data_WgG.copy()

        return data_x


class HeadAndWingsData:
    def __init__(self, descriptors):
        assert descriptors.optical_limit
        self.wd = descriptors.wd
        self.pd = descriptors.pd

        # Allocate head and wings
        self.data_Wvv, self.data_WxvG = self.zeros()
        
    def zeros(self):
        return (np.zeros(self.Wvv_shape, complex),  # head
                np.zeros(self.WxvG_shape, complex))  # wings

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
    def Wvv_shape(self):
        return (self.nw, 3, 3)

    @property
    def WxvG_shape(self):
        return (self.nw, 2, 3, self.nG)


class AugmentedBodyData(BodyData):
    """Data object containing the body data along with the head and
    wings data, if the data concerns the optical limit."""

    def __init__(self, descriptors, blockdist):
        super().__init__(descriptors, blockdist)

        if self.optical_limit:
            self.head_and_wings = HeadAndWingsData(self.descriptors)

    @property
    def optical_limit(self):
        return self.descriptors.optical_limit

    @property
    def Wvv_shape(self):
        if self.optical_limit:
            return self.head_and_wings.Wvv_shape

    @property
    def WxvG_shape(self):
        if self.optical_limit:
            return self.head_and_wings.WxvG_shape

    def copy_with_reduced_pd(self, pd):
        """Provide a copy of the object within a reduced plane-wave basis.
        """
        descriptors = Chi0Descriptors(self.wd, pd)
        # Create a new AugmentedBodyData object
        new_abd = self._new(descriptors, self.blockdist)

        new_abd.data_WgG[:] = map_WgG_array_to_reduced_pd(self.pd, pd,
                                                          self.blockdist,
                                                          self.data_WgG)
        if self.optical_limit:
            new_abd.head_and_wings.data_Wvv[:] = self.head_and_wings.data_Wvv

            # Map the wings to the reduced plane-wave description
            G2_G1 = PWMapping(pd, self.pd).G2_G1
            new_abd.head_and_wings.data_WxvG[:] \
                = self.head_and_wings.data_WxvG[..., G2_G1]

        return new_abd

    @classmethod
    def _new(cls, *args, **kwargs):
        return cls(*args, **kwargs)


class Chi0Data(AugmentedBodyData):
    """Data object containing the chi0 data arrays for a single q-point,
    while holding also the corresponding basis descriptors and block
    distributor."""

    @property
    def chi0_WgG(self):
        return self.data_WgG

    @property
    def chi0_Wvv(self):
        if self.optical_limit:
            return self.head_and_wings.data_Wvv

    @property
    def chi0_WxvG(self):
        if self.optical_limit:
            return self.head_and_wings.data_WxvG
