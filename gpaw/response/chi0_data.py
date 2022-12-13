import numpy as np

from gpaw.response.pw_parallelization import (Blocks1D,
                                              PlaneWaveBlockDistributor)
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.pair_functions import SingleQPWDescriptor


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
        self.data_wGG = None
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
        self.data_wGG = np.zeros(self.wGG_shape, complex)
        
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
        """Return data_wGG array with frequencies distributed to all cores."""
        return self.blockdist.distribute_frequencies(self.data_wGG, self.nw)

    def distribute_as(self, out_dist):
        """Distribute self.data_wGG as given in out_dist.
        out_dist: str 'wGG' for parallell over w and
        'WgG' for parallel over G"""
        return self.blockdist.distribute_as(self.data_wGG, self.nw, out_dist)

    def check_distribution(self, test_dist):
        """Checks if self.data_wGG is distributed according to test_dist"""
        _, __, same_dist = self.blockdist.check_distribution(self.data_wGG,
                                                             self.nw,
                                                             test_dist)
        return same_dist

    def get_reduced_ecut_array(self, ecut):
        """Provide a copy of the body data array within a reduced ecut.

        Needed for ecut extrapolation in G0W0.
        Note: Returns data_wGG array in wGG distribution.
        """
        pdr, pw_map = self.get_pw_reduction_map(ecut)
        data_wGG = self._copy_and_map_array(pw_map)

        return pdr, data_wGG

    def _copy_and_map_array(self, pw_map):
        # Get a copy of the full array, distributed over frequencies
        data_wGG = self.blockdist.distribute_as(self.data_wGG,
                                                self.nw, 'wGG')

        if pw_map is not None:
            G2_G1 = pw_map.G2_G1
            # Construct array subset with lower ecut
            data_wGG = data_wGG.take(G2_G1, axis=1).take(G2_G1, axis=2)

        if data_wGG is self.data_wGG:
            # If we were already frequency distributed and no real
            # reduction is happening, we may point to the original array,
            # but we want strictly to return a copy
            data_wGG = self.data_wGG.copy()

        return data_wGG

    def get_pw_reduction_map(self, ecut):
        """Get PWMapping to reduce plane-wave description."""
        from gpaw.pw.descriptor import PWMapping

        pd = self.pd

        if ecut == self.pd.ecut:
            pdr = pd  # reduced pd is equal to the original pd
            pw_map = None
        elif ecut < self.pd.ecut:
            # Create reduced pd
            pdr = SingleQPWDescriptor.from_q(
                pd.q_c, ecut, pd.gd, gammacentered=pd.gammacentered)
            pw_map = PWMapping(pdr, pd)

        return pdr, pw_map


class HeadAndWingsData:
    def __init__(self, descriptors):
        assert descriptors.optical_limit
        self.wd = descriptors.wd
        self.pd = descriptors.pd
        self.data_wxvG = None  # Wings
        self.data_wvv = None  # Head
        self.allocate_arrays()
        
    def allocate_arrays(self):
        self.data_wxvG = np.zeros(self.wxvG_shape, complex)
        self.data_wvv = np.zeros(self.wvv_shape, complex)

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

    def _copy_and_map_arrays(self, pw_map):
        data_wxvG = self.data_wxvG.copy()
        if pw_map is not None:
            data_wxvG = data_wxvG.take(pw_map.G2_G1, axis=3)
        data_wvv = self.data_wvv.copy()

        return data_wxvG, data_wvv


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
    def wxvG_shape(self):
        if self.optical_limit:
            return self.head_and_wings.wxvG_shape

    @property
    def wvv_shape(self):
        if self.optical_limit:
            return self.head_and_wings.wvv_shape

    def get_reduced_ecut_arrays(self, ecut):
        """Provide a copy of the data array(s) within a reduced ecut.
        """
        pdr, pw_map = self.get_pw_reduction_map(ecut)
        data_wGG = self._copy_and_map_array(pw_map)

        if self.optical_limit:
            data_wxvG, data_wvv = self.head_and_wings._copy_and_map_arrays(
                pw_map)
        else:
            data_wxvG = None
            data_wvv = None

        return pdr, data_wGG, data_wxvG, data_wvv


class Chi0Data(AugmentedBodyData):
    """Data object containing the chi0 data arrays for a single q-point,
    while holding also the corresponding basis descriptors and block
    distributor."""

    @property
    def chi0_wGG(self):
        return self.data_wGG

    @property
    def chi0_wxvG(self):
        if self.optical_limit:
            return self.head_and_wings.data_wxvG

    @property
    def chi0_wvv(self):
        if self.optical_limit:
            return self.head_and_wings.data_wvv
