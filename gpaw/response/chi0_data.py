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
        """Return 'wGG'-like array, with frequencies distributed over world.

        This is differs from get_array_distributed_as('wGG'), in that the
        frequencies are distributed over world, instead of among the block
        communicator."""
        return self.blockdist.distribute_frequencies(self.data_WgG, self.nw)

    def get_array_distributed_as(self, distribution):
        """Distribute self.data_WgG as given in out_dist.

        Parameters
        ----------
        distribution: str
            Array distribution. Choices: 'wGG' and 'WgG'
        """
        # This function is quite dangerous, since it sometimes returns a copy,
        # sometimes self.data_WgG... Should be changed to a true copy in the
        # future? XXX
        return self.blockdist.distribute_as(self.data_WgG, self.nw,
                                            distribution)

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
        data_wGG = self.get_array_distributed_as('wGG')

        if data_wGG is self.data_WgG:
            # If we were already frequency distributed (because there is no
            # block distribution at all), we may still be pointing to the
            # original array, but we want strictly to return a copy
            assert self.blockdist.blockcomm.size == 1
            data_wGG = self.data_WgG.copy()

        if pw_map is not None:
            G2_G1 = pw_map.G2_G1
            # Construct array subset with lower ecut
            data_wGG = data_wGG.take(G2_G1, axis=1).take(G2_G1, axis=2)

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

    def _copy_and_map_arrays(self, pw_map):
        data_Wvv = self.data_Wvv.copy()
        data_WxvG = self.data_WxvG.copy()
        if pw_map is not None:
            data_WxvG = data_WxvG.take(pw_map.G2_G1, axis=3)

        return data_Wvv, data_WxvG  # head and wings


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

    def get_reduced_ecut_arrays(self, ecut):
        """Provide a copy of the data array(s) within a reduced ecut.
        """
        pdr, pw_map = self.get_pw_reduction_map(ecut)
        data_wGG = self._copy_and_map_array(pw_map)

        if self.optical_limit:
            data_Wvv, data_WxvG = self.head_and_wings._copy_and_map_arrays(
                pw_map)
        else:
            data_Wvv = None
            data_WxvG = None

        return pdr, data_wGG, data_Wvv, data_WxvG


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
