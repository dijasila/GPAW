from abc import ABC, abstractmethod

import numpy as np

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.pw.descriptor import PWDescriptor

from gpaw.response.pw_parallelization import Blocks1D


class SingleQPWDescriptor(PWDescriptor):

    @staticmethod
    def from_q(q_c, ecut, gd, gammacentered=False):
        """Construct a plane wave descriptor for q_c with a given cutoff."""
        qd = KPointDescriptor([q_c])
        return SingleQPWDescriptor(ecut, gd, complex, qd,
                                   gammacentered=gammacentered)

    @property
    def q_c(self):
        return self.kd.bzk_kc[0]

    def copy(self):
        return self.copy_with()

    def copy_with(self, ecut=None, gammacentered=None):
        if ecut is None:
            ecut = self.ecut
        if gammacentered is None:
            gammacentered = self.gammacentered

        return SingleQPWDescriptor.from_q(
            self.q_c, ecut, self.gd, gammacentered=gammacentered)


class PairFunction(ABC):
    """Pair function data object.

    See gpaw.response.pair_integrator.PairFunctionIntegrator for the definition
    of a pair function and how it is calculated."""

    def __init__(self, pd):
        """Construct a pair function.

        Parameters
        ----------
        pd : SingleQPWDescriptor
        """
        self.pd = pd
        self.q_c = pd.q_c

        self.array = self.zeros()

    @abstractmethod
    def zeros(self):
        """Generate an array of zeros, representing the pair function."""


class LatticePeriodicPairFunction(PairFunction):
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

    def __init__(self, pd, wd, blockdist, distribution='WgG'):
        """Contruct the LatticePeriodicPairFunction.

        Parameters
        ----------
        pd : SingleQPWDescriptor
        wd : FrequencyDescriptor
        blockdist : PlaneWaveBlockDistributor
        distribution : str
            Memory distribution of the pair function array.
            Choices: 'WgG', 'GWg' and 'wGG'.
        """
        self.wd = wd
        self.blockdist = blockdist
        self.distribution = distribution

        nG = pd.ngmax
        self.blocks1d, self.shape = self._get_blocks_and_shape(nG)

        super().__init__(pd)

    def _get_blocks_and_shape(self, nG):
        """Get 1D block distribution and array shape

        Parameters
        ----------
        nG : int
            Number of plane-wave coefficients in the basis set
        """
        nw = len(self.wd)
        blockdist = self.blockdist
        distribution = self.distribution

        if distribution == 'WgG':
            blocks1d = Blocks1D(blockdist.blockcomm, nG)
            shape = (nw, blocks1d.nlocal, nG)
        elif distribution == 'GWg':
            blocks1d = Blocks1D(blockdist.blockcomm, nG)
            shape = (nG, nw, blocks1d.nlocal)
        elif distribution == 'wGG':
            blocks1d = Blocks1D(blockdist.blockcomm, nw)
            shape = (blocks1d.nlocal, nG, nG)
        else:
            raise NotImplementedError(f'Distribution: {distribution}')

        return blocks1d, shape

    def zeros(self):
        return np.zeros(self.shape, complex)

    def array_with_view(self, view):
        """Access a given view into the pair function array."""
        if view == 'WgG' and self.distribution in ['WgG', 'GWg']:
            if self.distribution == 'GWg':
                pf_GWg = self.array
                pf_WgG = pf_GWg.transpose((1, 2, 0))
            else:
                pf_WgG = self.array

            pf_x = pf_WgG
        else:
            raise ValueError(f'{view} is not a valid view, when array is of '
                             f'distribution {self.distribution}')

        return pf_x

    def copy_with_distribution(self, distribution='WgG'):
        """Copy the pair function to a specified memory distribution."""
        new_pf = self._new(*self.my_args(), distribution=distribution)
        new_pf.array[:] = self.array_with_view(distribution)

        return new_pf

    @classmethod
    def _new(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
    def my_args(self, pd=None, wd=None, blockdist=None):
        """Return construction arguments of the LatticePeriodicPairFunction."""
        if pd is None:
            pd = self.pd
        if wd is None:
            wd = self.wd
        if blockdist is None:
            blockdist = self.blockdist

        return pd, wd, blockdist

    def copy_with_reduced_pd(self, pd):
        """Copy the pair function, but within a reduced plane-wave basis."""
        if self.distribution != 'WgG':
            raise NotImplementedError('Not implemented for distribution '
                                      f'{self.distribution}')

        new_pf = self._new(*self.my_args(pd=pd),
                           distribution=self.distribution)
        new_pf.array[:] = map_WgG_array_to_reduced_pd(self.pd, pd,
                                                      self.blockdist,
                                                      self.array)

        return new_pf


def map_WgG_array_to_reduced_pd(pdi, pd, blockdist, in_WgG):
    """Map an output array to a reduced plane wave basis which is
    completely contained within the original basis, that is, from pdi to
    pd."""
    from gpaw.pw.descriptor import PWMapping

    # Initialize the basis mapping
    pwmapping = PWMapping(pdi, pd)
    G2_GG = tuple(np.meshgrid(pwmapping.G2_G1, pwmapping.G2_G1,
                              indexing='ij'))
    G1_GG = tuple(np.meshgrid(pwmapping.G1, pwmapping.G1,
                              indexing='ij'))

    # Distribute over frequencies
    nw = in_WgG.shape[0]
    tmp_wGG = blockdist.distribute_as(in_WgG, nw, 'wGG')

    # Allocate array in the new basis
    nG = pd.ngmax
    new_tmp_shape = (tmp_wGG.shape[0], nG, nG)
    new_tmp_wGG = np.zeros(new_tmp_shape, complex)

    # Extract values in the global basis
    for w, tmp_GG in enumerate(tmp_wGG):
        new_tmp_wGG[w][G2_GG] = tmp_GG[G1_GG]

    # Distribute over plane waves
    out_WgG = blockdist.distribute_as(new_tmp_wGG, nw, 'WgG')

    return out_WgG
