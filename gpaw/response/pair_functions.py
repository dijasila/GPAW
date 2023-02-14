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

    def copy_with(self, ecut=None, gd=None, gammacentered=None):
        if ecut is None:
            ecut = self.ecut
        if gd is None:
            gd = self.gd
        if gammacentered is None:
            gammacentered = self.gammacentered

        return SingleQPWDescriptor.from_q(
            self.q_c, ecut, gd, gammacentered=gammacentered)


class PairFunction(ABC):
    """Pair function data object.

    See gpaw.response.pair_integrator.PairFunctionIntegrator for the definition
    of a pair function and how it is calculated."""

    def __init__(self, qpd):
        """Construct a pair function.

        Parameters
        ----------
        qpd : SingleQPWDescriptor
        """
        self.qpd = qpd
        self.q_c = qpd.q_c

        self.array = self.zeros()

    @abstractmethod
    def zeros(self):
        """Generate an array of zeros, representing the pair function."""


class LatticePeriodicPairFunction(PairFunction):
    r"""Data object for lattice periodic pair functions.

    A pair function is considered to be lattice periodic, if it is invariant
    under translations of Bravais lattice vectors R:

    pf(r, r', z) = pf(r + R, r' + R, z).

    The Bloch lattice Fourier transform of a lattice periodic pair function,
                        __
                        \
    pf(r, r', q, z)  =  /  e^(-iq.[r-r'-R']) pf(r, r' + R', z)
                        ‾‾
                        R'

    is then periodic in both r and r' independently and can be expressed in an
    arbitrary lattice periodic basis.

    In the GPAW response code, lattice periodic pair functions are expanded in
    plane waves:

                   1   //
    pf_GG'(q, z) = ‾‾ || drdr' e^(-iG.r) pf(r, r', q, z) e^(iG'.r')
                   V0 //
                        V0

    Hence, the collection consists of a complex frequency descriptor and a
    plane-wave descriptor, where the latter is specific to the q-point in
    question.
    """

    def __init__(self, qpd, zd, blockdist, distribution='ZgG'):
        """Contruct the LatticePeriodicPairFunction.

        Parameters
        ----------
        qpd : SingleQPWDescriptor
        zd : ComplexFrequencyDescriptor
        blockdist : PlaneWaveBlockDistributor
        distribution : str
            Memory distribution of the pair function array.
            Choices: 'ZgG', 'GZg' and 'zGG'.
        """
        self.zd = zd
        self.blockdist = blockdist
        self.distribution = distribution

        nG = qpd.ngmax
        self.blocks1d, self.shape = self._get_blocks_and_shape(nG)

        super().__init__(qpd)

    def _get_blocks_and_shape(self, nG):
        """Get 1D block distribution and array shape

        Parameters
        ----------
        nG : int
            Number of plane-wave coefficients in the basis set
        """
        nz = len(self.zd)
        blockdist = self.blockdist
        distribution = self.distribution

        if distribution == 'ZgG':
            blocks1d = Blocks1D(blockdist.blockcomm, nG)
            shape = (nz, blocks1d.nlocal, nG)
        elif distribution == 'GZg':
            blocks1d = Blocks1D(blockdist.blockcomm, nG)
            shape = (nG, nz, blocks1d.nlocal)
        elif distribution == 'zGG':
            blocks1d = Blocks1D(blockdist.blockcomm, nz)
            shape = (blocks1d.nlocal, nG, nG)
        else:
            raise NotImplementedError(f'Distribution: {distribution}')

        return blocks1d, shape

    def zeros(self):
        return np.zeros(self.shape, complex)

    def array_with_view(self, view):
        """Access a given view into the pair function array."""
        if view == 'ZgG' and self.distribution in ['ZgG', 'GZg']:
            if self.distribution == 'GZg':
                pf_GZg = self.array
                pf_ZgG = pf_GZg.transpose((1, 2, 0))
            else:
                pf_ZgG = self.array

            pf_x = pf_ZgG
        else:
            raise ValueError(f'{view} is not a valid view, when array is of '
                             f'distribution {self.distribution}')

        return pf_x

    def copy_with_distribution(self, distribution='ZgG'):
        """Copy the pair function to a specified memory distribution."""
        new_pf = self._new(*self.my_args(), distribution=distribution)
        new_pf.array[:] = self.array_with_view(distribution)

        return new_pf

    @classmethod
    def _new(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
    def my_args(self, qpd=None, zd=None, blockdist=None):
        """Return construction arguments of the LatticePeriodicPairFunction."""
        if qpd is None:
            qpd = self.qpd
        if zd is None:
            zd = self.zd
        if blockdist is None:
            blockdist = self.blockdist

        return qpd, zd, blockdist

    def copy_with_reduced_pd(self, qpd):
        """Copy the pair function, but within a reduced plane-wave basis."""
        if self.distribution != 'ZgG':
            raise NotImplementedError('Not implemented for distribution '
                                      f'{self.distribution}')

        new_pf = self._new(*self.my_args(qpd=qpd),
                           distribution=self.distribution)
        new_pf.array[:] = map_WgG_array_to_reduced_pd(self.qpd, qpd,
                                                      self.blockdist,
                                                      self.array)

        return new_pf

    def copy_with_global_frequency_distribution(self):
        """Copy the pair function, but with distribution zGG over world."""
        # Make a copy, which is globally block distributed
        blockdist = self.blockdist.new_distributor(nblocks='max')
        new_pf = self._new(*self.my_args(blockdist=blockdist),
                           distribution='zGG')

        # Redistribute the data, distributing the frequencies over world
        assert self.distribution == 'ZgG'
        new_pf.array[:] = self.blockdist.distribute_frequencies(self.array,
                                                                len(self.zd))

        return new_pf


def map_WgG_array_to_reduced_pd(qpdi, qpd, blockdist, in_WgG):
    """Map an output array to a reduced plane wave basis which is
    completely contained within the original basis, that is, from qpdi to
    qpd."""
    from gpaw.pw.descriptor import PWMapping

    # Initialize the basis mapping
    pwmapping = PWMapping(qpdi, qpd)
    G2_GG = tuple(np.meshgrid(pwmapping.G2_G1, pwmapping.G2_G1,
                              indexing='ij'))
    G1_GG = tuple(np.meshgrid(pwmapping.G1, pwmapping.G1,
                              indexing='ij'))

    # Distribute over frequencies
    nw = in_WgG.shape[0]
    tmp_wGG = blockdist.distribute_as(in_WgG, nw, 'wGG')

    # Allocate array in the new basis
    nG = qpd.ngmax
    new_tmp_shape = (tmp_wGG.shape[0], nG, nG)
    new_tmp_wGG = np.zeros(new_tmp_shape, complex)

    # Extract values in the global basis
    for w, tmp_GG in enumerate(tmp_wGG):
        new_tmp_wGG[w][G2_GG] = tmp_GG[G1_GG]

    # Distribute over plane waves
    out_WgG = blockdist.distribute_as(new_tmp_wGG, nw, 'WgG')

    return out_WgG
