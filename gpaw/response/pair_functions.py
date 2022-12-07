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


class PairFunction(ABC):
    """
    Some documentation here!                                                   XXX
    """
    def __init__(self, pd):
        """Some documentation here!                                            XXX

        Parameters
        ----------
        pd : SingleQPWDescriptor
        """
        self.pd = pd
        self.q_c = pd.q_c

        self.array = self.zeros()

    @abstractmethod
    def zeros(self):
        """
        Document me!                                                           XXX
        """


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
        """
        Some documentation here!                                               XXX
        """
        self.wd = wd
        self.blockdist = blockdist
        self.distribution = distribution

        nG = pd.ngmax
        self.blocks1d, self.shape = self._get_blocks_and_shape(nG)

        super().__init__(pd)

    def _get_blocks_and_shape(self, nG):
        """
        Some documentation here!                                               XXX
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
