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


class PairFunctionDescriptors:
    """
    Some documentation here!                                                   XXX
    """
    def __init__(self, pd):
        # Document me!                                                         XXX
        self.pd = pd

        # Extract q_c
        # Should be retrievable directly from pd in the future                 XXX
        q_c = pd.kd.bzk_kc[0]
        self.q_c = q_c


class PairFunction(ABC):
    """
    Some documentation here!                                                   XXX
    """
    def __init__(self, descriptors):
        # Document me!                                                         XXX
        self.descriptors = descriptors
        self.array = self.zeros()

    @abstractmethod
    def zeros(self):
        """
        Document me!                                                           XXX
        """


class LatticePeriodicPairFunctionDescriptors(PairFunctionDescriptors):
    """Descriptor collection for lattice periodic pair functions."""

    def __init__(self, wd, pd):
        """Construct the descriptor collection

        Parameters
        ----------
        wd : FrequencyDescriptor
        pd : PWDescriptor
        """
        super().__init__(pd)
        self.wd = wd

        # Basis set size
        self.nw = len(wd)
        self.nG = pd.ngmax


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

    def __init__(self, descriptors, blockdist, distribution='WgG'):
        """
        Some documentation here!                                               XXX
        """
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

        super().__init__(descriptors)

    def zeros(self):
        return np.zeros(self.shape, complex)
