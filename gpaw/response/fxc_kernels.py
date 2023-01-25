"""Contains methods for calculating local LR-TDDFT kernels."""

from functools import partial

import numpy as np

from gpaw.response import timer
from gpaw.response.pw_parallelization import Blocks1D
from gpaw.response.chiks import ChiKS
from gpaw.response.goldstone import get_goldstone_scaling
from gpaw.response.localft import (LocalFTCalculator,
                                   add_LDA_dens_fxc, add_LSDA_trans_fxc)


class FXCScaling:
    """Helper for scaling fxc kernels."""

    def __init__(self, mode, lambd=None):
        self.mode = mode
        self.lambd = lambd

    @property
    def has_scaling(self):
        return self.lambd is not None

    def get_scaling(self):
        return self.lambd

    def calculate_scaling(self, chiks, Kxc_GG):
        if chiks.spincomponent in ['+-', '-+']:
            self.lambd = get_goldstone_scaling(self.mode, chiks, Kxc_GG)
        else:
            raise ValueError('No scaling method implemented for '
                             f'spincomponent={chiks.spincomponent}')


def fxc_factory(fxc, chiks: ChiKS, localft_calc: LocalFTCalculator,
                fxc_scaling=None):
    """Local exchange-correlation kernel factory.

    Parameters
    ----------
    fxc : str
        Approximation to the (local) xc kernel.
        Choices: ALDA, ALDA_X, ALDA_x
    fxc_scaling : None or FXCScaling
    """
    # Calculate the xc kernel Kxc_GG
    fxc_calculator = AdiabaticFXCCalculator(localft_calc)
    Kxc_GG = fxc_calculator(fxc, chiks.spincomponent, chiks.qpd)

    if fxc_scaling is not None:
        if not fxc_scaling.has_scaling:
            fxc_scaling.calculate_scaling(chiks, Kxc_GG)
        lambd = fxc_scaling.get_scaling()
        fxc_calculator.context.print(r'Rescaling the xc-kernel by a factor'
                                     f' of λ={lambd}')
        Kxc_GG *= lambd

    return Kxc_GG


class AdiabaticFXCCalculator:
    """Calculator for adiabatic local exchange-correlation kernels."""

    def __init__(self, localft_calc: LocalFTCalculator):
        """Contruct the fxc calculator based on a local FT calculator."""
        self.localft_calc = localft_calc

        self.gs = localft_calc.gs
        self.context = localft_calc.context

    @timer('Calculate XC kernel')
    def __call__(self, fxc, spincomponent, qpd):
        """Calculate the xc kernel matrix Kxc_GG' = 1 / V0 * fxc(G-G')."""
        # Generate a large_qpd to encompass all G-G' in qpd
        large_ecut = 4 * qpd.ecut  # G = 1D grid of |G|^2/2 < ecut
        large_qpd = qpd.copy_with(ecut=large_ecut,
                                  gammacentered=True,
                                  gd=self.gs.finegd)
        
        # Calculate fxc(Q) on the large plane-wave grid (Q = large grid index)
        add_fxc = create_add_fxc(fxc, spincomponent)
        fxc_Q = self.localft_calc(large_qpd, add_fxc)

        # Unfold the kernel according to Kxc_GG' = 1 / V0 * fxc(G-G')
        Kxc_GG = 1 / qpd.gd.volume * self.unfold_kernel_matrix(
            qpd, large_qpd, fxc_Q)

        return Kxc_GG

    @timer('Unfold kernel matrix')
    def unfold_kernel_matrix(self, qpd, large_qpd, fxc_Q):
        """Unfold the kernel fxc(Q) to the kernel matrix fxc_GG'=fxc(G-G')"""
        # Calculate (G-G') reciprocal space vectors
        dG_GGv = calculate_dG(qpd)
        GG_shape = dG_GGv.shape[:2]

        # Reshape to composite K = (G, G') index
        dG_Kv = dG_GGv.reshape(-1, dG_GGv.shape[-1])

        # Find unique dG-vectors
        # We need tight control of the decimals to avoid precision artifacts
        dG_dGv, dG_K = np.unique(dG_Kv.round(decimals=6),
                                 return_inverse=True, axis=0)

        # Map fxc(Q) onto fxc(G-G') index dG
        Q_dG = self.get_Q_dG_map(large_qpd, dG_dGv)
        fxc_dG = fxc_Q[Q_dG]

        # Unfold fxc(G-G') to fxc_GG'
        fxc_GG = fxc_dG[dG_K].reshape(GG_shape)

        return fxc_GG

    def get_Q_dG_map(self, large_qpd, dG_dGv):
        """Create mapping between (G-G') index dG and large_qpd index Q."""
        G_Qv = large_qpd.get_reciprocal_vectors(add_q=False)
        # Make sure to match the precision of dG_dGv
        G_Qv = G_Qv.round(decimals=6)

        # Distribute dG over world
        # This is necessary because the next step is to create a K_QdGv buffer
        # of which the norm is taken. When the number of plane-wave
        # coefficients is large, this step becomes a memory bottleneck, hence
        # the distribution.
        dGblocks = Blocks1D(self.context.world, dG_dGv.shape[0])
        dG_mydGv = dG_dGv[dGblocks.myslice]

        # Determine Q index for each dG index
        diff_QmydG = np.linalg.norm(G_Qv[:, np.newaxis] - dG_mydGv[np.newaxis],
                                    axis=2)
        Q_mydG = np.argmin(diff_QmydG, axis=0)

        # Check that all the identified Q indices produce identical reciprocal
        # lattice vectors
        assert np.allclose(np.diagonal(diff_QmydG[Q_mydG]), 0.),\
            'Could not find a perfect matching reciprocal wave vector in '\
            'large_qpd for all dG_dGv'

        # Collect the global Q_dG map
        Q_dG = dGblocks.collect(Q_mydG)

        return Q_dG


def create_add_fxc(fxc, spincomponent):
    """Create an add_fxc function according to the requested functional and
    spin component."""
    assert fxc in ['ALDA_x', 'ALDA_X', 'ALDA']

    if spincomponent in ['00', 'uu', 'dd']:
        add_fxc = partial(add_LDA_dens_fxc, fxc=fxc)
    elif spincomponent in ['+-', '-+']:
        add_fxc = partial(add_LSDA_trans_fxc, fxc=fxc)
    else:
        raise ValueError(spincomponent)

    return add_fxc


def calculate_dG(qpd):
    """Calculate dG_GG' = (G-G') for the plane wave basis in qpd."""
    nG = qpd.ngmax
    G_Gv = qpd.get_reciprocal_vectors(add_q=False)

    dG_GGv = np.zeros((nG, nG, 3))
    for v in range(3):
        dG_GGv[:, :, v] = np.subtract.outer(G_Gv[:, v], G_Gv[:, v])

    return dG_GGv
