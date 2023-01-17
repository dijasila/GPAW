"""Contains methods for calculating local LR-TDDFT kernels."""

from functools import partial

import numpy as np

from gpaw.response import ResponseGroundStateAdapter, ResponseContext, timer
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


class FXCFactory:
    """Exchange-correlation kernel factory."""

    def __init__(self,
                 gs: ResponseGroundStateAdapter,
                 context: ResponseContext):
        self.gs = gs
        self.context = context

    def __call__(self, fxc, chiks: ChiKS,
                 calculator=None,
                 fxc_scaling=None):
        """Get the xc kernel Kxc_GG.

        Parameters
        ----------
        fxc : str
            Approximation to the (local) xc kernel.
            Choices: ALDA, ALDA_X, ALDA_x
        calculator : dict (or None for default calculator)
            Parameters to set up the FXCCalculator. The 'method' key
            determines what calculator is initilized and remaining parameters
            are passed to the calculator as key-word arguments.
        fxc_scaling : None or FXCScaling
        """
        if calculator is None:
            localft_calc = LocalFTCalculator.from_rshe_parameters(
                self.gs, self.context,
                rshelmax=-1,
                rshewmin=None)
            calculator = {'method': 'new',
                          'localft_calc': localft_calc}
        assert isinstance(calculator, dict) and 'method' in calculator

        # Generate the desired calculator
        calc_kwargs = calculator.copy()
        method = calc_kwargs.pop('method')
        fxc_calculator = self.get_fxc_calculator(method=method, **calc_kwargs)

        Kxc_GG = fxc_calculator(fxc, chiks.spincomponent, chiks.pd)

        if fxc_scaling is not None:
            if not fxc_scaling.has_scaling:
                fxc_scaling.calculate_scaling(chiks, Kxc_GG)
            lambd = fxc_scaling.get_scaling()
            self.context.print(r'Rescaling the xc-kernel by a factor of λ='
                               f'{lambd}')
            Kxc_GG *= lambd

        return Kxc_GG

    def get_fxc_calculator(self, *, method, **calc_kwargs):
        """Factory function for initializing fxc calculators."""
        fxc_calculator_cls = self.get_fxc_calculator_cls(method)

        return fxc_calculator_cls(**calc_kwargs)

    @staticmethod
    def get_fxc_calculator_cls(method):
        """Factory function for selecting fxc calculators."""
        if method == 'new':
            return NewAdiabaticFXCCalculator

        raise ValueError(f'Invalid fxc calculator method {method}')


class NewAdiabaticFXCCalculator:
    """Calculator for adiabatic local exchange-correlation kernels."""

    def __init__(self, localft_calc: LocalFTCalculator):
        """Contruct the fxc calculator based on a local FT calculator."""
        self.localft_calc = localft_calc

        self.gs = localft_calc.gs
        self.context = localft_calc.context

    @timer('Calculate XC kernel')
    def __call__(self, fxc, spincomponent, pd):
        """Calculate the xc kernel matrix Kxc_GG' = 1 / V0 * fxc(G-G')."""
        # Generate a large_pd to encompass all G-G' in pd
        large_ecut = 4 * pd.ecut  # G = 1D grid of |G|^2/2 < ecut
        large_pd = pd.copy_with(ecut=large_ecut,
                                gammacentered=True,
                                gd=self.gs.finegd)
        
        # Calculate fxc(Q) on the large plane-wave grid (Q = large grid index)
        add_fxc = create_add_fxc(fxc, spincomponent)
        fxc_Q = self.localft_calc(large_pd, add_fxc)

        # Unfold the kernel according to Kxc_GG' = 1 / V0 * fxc(G-G')
        Kxc_GG = 1 / pd.gd.volume * self.unfold_kernel_matrix(
            pd, large_pd, fxc_Q)

        return Kxc_GG

    @timer('Unfold kernel matrix')
    def unfold_kernel_matrix(self, pd, large_pd, fxc_Q):
        """Unfold the kernel fxc(Q) to the kernel matrix fxc_GG'=fxc(G-G')"""
        # Calculate (G-G') reciprocal space vectors
        dG_GGv = calculate_dG(pd)
        GG_shape = dG_GGv.shape[:2]

        # Reshape to composite K = (G, G') index
        dG_Kv = dG_GGv.reshape(-1, dG_GGv.shape[-1])

        # Find unique dG-vectors
        # We need tight control of the decimals to avoid precision artifacts
        dG_dGv, dG_K = np.unique(dG_Kv.round(decimals=6),
                                 return_inverse=True, axis=0)

        # Map fxc(Q) onto fxc(G-G') index dG
        Q_dG = self.get_Q_dG_map(large_pd, dG_dGv)
        fxc_dG = fxc_Q[Q_dG]

        # Unfold fxc(G-G') to fxc_GG'
        fxc_GG = fxc_dG[dG_K].reshape(GG_shape)

        return fxc_GG

    @staticmethod
    def get_Q_dG_map(large_pd, dG_dGv):
        """Create mapping between (G-G') index dG and large_pd index Q."""
        G_Qv = large_pd.get_reciprocal_vectors(add_q=False)
        # Make sure to match the precision of dG_dGv
        G_Qv = G_Qv.round(decimals=6)

        diff_QdG = np.linalg.norm(G_Qv[:, np.newaxis] - dG_dGv[np.newaxis],
                                  axis=2)
        Q_dG = np.argmin(diff_QdG, axis=0)

        # Check that all the identified Q indeces produce identical reciprocal
        # lattice vectors
        assert np.allclose(np.diagonal(diff_QdG[Q_dG]), 0.),\
            'Could not find a perfect matching reciprocal wave vector in '\
            'large_pd for all dG_dGv'

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


def calculate_dG(pd):
    """Calculate dG_GG' = (G-G') for the plane wave basis in pd."""
    nG = pd.ngmax
    G_Gv = pd.get_reciprocal_vectors(add_q=False)

    dG_GGv = np.zeros((nG, nG, 3))
    for v in range(3):
        dG_GGv[:, :, v] = np.subtract.outer(G_Gv[:, v], G_Gv[:, v])

    return dG_GGv
