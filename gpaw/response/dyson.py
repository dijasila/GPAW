from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from gpaw.typing import Array1D
from gpaw.response import timer
from gpaw.response.pair_functions import Chi
from gpaw.response.fxc_kernels import FXCKernel


class HXCScaling(ABC):
    """Helper for scaling the hxc contribution to Dyson equations."""

    def __init__(self, lambd=None):
        self._lambd = lambd

    @property
    def lambd(self):
        return self._lambd

    def calculate_scaling(self, dyson_equations):
        self._lambd = self._calculate_scaling(dyson_equations)

    @abstractmethod
    def _calculate_scaling(self, dyson_equations: DysonEquations) -> float:
        """Calculate hxc scaling coefficient."""


class HXCKernel:
    """Hartree-exchange-correlation kernel in a plane-wave basis."""

    def __init__(self,
                 Vbare_G: Array1D | None = None,
                 fxc_kernel: FXCKernel | None = None):
        """Construct the Hxc kernel."""
        self.Vbare_G = Vbare_G
        self.fxc_kernel = fxc_kernel

        if Vbare_G is None:
            assert fxc_kernel is not None
            self.nG = fxc_kernel.GG_shape[0]
        else:
            self.nG = len(Vbare_G)
            if fxc_kernel is not None:
                assert fxc_kernel.GG_shape[0] == self.nG

    def get_Khxc_GG(self):
        """Hartree-exchange-correlation kernel."""
        # Allocate array
        Khxc_GG = np.zeros((self.nG, self.nG), dtype=complex)
        if self.Vbare_G is not None:  # Add the Hartree kernel
            Khxc_GG.flat[::self.nG + 1] += self.Vbare_G
        if self.fxc_kernel is not None:  # Add the xc kernel
            # Unfold the fxc kernel into the Kxc kernel matrix
            Khxc_GG += self.fxc_kernel.get_Kxc_GG()
        return Khxc_GG


class DysonSolver:
    """Class for inversion of Dyson-like equations."""

    def __init__(self, context):
        self.context = context

    @timer('Invert Dyson-like equations')
    def __call__(self, chiks: Chi, hxc_kernel: HXCKernel,
                 hxc_scaling: HXCScaling | None = None) -> Chi:
        """Solve the dyson equation and return the many-body susceptibility."""
        dyson_equations = DysonEquations(chiks, hxc_kernel)
        if hxc_scaling:
            if not hxc_scaling.lambd:  # calculate, if it hasn't been already
                hxc_scaling.calculate_scaling(dyson_equations)
            lambd = hxc_scaling.lambd
            self.context.print(r'Rescaling the self-enhancement function by a '
                               f'factor of λ={lambd}')
        self.context.print('Inverting Dyson-like equations')
        return dyson_equations.invert(hxc_scaling=hxc_scaling)


class DysonEquations(Sequence):
    """Sequence of Dyson-like equations at different complex frequencies z."""

    def __init__(self, chiks: Chi, hxc_kernel: HXCKernel):
        assert chiks.distribution == 'zGG' and\
            chiks.blockdist.fully_block_distributed, \
            "chiks' frequencies need to be fully distributed over world"
        nG = hxc_kernel.nG
        assert chiks.array.shape[1:] == (nG, nG)
        self.chiks = chiks
        self.Khxc_GG = hxc_kernel.get_Khxc_GG()
        # Inherit basic descriptors from chiks
        self.qpd = chiks.qpd
        self.zd = chiks.zd
        self.zblocks = chiks.blocks1d
        self.spincomponent = chiks.spincomponent

    def __len__(self):
        return self.zblocks.nlocal

    def __getitem__(self, z):
        chiks_GG = self.chiks.array[z]
        xi_GG = chiks_GG @ self.Khxc_GG
        return DysonEquation(chiks_GG, xi_GG)

    def invert(self, hxc_scaling: HXCScaling | None = None) -> Chi:
        """Invert Dyson equations to obtain χ(z)."""
        # Scaling coefficient of the self-enhancement function
        lambd = hxc_scaling.lambd if hxc_scaling else None
        chi = self.chiks.new()
        for z, dyson_equation in enumerate(self):
            chi.array[z] = dyson_equation.invert(lambd=lambd)
        return chi


class DysonEquation:
    """Dyson equation at wave vector q and frequency z.

    The Dyson equation is given in plane-wave components as

    χ(q,z) = χ_KS(q,z) + Ξ(q,z) χ(q,z),

    where the self-enhancement function encodes the electron correlations
    induced by by the effective (Hartree-exchange-correlation) interaction:

    Ξ(q,z) = χ_KS(q,z) K_hxc(q,z)

    See [to be published] for more information.
    """

    def __init__(self, chiks_GG, xi_GG):
        self.nG = chiks_GG.shape[0]
        self.chiks_GG = chiks_GG
        self.xi_GG = xi_GG

    def invert(self, lambd: float | None = None):
        """Invert the Dyson equation (with or without a rescaling of Ξ).

        χ(q,z) = [1 - λ Ξ(q,z)]^(-1) χ_KS(q,z)
        """
        if lambd is None:
            lambd = 1.  # no rescaling
        enhancement_GG = np.linalg.inv(
            np.eye(self.nG) - lambd * self.xi_GG)
        return enhancement_GG @ self.chiks_GG


class DysonEnhancer:
    """Class for applying self-enhancement functions."""
    def __init__(self, context):
        self.context = context

    def __call__(self, chiks: Chi, xi: Chi) -> Chi:
        """Solve the Dyson equation and return the many-body susceptibility."""
        assert chiks.distribution == 'zGG' and \
            chiks.blockdist.fully_block_distributed
        assert xi.distribution == 'zGG' and \
            xi.blockdist.fully_block_distributed
        assert chiks.spincomponent == xi.spincomponent
        assert np.allclose(chiks.zd.hz_z, xi.zd.hz_z)
        assert np.allclose(chiks.qpd.q_c, xi.qpd.q_c)

        chi = chiks.new()
        chi.array = self.invert_dyson(chiks.array, xi.array)

        return chi

    @timer('Invert Dyson-like equation')
    def invert_dyson(self, chiks_zGG, xi_zGG):
        r"""Invert the frequency dependent Dyson equation in plane-wave basis:
                                           __
                                           \
        χ_GG'^+-(q,z) = χ_KS,GG'^+-(q,z) + /  Ξ_GG1^++(q,z) χ_G1G'^+-(q,z)
                                           ‾‾
                                           G1
        """
        self.context.print('Inverting Dyson-like equation')
        chi_zGG = np.empty_like(chiks_zGG)
        for chi_GG, chiks_GG, xi_GG in zip(chi_zGG, chiks_zGG, xi_zGG):
            chi_GG[:] = self.invert_dyson_single_frequency(chiks_GG, xi_GG)
        return chi_zGG

    @staticmethod
    def invert_dyson_single_frequency(chiks_GG, xi_GG):
        enhancement_GG = np.linalg.inv(np.eye(len(chiks_GG)) - xi_GG)
        chi_GG = enhancement_GG @ chiks_GG
        return chi_GG
