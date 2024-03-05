from __future__ import annotations

import numpy as np

from gpaw.typing import Array1D
from gpaw.response import timer
from gpaw.response.pair_functions import Chi
from gpaw.response.fxc_kernels import FXCKernel
from gpaw.response.goldstone import get_goldstone_scaling


class HXCScaling:
    """Helper for scaling hxc kernels."""

    def __init__(self, mode, lambd=None):
        self.mode = mode
        self._lambd = lambd

    @property
    def lambd(self):
        return self._lambd

    def calculate_scaling(self, chiks, Khxc_GG, dyson_solver):
        if chiks.spincomponent in ['+-', '-+']:
            self._lambd = get_goldstone_scaling(
                self.mode, chiks, Khxc_GG, dyson_solver)
        else:
            raise ValueError('No scaling method implemented for '
                             f'spincomponent={chiks.spincomponent}')


class HXCKernel:
    """Hartree-exchange-correlation kernel in a plane-wave basis."""

    def __init__(self,
                 Vbare_G: Array1D | None,
                 fxc_kernel: FXCKernel | None):
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
    """Class for invertion of Dyson-like equations."""

    def __init__(self, context):
        self.context = context

    def __call__(self, chiks: Chi, hxc_kernel: HXCKernel,
                 hxc_scaling: HXCScaling | None = None) -> Chi:
        """Solve the dyson equation and return the many-body susceptibility."""
        assert chiks.distribution == 'zGG' and\
            chiks.blockdist.fully_block_distributed, \
            "DysonSolver needs chiks' frequencies to be distributed over world"

        Khxc_GG = hxc_kernel.get_Khxc_GG()

        # Apply kernel scaling, if specified
        if hxc_scaling is not None:
            if hxc_scaling.lambd is None:  # calculate, if it hasn't already
                hxc_scaling.calculate_scaling(chiks, Khxc_GG, self)
            lambd = hxc_scaling.lambd
            self.context.print(r'Rescaling the xc kernel by a factor of '
                               f'λ={lambd}')
            Khxc_GG *= lambd

        chi = chiks.new()
        chi.array = self.invert_dyson(chiks.array, Khxc_GG)

        return chi

    @timer('Invert Dyson-like equation')
    def invert_dyson(self, chiks_zGG, Khxc_GG):
        """Invert the frequency dependent Dyson equation in plane wave basis:

        chi_zGG' = chiks_zGG' + chiks_zGG1 Khxc_G1G2 chi_zG2G'
        """
        self.context.print('Inverting Dyson-like equation')
        chi_zGG = np.empty_like(chiks_zGG)
        for z, chiks_GG in enumerate(chiks_zGG):
            chi_GG = self.invert_dyson_single_frequency(chiks_GG, Khxc_GG)

            chi_zGG[z] = chi_GG

        return chi_zGG

    @staticmethod
    def invert_dyson_single_frequency(chiks_GG, Khxc_GG):
        """Invert the single frequency Dyson equation in plane wave basis:

        chi_GG' = chiks_GG' + chiks_GG1 Khxc_G1G2 chi_G2G'
        """
        enhancement_GG = np.linalg.inv(np.eye(len(chiks_GG)) -
                                       np.dot(chiks_GG, Khxc_GG))
        chi_GG = enhancement_GG @ chiks_GG

        return chi_GG


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
