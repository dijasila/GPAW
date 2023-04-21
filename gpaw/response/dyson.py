from __future__ import annotations

import numpy as np

from gpaw.response import timer
from gpaw.response.chiks import ChiKS
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
                 Vbare_G,
                 fxc_kernel: FXCKernel | None,
                 scaling: HXCScaling | None = None):
        """Construct the Hxc kernel."""
        self.Vbare_G = Vbare_G
        self.fxc_kernel = fxc_kernel
        self.scaling = scaling

        if Vbare_G is None:
            self.nG = fxc_kernel.GG_shape[0]
        else:
            self.nG = len(Vbare_G)

    def get_Khxc_GG(self):
        """Hartree-exchange-correlation kernel."""
        # Allocate array
        Khxc_GG = np.zeros((self.nG, self.nG), dtype=complex)

        if self.Vbare_G is not None:  # Add the Hartree kernel
            Khxc_GG.flat[::self.nG + 1] += self.Vbare_G

        if self.fxc_kernel is not None:  # Add the xc kernel
            # Unfold the fxc kernel into the Kxc kernel matrix
            Khxc_GG += self.fxc_kernel.get_Kxc_GG()

        # Apply kernel scaling, if such a scaling parameter exists
        if self.scaling is not None and self.scaling.lambd is not None:
            Khxc_GG *= self.scaling.lambd

        return Khxc_GG


class DysonSolver:
    """Class for invertion of Dyson-like equations."""

    def __init__(self, context):
        self.context = context

    def __call__(self, chiks: ChiKS, hxc_kernel: HXCKernel):
        """Solve the dyson equation and return chi."""
        assert chiks.distribution == 'zGG' and\
            chiks.blockdist.fully_block_distributed,\
            "Chi assumes that chiks's frequencies are distributed over world"

        Khxc_GG = hxc_kernel.get_Khxc_GG()

        # Calculate kernel scaling, if specified
        hxc_scaling = hxc_kernel.scaling
        if hxc_scaling is not None and hxc_scaling.lambd is None:
            hxc_scaling.calculate_scaling(chiks, Khxc_GG, self)
            lambd = hxc_scaling.lambd
            self.context.print(r'Rescaling the xc kernel by a factor of '
                               f'λ={lambd}')
            Khxc_GG *= lambd

        return self.invert_dyson(chiks.array, Khxc_GG)

    @timer('Invert Dyson-like equation')
    def invert_dyson(self, chiks_wGG, Khxc_GG):
        """Invert the frequency dependent Dyson equation in plane wave basis:

        chi_wGG' = chiks_wGG' + chiks_wGG1 Khxc_G1G2 chi_wG2G'
        """
        self.context.print('Inverting Dyson-like equation')
        chi_wGG = np.empty_like(chiks_wGG)
        for w, chiks_GG in enumerate(chiks_wGG):
            chi_GG = self.invert_dyson_single_frequency(chiks_GG, Khxc_GG)

            chi_wGG[w] = chi_GG

        return chi_wGG

    @staticmethod
    def invert_dyson_single_frequency(chiks_GG, Khxc_GG):
        """Invert the single frequency Dyson equation in plane wave basis:

        chi_GG' = chiks_GG' + chiks_GG1 Khxc_G1G2 chi_G2G'
        """
        enhancement_GG = np.linalg.inv(np.eye(len(chiks_GG)) -
                                       np.dot(chiks_GG, Khxc_GG))
        chi_GG = enhancement_GG @ chiks_GG

        return chi_GG
