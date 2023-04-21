from __future__ import annotations

import numpy as np

from gpaw.response import timer
from gpaw.response.chiks import ChiKS
from gpaw.response.fxc_kernels import FXCKernel
from gpaw.response.goldstone import get_goldstone_scaling


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

    def calculate_scaling(self, chiks, Kxc_GG, dyson_solver):
        if chiks.spincomponent in ['+-', '-+']:
            self.lambd = get_goldstone_scaling(
                self.mode, chiks, Kxc_GG, dyson_solver)
        else:
            raise ValueError('No scaling method implemented for '
                             f'spincomponent={chiks.spincomponent}')


class HXCKernel:
    """Hartree-exchange-correlation kernel in a plane-wave basis."""

    def __init__(self,
                 Vbare_G,
                 fxc_kernel: FXCKernel | None,
                 fxc_scaling: FXCScaling | None = None):
        """Construct the Hxc kernel."""
        self.Vbare_G = Vbare_G
        self.fxc_kernel = fxc_kernel
        self.fxc_scaling = fxc_scaling

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

        # Apply kernel scaling, if such a scaling exists
        fxc_scaling = self.fxc_scaling
        if fxc_scaling is not None and fxc_scaling.has_scaling:
            Khxc_GG *= fxc_scaling.get_scaling()

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
        fxc_scaling = hxc_kernel.fxc_scaling
        if fxc_scaling is not None and not fxc_scaling.has_scaling:
            fxc_scaling.calculate_scaling(chiks, Khxc_GG, self)
            lambd = fxc_scaling.get_scaling()
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
