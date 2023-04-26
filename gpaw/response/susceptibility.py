from __future__ import annotations

import numpy as np

from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.pair_functions import (SingleQPWDescriptor, Chi,
                                          get_pw_coordinates)
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.coulomb_kernels import get_coulomb_kernel
from gpaw.response.fxc_kernels import FXCKernel, AdiabaticFXCCalculator
from gpaw.response.dyson import DysonSolver, HXCKernel


class ChiFactory:
    r"""User interface to calculate individual elements of the four-component
    susceptibility tensor χ^μν, see [PRB 103, 245110 (2021)]."""

    def __init__(self,
                 chiks_calc: ChiKSCalculator,
                 fxc_calculator: AdiabaticFXCCalculator | None = None):
        """Contruct a many-body susceptibility factory."""
        self.chiks_calc = chiks_calc
        self.gs = chiks_calc.gs
        self.context = chiks_calc.context
        self.dyson_solver = DysonSolver(self.context)

        # If no fxc_calculator is supplied, fall back to default
        if fxc_calculator is None:
            fxc_calculator = AdiabaticFXCCalculator.from_rshe_parameters(
                self.gs, self.context)
        else:
            assert fxc_calculator.gs is chiks_calc.gs
            assert fxc_calculator.context is chiks_calc.context
        self.fxc_calculator = fxc_calculator

        # Prepare a buffer for the fxc kernels
        self.fxc_kernel_cache: dict[str, FXCKernel] = {}

    def __call__(self, spincomponent, q_c, complex_frequencies,
                 fxc=None, hxc_scaling=None, txt=None) -> tuple[Chi, Chi]:
        r"""Calculate a given element (spincomponent) of the four-component
        Kohn-Sham susceptibility tensor and construct a corresponding many-body
        susceptibility object within a given approximation to the
        exchange-correlation kernel.

        Parameters
        ----------
        spincomponent : str
            Spin component (μν) of the susceptibility.
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented.
        q_c : list or ndarray
            Wave vector
        complex_frequencies : np.array or ComplexFrequencyDescriptor
            Array of complex frequencies to evaluate the response function at
            or a descriptor of those frequencies.
        fxc : str (None defaults to ALDA)
            Approximation to the (local) xc kernel.
            Choices: RPA, ALDA, ALDA_X, ALDA_x
        hxc_scaling : None or HXCScaling
            Supply an HXCScaling object to scale the hxc kernel.
        txt : str
            Save output of the calculation of this specific component into
            a file with the filename of the given input.
        """
        # Fall back to ALDA per default
        if fxc is None:
            fxc = 'ALDA'

        # Initiate new output file, if supplied
        if txt is not None:
            self.context.new_txt_and_timer(txt)

        # Print to output file
        self.context.print('---------------', flush=False)
        self.context.print('Calculating susceptibility spincomponent='
                           f'{spincomponent} with q_c={q_c}', flush=False)
        self.context.print('---------------')

        # Calculate chiks
        chiks = self.calculate_chiks(spincomponent, q_c, complex_frequencies)

        # Construct the hxc kernel
        hartree_kernel = self.get_hartree_kernel(spincomponent, chiks.qpd)
        xc_kernel = self.get_xc_kernel(fxc, spincomponent, chiks.qpd)
        hxc_kernel = HXCKernel(hartree_kernel, xc_kernel, scaling=hxc_scaling)

        # Solve dyson equation
        chi = self.dyson_solver(chiks, hxc_kernel)

        return chiks, chi

    def get_hartree_kernel(self, spincomponent, qpd):
        if spincomponent in ['+-', '-+']:
            # No Hartree term in Dyson equation
            return None
        else:
            return get_coulomb_kernel(qpd, self.gs.kd.N_c)

    def get_xc_kernel(self,
                      fxc: str,
                      spincomponent: str,
                      qpd: SingleQPWDescriptor):
        """Get the requested xc-kernel object."""
        if fxc == 'RPA':
            # No xc-kernel
            return None

        if qpd.gammacentered:
            # When using a gamma-centered plane-wave basis, we can reuse the
            # fxc kernel for all q-vectors. Thus, we keep a cache of calculated
            # kernels
            key = f'{fxc},{spincomponent}'
            if key not in self.fxc_kernel_cache:
                self.fxc_kernel_cache[key] = self.fxc_calculator(
                    fxc, spincomponent, qpd)
            fxc_kernel = self.fxc_kernel_cache[key]
        else:
            # Always compute the kernel
            fxc_kernel = self.fxc_calculator(fxc, spincomponent, qpd)

        return fxc_kernel

    def calculate_chiks(self, spincomponent, q_c, complex_frequencies):
        """Calculate the Kohn-Sham susceptibility."""
        q_c = np.asarray(q_c)
        if isinstance(complex_frequencies, ComplexFrequencyDescriptor):
            zd = complex_frequencies
        else:
            zd = ComplexFrequencyDescriptor.from_array(complex_frequencies)

        # Perform actual calculation
        chiks = self.chiks_calc.calculate(spincomponent, q_c, zd)
        # Distribute frequencies over world
        chiks = chiks.copy_with_global_frequency_distribution()

        return chiks


def get_inverted_pw_mapping(qpd1, qpd2):
    """Get the planewave coefficients mapping GG' of qpd1 into -G-G' of qpd2"""
    G1_Gc = get_pw_coordinates(qpd1)
    G2_Gc = get_pw_coordinates(qpd2)

    mG2_G1 = []
    for G1_c in G1_Gc:
        found_match = False
        for G2, G2_c in enumerate(G2_Gc):
            if np.all(G2_c == -G1_c):
                mG2_G1.append(G2)
                found_match = True
                break
        if not found_match:
            raise ValueError('Could not match qpd1 and qpd2')

    # Set up mapping from GG' to -G-G'
    invmap_GG = tuple(np.meshgrid(mG2_G1, mG2_G1, indexing='ij'))

    return invmap_GG


def symmetrize_reciprocity(qpd, X_wGG):
    """In collinear systems without spin-orbit coupling, the plane wave
    susceptibility is reciprocal in the sense that e.g.

    χ_(GG')^(+-)(q, ω) = χ_(-G'-G)^(+-)(-q, ω)

    This method symmetrizes A_wGG in the case where q=0.
    """
    from gpaw.test.response.test_chiks import get_inverted_pw_mapping

    q_c = qpd.q_c
    if np.allclose(q_c, 0.):
        invmap_GG = get_inverted_pw_mapping(qpd, qpd)
        for X_GG in X_wGG:
            # Symmetrize [χ_(GG')(q, ω) + χ_(-G'-G)(-q, ω)] / 2
            X_GG[:] = (X_GG + X_GG[invmap_GG].T) / 2.
