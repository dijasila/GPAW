from functools import partial

import numpy as np

from ase.units import Hartree

from gpaw.response import ResponseContext
from gpaw.response.pair_integrator import PairFunctionIntegrator
from gpaw.response.chiks import get_spin_rotation, get_temporal_part
from gpaw.response.frequencies import FrequencyDescriptor


class JDOSCalculator(PairFunctionIntegrator):
    r"""Joint density of states calculator.

    Here, the joint density of states of collinear systems is defined as the
    spectral part of the four-component Kohn-Sham susceptibility,
    see [PRB 103, 245110 (2021)]:

                   __  __
                1  \   \   /
    g^μν(q,ω) = ‾  /   /   | σ^μ_ss' σ^ν_s's (f_nks - f_n'k+qs')
                V  ‾‾  ‾‾  \
                   k   t                                \
                             x δ(ħω - [ε_n'k's'-ε_nks]) |
                                                        /

    where t is a composite band and spin transition index: (n, s) -> (n', s').
    """

    def __init__(self, gs, context=None, **kwargs):
        """Contruct the JDOSCalculator

        Parameters
        ----------
        gs : ResponseGroundStateAdapter
        context : ResponseContext
        kwargs : see gpaw.response.pair_integrator.PairFunctionIntegrator
        """
        if context is None:
            context = ResponseContext()
        assert isinstance(context, ResponseContext)

        super().__init__(gs, context, **kwargs)

    def calculate(self, spincomponent, q_c, wd,
                  eta=0.2,
                  nbands=None,
                  bandsummation='pairwise'):
        """Calculate g^μν(q,ω) using a lorentzian broadening of the δ-function

        Parameters
        ----------
        spincomponent : str
            Spin component (μν) of the joint density of states.
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented.
        q_c : list or np.array
            Wave vector in relative coordinates
        wd : FrequencyDescriptor
            Frequencies to evaluate g^μν(q,ω) at
        eta : float
            HWHM broadening of the δ-function
        nbands : int
            Number of bands to include in the sum over states
        bandsummation : str
            Band summation strategy (does not change the result, but can affect
            the run-time).
            'pairwise': sum over pairs of bands
            'double': double sum over band indices.
        """
        assert isinstance(wd, FrequencyDescriptor)

        # Set inputs on self, so that they can be accessed later
        self.spincomponent = spincomponent
        self.wd = wd
        self.eta = eta / Hartree  # eV -> Hartree
        self.bandsummation = bandsummation

        # Analyze the requested spin component
        spinrot = get_spin_rotation(spincomponent)

        # Prepare to sum over bands and spins
        n1_t, n2_t, s1_t, s2_t = self.get_band_and_spin_transitions_domain(
            spinrot, nbands=nbands, bandsummation=bandsummation)
        self.print_information(q_c, len(wd), eta,
                               spincomponent, nbands, len(n1_t))

        # Allocate array
        jdos_w = np.zeros(len(wd), dtype=float)

        # Perform actual in-place integration
        self.context.print('Integrating the joint density of states:')
        pd = self._get_pw_descriptor(q_c, ecut=1e-3)  # No plane-wave repr.
        self._integrate(pd, jdos_w, n1_t, n2_t, s1_t, s2_t)

        return jdos_w

    def add_integrand(self, kptpair, weight, _, jdos_w):
        r"""Add the g^μν(q,ω) integrand of the outer k-point integral:
                        __
                  -1    \  σ^μ_ss' σ^ν_s's (f_nks - f_n'k's')
        (...)_k = ‾‾ Im /  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                  π     ‾‾   ħω - (ε_n'k's' - ε_nks) + iħη
                        t

        NB: Since the implemented spin matrices are real, the dissipative part
        is equal to the imaginary part (up to a factor of π) of the full
        integrand.
        """
        # Get bands and spins of the transitions
        n1_t, n2_t, s1_t, s2_t = kptpair.get_transitions()
        # Get (f_n'k's' - f_nks) and (ε_n'k's' - ε_nks)
        df_t, deps_t = kptpair.df_t, kptpair.deps_t

        # Construct jdos integrand via the imaginary part of the frequency
        # dependence in χ_KS^μν(q,ω)
        x_wt = get_temporal_part(self.spincomponent, self.wd.omega_w, self.eta,
                                 n1_t, n2_t, s1_t, s2_t, df_t, deps_t,
                                 self.bandsummation)
        integrand_wt = -x_wt.imag / np.pi

        with self.context.timer('Perform sum over t-transitions'):
            jdos_w += weight * np.sum(integrand_wt, axis=1)

    def print_information(self, q_c, nw, eta, spincomponent, nbands, nt):
        """Print information about the joint density of states calculation"""
        p = partial(self.context.print, flush=False)

        p('Calculating the joint density of states with:')
        p('    q_c: [%f, %f, %f]' % (q_c[0], q_c[1], q_c[2]))
        p('    Number of frequency points: %d' % nw)
        p('    Broadening (eta): %f' % eta)
        p('    Spin component: %s' % spincomponent)
        if nbands is None:
            p('    Bands included: All')
        else:
            p('    Number of bands included: %d' % nbands)
        p('Resulting in:')
        p('    A total number of band and spin transitions of: %d' % nt)
        p('')

        self.print_basic_information()
        
