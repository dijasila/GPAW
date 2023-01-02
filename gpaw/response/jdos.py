import numpy as np

from ase.units import Hartree

from gpaw.response import ResponseContext
from gpaw.response.pair_integrator import PairFunctionIntegrator
from gpaw.response.pair_functions import PairFunction
from gpaw.response.chiks import get_spin_rotation, get_temporal_part
from gpaw.response.frequencies import FrequencyDescriptor


class JDOS(PairFunction):

    def __init__(self, spincomponent, pd, wd, eta):
        self.spincomponent = spincomponent
        self.wd = wd
        self.eta = eta

        super().__init__(pd)
    
    def zeros(self):
        nw = len(self.wd)
        return np.zeros(nw, float)


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

    def __init__(self, gs, context=None,
                 nbands=None, bandsummation='pairwise',
                 **kwargs):
        """Contruct the JDOSCalculator

        Parameters
        ----------
        gs : ResponseGroundStateAdapter
        context : ResponseContext
        nbands : int
            Number of bands to include in the sum over states
        bandsummation : str
            Band summation strategy (does not change the result, but can affect
            the run-time).
            'pairwise': sum over pairs of bands
            'double': double sum over band indices.
        kwargs : see gpaw.response.pair_integrator.PairFunctionIntegrator
        """
        if context is None:
            context = ResponseContext()
        assert isinstance(context, ResponseContext)

        super().__init__(gs, context, **kwargs)

        self.nbands = nbands
        self.bandsummation = bandsummation

    def calculate(self, spincomponent, q_c, wd, eta=0.2):
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
        """
        assert isinstance(wd, FrequencyDescriptor)
        eta = eta / Hartree  # eV -> Hartree

        # Analyze the requested spin component
        spinrot = get_spin_rotation(spincomponent)

        # Prepare to sum over bands and spins
        n1_t, n2_t, s1_t, s2_t = self.get_band_and_spin_transitions_domain(
            spinrot, nbands=self.nbands, bandsummation=self.bandsummation)
        self.context.print(self.get_information(
            q_c, len(wd), eta, spincomponent, self.nbands, len(n1_t)))

        # Set up output data structure
        # We need a dummy plane-wave descriptor (without plane-waves, hence the
        # vanishing ecut) for the PairFunctionIntegrator to be able to analyze
        # the symmetries of the system and reduce the k-point integration
        pd = self.get_pw_descriptor(q_c, ecut=1e-3)
        jdos = JDOS(spincomponent, pd, wd, eta)

        # Perform actual in-place integration
        self.context.print('Integrating the joint density of states:')
        self._integrate(jdos, n1_t, n2_t, s1_t, s2_t)

        return jdos

    def add_integrand(self, kptpair, weight, jdos):
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
        # Specify notation
        jdos_w = jdos.array

        # Get bands and spins of the transitions
        n1_t, n2_t, s1_t, s2_t = kptpair.get_transitions()
        # Get (f_n'k's' - f_nks) and (ε_n'k's' - ε_nks)
        df_t, deps_t = kptpair.df_t, kptpair.deps_t

        # Construct jdos integrand via the imaginary part of the frequency
        # dependence in χ_KS^μν(q,ω)
        x_wt = get_temporal_part(jdos.spincomponent, jdos.wd.omega_w + 1.j * jdos.eta,
                                 n1_t, n2_t, s1_t, s2_t, df_t, deps_t,
                                 self.bandsummation)
        integrand_wt = -x_wt.imag / np.pi

        with self.context.timer('Perform sum over t-transitions'):
            jdos_w += weight * np.sum(integrand_wt, axis=1)

    def get_information(self, q_c, nw, eta, spincomponent, nbands, nt):
        """Get information about the joint density of states calculation"""
        s = '\n'

        s += 'Calculating the joint density of states with:\n'
        s += '    q_c: [%f, %f, %f]\n' % (q_c[0], q_c[1], q_c[2])
        s += '    Number of frequency points: %d\n' % nw
        s += '    Broadening (eta): %f\n' % (eta * Hartree)
        s += '    Spin component: %s\n' % spincomponent
        if nbands is None:
            s += '    Bands included: All\n'
        else:
            s += '    Number of bands included: %d\n' % nbands
        s += 'Resulting in:\n'
        s += '    A total number of band and spin transitions of: %d\n' % nt
        s += '\n'

        s += self.get_basic_information()

        return s
        
