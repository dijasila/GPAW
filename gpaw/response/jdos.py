from functools import partial

import numpy as np

from ase.units import Hartree

from gpaw.response import ResponseContext
from gpaw.response.kslrf import PairFunctionIntegrator
from gpaw.response.chiks import get_spin_rotation, get_double_temporal_part
from gpaw.response.frequencies import FrequencyDescriptor


class JDOSCalculator(PairFunctionIntegrator):
    """
    Some documentation here!                                                   XXX
    """

    def __init__(self, gs, context=None, **kwargs):
        """
        Some documentation here!                                               XXX
        """
        if context is None:
            context = ResponseContext()
        assert isinstance(context, ResponseContext)

        super().__init__(gs, context, **kwargs)

    def calculate(self, q_c, wd, eta=0.2, spincomponent=None, nbands=None):
        """
        Some documentation here!                                               XXX
        To do:
          - bandsummation

        Parameters
        ----------
        spincomponent : str or int
            What susceptibility should be calculated?
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented.
            'all' is an alias for '00', kept for backwards compability
            Likewise 0 or 1, can be used for 'uu' or 'dd'
        """
        assert isinstance(wd, FrequencyDescriptor)

        # Set inputs on self, so that they can be accessed later
        self.spincomponent = spincomponent
        self.wd = wd
        self.eta = eta / Hartree  # eV -> Hartree

        # Analyze the requested spin component
        spinrot = get_spin_rotation(spincomponent)

        # Prepare to sum over bands and spins
        n1_t, n2_t, s1_t, s2_t = self.get_band_and_spin_transitions_domain(
            spinrot, nbands=nbands, bandsummation='double')
        self.print_information(q_c, len(wd), eta,
                               spincomponent, nbands, len(n1_t))

        # Allocate array
        jdos_w = np.zeros(len(wd), dtype=float)

        # Perform actual in-place integration
        self.context.print('Integrating the joint density of states:')
        self._integrate(q_c, jdos_w, n1_t, n2_t, s1_t, s2_t,
                        ecut=1e-3)

        return jdos_w

    def add_integrand(self, kptpair, weight, jdos_w):
        r"""
        Some documentation here!                                              XXX

        bandsummation: double

                        __
                  -1    \  σ^μ_ss' σ^ν_s's (f_nks - f_n'k's')
        (...)_k = ‾‾ Im /  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                  π     ‾‾   ħω - (ε_n'k's' - ε_nks) + iħη
                        t
        """
        # Get bands and spins of the transitions
        _, _, s1_t, s2_t = kptpair.get_transitions()
        # Get (f_n'k's' - f_nks) and (ε_n'k's' - ε_nks)
        df_t, deps_t = kptpair.df_t, kptpair.deps_t

        # Construct jdos integrand via the dissipative part of the frequency
        # dependence of a causal linear response function
        # NB: Since the implemented spin matrices are real, the dissipative
        # part is equal to the imaginary part
        x_wt = get_double_temporal_part(self.spincomponent, s1_t, s2_t,
                                        df_t, deps_t,
                                        self.wd.omega_w, self.eta)
        integrand_wt = - x_wt.imag / np.pi

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
        
