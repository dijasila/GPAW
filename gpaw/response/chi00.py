"""Currently stuff created along the way to have backwards compability
in new versatile format at some point, is stored here. The things contained
here are not expected to work at the moment."""

from gpaw.response.chiKS import ChiKS


class Chi0(ChiKS):
    """Class to keep backwards compability for dielectric response."""

    def __init__(self, *args, **kwargs):
        """
        """
        
        ChiKS.__init__(self, *args, **kwargs)
        self.calculate = self.old_calculate

    def old_calculate(self, q_c, spin='all', A_x=None):
        """Calculate spin susceptibility in plane wave mode.

        Parameters
        ----------
        q_c : list or ndarray
            Momentum vector.
        spin : str or int
            What susceptibility should be calculated?
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented
            'all' is an alias for '00', kept for backwards compability
            Likewise 0 or 1, can be used for 'uu' or 'dd'
        A_x : ndarray
            Output array. If None, the output array is created.

        Returns
        -------
        pd : Planewave descriptor
            Planewave descriptor for q_c.
        chi_wGG : ndarray
            The response function.
        
        Note
        ----
        When running a '00'='all' calculation and q_c = [0., 0., 0.],
        there may be additional outputs (kept only for backwards compability):
        chi_wxvG : ndarray or None
            Wings of the density response function.
        chi_wvv : ndarray or None
            Head of the density response function.

        Future: Instead of having an optical limit, in which another response
                function is calculated, the two calculations should be
                separated.
        """
        return

    def setup_chi_wings(self, nG):  # For chi00
        optical_limit = np.allclose(q_c, 0.0)
        if optical_limit:
            chi_wxvG = np.zeros((len(self.omega_w), 2, 3, nG), complex)
            chi_wvv = np.zeros((len(self.omega_w), 3, 3), complex)
            self.plasmafreq_vv = np.zeros((3, 3), complex)
        else:
            chi_wxvG = None
            chi_wvv = None
            self.plasmafreq_vv = None

    def get_pw_frequencies(self, frequencies, **kwargs):
        """Frequencies to evaluate response function at in GPAW units"""
        if frequencies is None:
            return get_nonlinear_frequency_grid(self.calc, self.nbands,
                                                fd=self.fd, **kwargs)
        else:
            return ArrayDescriptor(np.asarray(frequencies) / Hartree)

    
    def pw_setup_susceptibility(self, pd, spin):
        """Set up chiKS to calculate a spicific spin susceptibility."""
        assert spin in ['00', 'uu', 'dd', '+-', '-+']
        self._pw_setup_sus = self.create_pw_setup_susceptibility(spin)
        self._pw_setup_sus(pd, **self.extraargs)

    def create_pw_setup_susceptibility(self, spin):
        """Creator component deciding how to set up spin susceptibility."""
        if spin == '00':
            return self.pw_setup_chi00
        else:
            return self.pw_setup_chimunu
    
    def pw_setup_chi00(self, pd, hilbert=True, timeordered=False,
                       intraband=True, integrationmode=None,
                       disable_point_group=False,
                       disable_time_reversal=False,
                       disable_non_symmorphic=True,
                       rate=0.0, eshift=0.0, **extraargs):
        """Set up additional parameters for plane wave chi00 calculation
        The chi00 calculation could use further refactorization XXX

        Parameters
        ----------

        old stuff:
            If None, a nonlinear frequency grid is used:
            domega0, omega2, omegamax : float
                Input parameters for nonlinear frequency grid.
                Passed through kwargs


        hilbert : bool
            Switch for hilbert transform. If True, the full density
            response is determined from a hilbert transform of its spectral
            function. This is typically much faster, but does not work for
            imaginary frequencies.
        timeordered : bool
            Calculate the time ordered density response function.
            In this case the hilbert transform cannot be used.
        intraband : bool
            Switch for including the intraband contribution to the density
            response function.
        disable_point_group : bool
            Do not use the point group symmetry operators.
        disable_time_reversal : bool
            Do not use time reversal symmetry.
        disable_non_symmorphic : bool
            Do no use non symmorphic symmetry operators.
        integrationmode : str
            Integrator for the kpoint integration.
        rate : float
            Unknown parameter for intraband calculation in optical limit
        eshift : float
            Shift unoccupied bands
        """

        self.hilbert = hilbert
        self.timeordered = bool(timeordered)
            
        if self.eta == 0.0:  # Refacor using factory method XXX
            # If eta is 0 then we must be working with imaginary frequencies.
            # In this case chi is hermitian and it is therefore possible to
            # reduce the computational costs by a only computing half of the
            # response function.
            assert not self.hilbert
            assert not self.timeordered
            assert not self.omega_w.real.any()
            
            self.integration_kind = 'hermitian response function'
        elif self.hilbert:
            # The spectral function integrator assumes that the form of the
            # integrand is a function (a matrix element) multiplied by
            # a delta function and should return a function of at user defined
            # x's (frequencies). Thus the integrand is tuple of two functions
            # and takes an additional argument (x).
            assert self.frequencies is None
            self.integration_kind = 'spectral function'
        else:
            self.extraintargs['eta'] = self.eta
            self.extraintargs['timeordered'] = self.timeordered
            self.integration_kind = 'response function'

        self.include_intraband = intraband
        
        PWSA = self.setup_kpoint_integration(integrationmode, pd,
                                             disable_point_group,
                                             disable_time_reversal,
                                             disable_non_symmorphic,
                                             **extraargs)
        
        if rate == 'eta':
            self.rate = self.eta
        else:
            self.rate = rate / Hartree

        self.eshift = eshift / Hartree

        return PWSA

    def pw_setup_chimunu(self, pd, **unused):
        """Disable stuff, that has been developed for chi00 only"""

        self.extraintargs['eta'] = self.eta
        self.extraintargs['timeordered'] = False
        self.integration_kind = 'response function'

        PWSA = self.setup_kpoint_integration('point integration', pd,
                                             True, True, True,  # disable sym
                                             **unused)
        self.eshift = 0.0

        return PWSA
