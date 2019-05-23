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

    @timer('dist freq')
    def distribute_frequencies(self, chi0_wGG):
        """Distribute frequencies to all cores."""

        world = self.world
        comm = self.blockcomm

        if world.size == 1:
            return chi0_wGG

        nw = len(self.omega_w)
        nG = chi0_wGG.shape[2]
        mynw = (nw + world.size - 1) // world.size
        mynG = (nG + comm.size - 1) // comm.size

        wa = min(world.rank * mynw, nw)
        wb = min(wa + mynw, nw)

        if self.blockcomm.size == 1:
            return chi0_wGG[wa:wb].copy()

        if self.kncomm.rank == 0:
            bg1 = BlacsGrid(comm, 1, comm.size)
            in_wGG = chi0_wGG.reshape((nw, -1))
        else:
            bg1 = DryRunBlacsGrid(mpi.serial_comm, 1, 1)
            in_wGG = np.zeros((0, 0), complex)
        md1 = BlacsDescriptor(bg1, nw, nG**2, nw, mynG * nG)

        bg2 = BlacsGrid(world, world.size, 1)
        md2 = BlacsDescriptor(bg2, nw, nG**2, mynw, nG**2)

        r = Redistributor(world, md1, md2)
        shape = (wb - wa, nG, nG)
        out_wGG = np.empty(shape, complex)
        r.redistribute(in_wGG, out_wGG.reshape((wb - wa, nG**2)))

        return out_wGG


class ArrayDescriptor:
    """Describes a single dimensional array."""

    def __init__(self, data_x):
        self.data_x = np.array(np.sort(data_x))
        self._data_len = len(data_x)

    def __len__(self):
        return self._data_len

    def get_data(self):
        return self.data_x

    def get_closest_index(self, scalars_w):
        """Get closest index.

        Get closest index approximating scalars from below."""
        diff_xw = self.data_x[:, np.newaxis] - scalars_w[np.newaxis]
        return np.argmin(diff_xw, axis=0)

    def get_index_range(self, lim1_m, lim2_m):
        """Get index range. """

        i0_m = np.zeros(len(lim1_m), int)
        i1_m = np.zeros(len(lim2_m), int)

        for m, (lim1, lim2) in enumerate(zip(lim1_m, lim2_m)):
            i_x = np.logical_and(lim1 <= self.data_x,
                                 lim2 >= self.data_x)
            if i_x.any():
                inds = np.argwhere(i_x)
                i0_m[m] = inds.min()
                i1_m[m] = inds.max() + 1

        return i0_m, i1_m


import numbers


class FrequencyDescriptor(ArrayDescriptor):

    def __init__(self, domega0, omega2, omegamax):
        beta = (2**0.5 - 1) * domega0 / omega2
        wmax = int(omegamax / (domega0 + beta * omegamax))
        w = np.arange(wmax + 2)  # + 2 is for buffer
        omega_w = w * domega0 / (1 - beta * w)

        ArrayDescriptor.__init__(self, omega_w)

        self.domega0 = domega0
        self.omega2 = omega2
        self.omegamax = omegamax
        self.omegamin = 0

        self.beta = beta
        self.wmax = wmax
        self.omega_w = omega_w
        self.wmax = wmax
        self.nw = len(omega_w)

    def get_closest_index(self, o_m):
        beta = self.beta
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        if isinstance(w_m, np.ndarray):
            w_m[w_m >= self.wmax] = self.wmax - 1
        elif isinstance(w_m, numbers.Integral):
            if w_m >= self.wmax:
                w_m = self.wmax - 1
        else:
            raise TypeError
        return w_m

    def get_index_range(self, omega1_m, omega2_m):
        omega1_m = omega1_m.copy()
        omega2_m = omega2_m.copy()
        omega1_m[omega1_m < 0] = 0
        omega2_m[omega2_m < 0] = 0
        w1_m = self.get_closest_index(omega1_m)
        w2_m = self.get_closest_index(omega2_m)
        o1_m = self.omega_w[w1_m]
        o2_m = self.omega_w[w2_m]
        w1_m[o1_m < omega1_m] += 1
        w2_m[o2_m < omega2_m] += 1
        return w1_m, w2_m


def find_maximum_frequency(calc, nbands, fd=None):
    """Determine the maximum electron-hole pair transition energy."""
    epsmin = 10000.0
    epsmax = -10000.0
    for kpt in calc.wfs.kpt_u:
        epsmin = min(epsmin, kpt.eps_n[0])
        epsmax = max(epsmax, kpt.eps_n[nbands - 1])

    if fd is not None:
        print('Minimum eigenvalue: %10.3f eV' % (epsmin * Hartree), file=fd)
        print('Maximum eigenvalue: %10.3f eV' % (epsmax * Hartree), file=fd)

        return epsmax - epsmin


def get_nonlinear_frequency_grid(calc, nbands, fd=None,
                                 domega0=0.1, omega2=10.0, omegamax=None,
                                 **unused):
    domega0 = domega0 / Hartree
    omega2 = omega2 / Hartree
    omegamax = None if omegamax is None else omegamax / Hartree
    if omegamax is None:
        omegamax = find_maximum_frequency(calc, nbands)

    if fd is not None:
        print('Using nonlinear frequency grid from 0 to %.3f eV' %
              (omegamax * Hartree), file=fd)
    return FrequencyDescriptor(domega0, omega2, omegamax)



def get_kpoint_tetrahint_domain(pd, PWSA, pbc):
    # Could use documentation XXX
    bzk_kc = PWSA.get_reduced_kd(pbc_c=pbc).bzk_kc
    if (~pbc).any():
        bzk_kc = np.append(bzk_kc,
                           bzk_kc + (~pbc).astype(int),
                           axis=0)

    return bzk_kc


def calculate_kpoint_tetrahint_prefactor(calc, PWSA, bzk_kv):
    """If there are non-periodic directions it is possible that the
    integration domain is not compatible with the symmetry operations
    which essentially means that too large domains will be
    integrated. We normalize by vol(BZ) / vol(domain) to make
    sure that to fix this.
    """
    vol = abs(np.linalg.det(calc.wfs.gd.cell_cv))
    domainvol = convex_hull_volume(bzk_kv) * PWSA.how_many_symmetries()
    bzvol = (2 * np.pi)**3 / vol
    frac = bzvol / domainvol
    return (2 * frac * PWSA.how_many_symmetries() /
            (calc.wfs.nspins * (2 * np.pi)**3))
