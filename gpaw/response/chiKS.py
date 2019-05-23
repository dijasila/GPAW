from gpaw.response.kslrf import PlaneWaveKSLRF


class chiKS(PlaneWaveKSLRF):
    """Class calculating the four-component Kohn-Sham susceptibility tensor."""

    def __init__(self, *args, **kwargs):
        """Initialize the chiKS object in plane wave mode.
        
        INSERT: Description of the matrix elements XXX
        """
        PlaneWaveKSLRF.__init__(self, *args, **kwargs)

        # INSERT: some attachment of PairDensity object XXX

        # The class is calculating one spin component at a time
        self.spin = None
        # PAW correction might change with spin component and momentum transfer
        self.Q_aGii = None

    def calculate(self, q_c, spin='all', A_x=None):
        """Calculate spin susceptibility in plane wave mode.

        Parameters
        ----------
        q_c : list or ndarray
            Momentum transfer.
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
        """
        # Set up plane wave description with the gived momentum transfer q
        q_c = np.asarray(q_c, dtype=float)
        self.pd = self.get_PWDescriptor(q_c)
        self.PWSA = self.get_PWSymmetryAnalyzer(self.pd)
        self.Q_aGii = self.pairdensity.initialize_paw_corrections(self.pd)  # PairDensity object missing XXX

        # Analyze the requested spin component
        self.spin = get_unique_spin_component(spin)

        # Print information about the set up calculation
        self.print_information()  # Print more/less information XXX
        if extra_parameters.get('df_dry_run'):  # Exit after setting up
            print('    Dry run exit', file=self.fd)
            raise SystemExit

        chi_wGG = self.setup_output_array(A_x)

        n_M, m_M, bzk_kv, spins, flip = self.get_summation_domain()

        return self._pw_calculate_susceptibility(chi_wGG,
                                                 n_M, m_M, bzk_kv, spins, flip)

    @timer('Calculate Kohn-Sham susceptibility')
    def _pw_calculate_susceptibility(self, chi_wGG,
                                     n_M, m_M, bzk_kv, spins, flip):
        """In-place calculation of the KS susceptibility in plane wave mode."""

        prefactor = self._calculate_kpoint_integration_prefactor(bzk_kv)
        chi_wGG /= prefactor

        self.integrator.integrate(kind=self.integration_kind,
                                  banddomain=(n_M, m_M),
                                  ksdomain=(bzk_kv, spins),
                                  integrand=(self._get_matrix_element,
                                             self._get_eigenvalues),
                                  x=self.wd,  # Frequency Descriptor
                                  # Arguments for integrand functions
                                  out_wxx=chi_wGG,  # Output array
                                  **self.extraintargs)

        # The response function is integrated only over the IBZ. The
        # chi calculated above must therefore be extended to include the
        # response from the full BZ. This extension can be performed as a
        # simple post processing of the response function that makes
        # sure that the response function fulfills the symmetries of the little
        # group of q. Due to the specific details of the implementation the chi
        # calculated above is normalized by the number of symmetries (as seen
        # below) and then symmetrized.
        chi_wGG *= prefactor
        tmpchi_wGG = self.redistribute(chi_wGG)  # distribute over frequencies
        self.PWSA.symmetrize_wGG(tmpchi_wGG)
        self.redistribute(tmpchi_wGG, chi_wGG)

        return self.pd, chi_wGG

    def setup_kpoint_integration(self, integrationmode, pd,
                                 disable_point_group,
                                 disable_time_reversal,
                                 disable_non_symmorphic, **extraargs):
        if integrationmode is None:
            self.integrationmode == 'point integration'
        else:
            self.integrationmode = integrationmode

        # The integration domain is reduced to the irreducible zone
        # of the little group of q.
        PWSA = PWSymmetryAnalyzer
        PWSA = PWSA(self.calc.wfs.kd, pd,
                    timer=self.timer, txt=self.fd,
                    disable_point_group=disable_point_group,
                    disable_time_reversal=disable_time_reversal,
                    disable_non_symmorphic=disable_non_symmorphic)

        self._setup_integrator = self.create_setup_integrator()
        self._setup_integrator(**extraargs)

        return PWSA

    def create_setup_integrator(self):
        """Creator component deciding how to set up kpoint integrator.
        The integrator class is a general class for brillouin zone
        integration that can integrate user defined functions over user
        defined domains and sum over bands.
        """
        
        if self.integrationmode == 'point integration':
            print('Using integration method: PointIntegrator',
                  file=self.fd)
            return self.setup_point_integrator

        elif self.integrationmode == 'tetrahedron integration':
            print('Using integration method: TetrahedronIntegrator',
                  file=self.fd)
            return self.setup_tetrahedron_integrator

        raise ValueError(self.integrationmode)

    def setup_point_integrator(self, **unused):
        self._get_kpoint_int_domain = partial(get_kpoint_pointint_domain,
                                              calc=self.calc)
        self.integrator = PointIntegrator(self.calc.wfs.gd.cell_cv,
                                          response=self.response,  # should be removed XXX
                                          comm=self.world,
                                          timer=self.timer,
                                          txt=self.fd,
                                          eshift=self.eshift,
                                          nblocks=self.nblocks)
        self._calculate_kpoint_integration_prefactor =\
            calculate_kpoint_pointint_prefactor

    def setup_tetrahedron_integrator(self, pbc=None, **unused):
        """
        Parameters
        ----------
        pbc : list
            Periodic directions of the system. Defaults to [True, True, True].
        """
        self.setup_pbc(pbc)
        self._get_kpoint_int_domain = partial(get_kpoint_tetrahint_domain,
                                              pbc=self.pbc)
        self.integrator = TetrahedronIntegrator(self.calc.wfs.gd.cell_cv,
                                                response=self.response,  # should be removed XXX
                                                comm=self.world,
                                                timer=self.timer,
                                                eshift=self.eshift,
                                                txt=self.fd,
                                                nblocks=self.nblocks)
        self._calculate_kpoint_integration_prefactor =\
            calculate_kpoint_tetrahint_prefactor

    def setup_pbc(self, pbc):
        if pbc is not None:
            self.pbc = np.array(pbc)
        else:
            self.pbc = np.array([True, True, True])

        if self.pbc is not None and (~self.pbc).any():
            assert np.sum((~self.pbc).astype(int)) == 1, \
                print('Only one non-periodic direction supported atm.')
            print('Nonperiodic BC\'s: ', (~self.pbc),
                  file=self.fd)

    def get_summation_domain(self):
        """Find the relevant (n, k, s) -> (m, k + q, s') domain to sum over"""
        n_M, m_M = get_band_summation_domain(self.bandsummation, self.nbands,
                                             nocc1=self.nocc1,
                                             nocc2=self.nocc2)
        spins, flip = get_spin_summation_domain(self.bandsummation, self.spin,
                                                self.calc.wfs.nspins)

        bzk_kv = self.get_kpoint_integration_domain(self.pd)

        return n_M, m_M, bzk_kv, spins, flip

    @timer('Get kpoint integration domain')
    def get_kpoint_integration_domain(self):
        bzk_kc = self._get_kpoint_int_domain(self.pd, self.PWSA)
        bzk_kv = np.dot(bzk_kc, self.pd.gd.icell_cv) * 2 * np.pi
        return bzk_kv

    def _get_kpoint_int_domain(pd):
        raise NotImplementedError('Please set up integrator before calling')

    @timer('Get pair density')
    def get_pw_pair_density(self, k_v, s, n_M, m_M, block=True):
        """A function that returns pair-densities.

        A pair density is defined as::

         <nks| e^(-i (q + G) r) |mk+qs'> = <mk+qs'| e^(i (q + G) r |nks>,

        where s and s' are spins, n and m are band indices, k is
        the kpoint and q is the momentum transfer.

        Parameters
        ----------
        k_v : ndarray
            Kpoint coordinate in cartesian coordinates.
        s : int
            Spin index.

        Return
        ------
        n_MG : ndarray
            Pair densities.
        """
        pd = self.pd
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)

        q_c = pd.kd.bzk_kc[0]

        extrapolate_q = False  # SHOULD THIS ONLY BE USED IN OPTICAL LIM? XXX
        if self.calc.wfs.kd.refine_info is not None:
            K1 = self.KSPair.find_kpoint(k_c)
            label = self.calc.wfs.kd.refine_info.label_k[K1]
            if label == 'zero':
                return None
            elif (self.calc.wfs.kd.refine_info.almostoptical
                  and label == 'mh'):
                if not hasattr(self, 'pd0'):
                    self.pd0 = self.get_PWDescriptor([0, ] * 3)
                pd = self.pd0
                extrapolate_q = True

        if self.Q_aGii is None:
            self.Q_aGii = self.KSPair.initialize_paw_corrections(pd)

        kptpair = self.KSPair.get_kpoint_pair(pd, s, k_c, block=block)

        n_MG = self.KSPair.get_pair_density(pd, kptpair, n_M, m_M,
                                            Q_aGii=self.Q_aGii, block=block)
        
        if self.integrationmode == 'point integration':
            n_MG *= np.sqrt(self.PWSA.get_kpoint_weight(k_c) /
                            self.PWSA.how_many_symmetries())
        
        df_M = kptpair.get_occupation_differences(n_M, m_M)
        df_M[np.abs(df_M) <= 1e-20] = 0.0

        '''  # Change integrator stuff correspondingly XXX
        if self.bandsummation == 'double':
            df_nm = np.abs(df_nm)
        df_nm[df_nm <= 1e-20] = 0.0
        
        n_nmG *= df_nm[..., np.newaxis]**0.5
        '''

        if extrapolate_q:  # SHOULD THIS ONLY BE USED IN OPTICAL LIM? XXX
            q_v = np.dot(q_c, pd.gd.icell_cv) * (2 * np.pi)
            nq_M = np.dot(n_MG[:, :, :3], q_v)
            n_MG = n_MG[:, :, 2:]
            n_MG[:, :, 0] = nq_M

        return n_MG, df_M


def get_unique_spin_component(spin):
    """Convert all supported input to chiKS standard."""
    if isinstance(spin, str):
        if spin in ['00', 'uu', 'dd', '+-', '-+']:
            return spin
        elif spin == 'all':
            return '00'
    elif isinstance(spin, int):
        if spin == 0:
            return 'uu'
        elif spin == 1:
            return 'dd'
    raise ValueError(spin)
