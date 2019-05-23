from gpaw.response.kslrf import PlaneWaveKSLRF


class chiKS(PlaneWaveKSLRF):
    """Class calculating the four-component Kohn-Sham susceptibility tensor."""

    def __init__(self, *args, **kwargs):
        """Initialize the chiKS object in plane wave mode.
        
        INSERT: Description of the matrix elements XXX
        """
        PlaneWaveKSLRF.__init__(self, *args, **kwargs)

        self.PME = PairDensity(*someargs, **somekwargs)  # Write this

        # The class is calculating one spin component at a time
        self.spincomponent = None

    def calculate(self, q_c, spincomponent='all', A_x=None):
        """Calculate a component of the susceptibility tensor.

        Parameters
        ----------
        spincomponent : str or int
            What susceptibility should be calculated?
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented.
            'all' is an alias for '00', kept for backwards compability
            Likewise 0 or 1, can be used for 'uu' or 'dd'
        """
        # Analyze the requested spin component
        self.spincomponent = spincomponent
        spinrot = get_spin_rotation(spincomponent)

        return PlaneWaveKSLRF.calculate(self, q_c, spinrot=spinrot, A_x=A_x)

    def get_integrand(self, *args, **kwargs):  # Write me XXX
        """Use PairDensity object to calculate the integrand"""  # Make good description XXX
        return self.get_pair_density(*args, **kwargs)
        
    @timer('Get pair density')  # old stuff XXX
    def get_pair_density(self, k_v, s, n_M, m_M, block=True):
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


def get_spin_rotation(spincomponent):
    """Get the spin rotation corresponding to the given spin component."""
    if spincomponent == '00':
        return 'I'
    elif spincomponent in ['uu', 'dd']:
        return spincomponent
    elif spincomponent in ['+-', '-+']:
        return spincomponent[-1]
    else:
        raise ValueError(spincomponent)
