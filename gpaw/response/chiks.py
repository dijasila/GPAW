import numpy as np

from gpaw.response.kslrf import PlaneWaveKSLRF


class chiKS(PlaneWaveKSLRF):
    """Class calculating the four-component Kohn-Sham susceptibility tensor."""

    def __init__(self, *args, **kwargs):
        """Initialize the chiKS object in plane wave mode.
        
        INSERT: Description of the matrix elements XXX
        """
        PlaneWaveKSLRF.__init__(self, *args, **kwargs)

        self.pme = PairDensity(*someargs, **somekwargs)  # Write this (maybe creator method XXX)

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

        # Reset PAW correction in case momentum has changed
        self.pme.Q_aGii = None

        return PlaneWaveKSLRF.calculate(self, q_c, spinrot=spinrot, A_x=A_x)

    def add_integrand(self, k_v, n1_t, n2_t, s1_t, s2_t, A_wGG):  # Write me XXX
        """Use PairDensity object to calculate the integrand for all relevant
        transitions of the given k-point.

        Depending on the bandsummation, the collinear four-component Kohn-Sham
        susceptibility tensor as:

        bandsummation: double

                      __
                      \  smu_ss' snu_s's (f_n'k's' - f_nks)
        chiKSmunu =   /  ---------------------------------- n_T*(q+G) n_T(q+G')
                      ‾‾ hw - (eps_n'k's'-eps_nks) + ih eta
                      T

        bandsummation: pairwise (using spin-conserving time-reversal symmetry)

                      __ __
                      \  | smu_ss' snu_s's (f_n'k's' - f_nks)
        chiKSmunu =   /  | ----------------------------------
                      ‾‾ | hw - (eps_n'k's'-eps_nks) + ih eta
                      T  ‾‾
                                                         __
                       smu_s's snu_ss' (f_n'k's' - f_nks) |
           -delta_n'>n ---------------------------------- | n_T*(q+G) n_T(q+G')
                       hw + (eps_n'k's'-eps_nks) + ih eta |
                                                         ‾‾
        """
        # Get all pairs of Kohn-Sham transitions:
        # (n1_t, k_c, s1_t) -> (n2_t, k_c + q_c, s2_t)
        k_c = np.dot(self.pd.gd.cell_cv, k_v) / (2 * np.pi)
        kspairs = self.kspair.get_pairs(k_c, self.pd, n1_t, n2_t, s1_t, s2_t)

        # Get (f_n'k's' - f_nks) and (eps_n'k's' - eps_nks)
        df_t = kspairs.get_occupation_differences()
        df_t[np.abs(df_t) <= 1e-20] = 0.0
        deps_t = kspairs.get_energy_differences()
        
        # Calculate the pair densities
        n_tG = self.pme(kspairs, self.pd)  # Should this include some extrapolate_q? XXX

        # In-place calculation of the integrand (depends on bandsummation):
        self._add_integrand(s1_t, s2_t, df_t, deps_t, n_tG, A_wGG)

    def _add_integrand(self, s1_t, s2_t, df_t, deps_t, n_tG, A_wGG):
        pass

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
