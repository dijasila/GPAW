import numpy as np

from gpaw.response.kslrf import PlaneWaveKSLRF
from gpaw.utilities.blas import gemm


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

                      __ /
                      \  | smu_ss' snu_s's (f_n'k's' - f_nks)
        chiKSmunu =   /  | ----------------------------------
                      ‾‾ | hw - (eps_n'k's'-eps_nks) + ih eta
                      T  \
                                                          \
                       smu_s's snu_ss' (f_n'k's' - f_nks) |
           -delta_n'>n ---------------------------------- | n_T*(q+G) n_T(q+G')
                       hw + (eps_n'k's'-eps_nks) + ih eta |
                                                          /
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

        self._add_integrand(n1_t, n2_t, s1_t, s2_t,
                            df_t, deps_t, n_tG, A_wGG)

    def _add_integrand(self, n1_t, n2_t, s1_t, s2_t,
                       df_t, deps_t, n_tG, A_wGG):
        """In-place calculation of the integrand (depends on bandsummation)"""
        x_wt = self.get_temporal_part(n1_t, n2_t, s1_t, s2_t, df_t, deps_t)

        for x_t, A_GG in zip(x_wt, A_wGG):  # Why in a for-loop? XXX
            # Multiply temporal part with n_t(q+G'), divide summation in blocks
            nx_tG = n_tG[:, self.Ga:self.Gb] * x_t[:, np.newaxis]
            # Multiply with n_t*(q+G) and sum over transitions t:
            gemm(1.0, n_tG.conj(), np.ascontiguousarray(nx_tG.T), 1.0, A_wGG)

    def get_temporal_part(self, n1_t, n2_t, s1_t, s2_t, df_t, deps_t):
        """Get the temporal part of the susceptibility integrand."""
        _get_temporal_part = self.create_get_temporal_part()
        return _get_temporal_part(n1_t, n2_t, s1_t, s2_t, df_t, deps_t)

    def create_get_temporal_part(self):
        """Creator component, deciding how to calculate the temporal part"""
        if self.bandsummation == 'double':
            return self.get_double_temporal_part
        elif self.bandsummation == 'pairwise':
            return self.get_pairwise_temporal_part
        raise ValueError(self.bandsummation)

    def get_double_temporal_part(self, n1_t, n2_t, s1_t, s2_t, df_t, deps_t):
        """Get:
        
               smu_ss' snu_s's (f_n'k's' - f_nks)
        x_wt = ----------------------------------
               hw - (eps_n'k's'-eps_nks) + ih eta
        """
        # Some spin things
        return

    def get_pairwise_temporal_part(self, n1_t, n2_t, s1_t, s2_t, df_t, deps_t):
        """Get:
               /
               | smu_ss' snu_s's (f_n'k's' - f_nks)
        x_wt = | ----------------------------------
               | hw - (eps_n'k's'-eps_nks) + ih eta
               \
                                                           \
                        smu_s's snu_ss' (f_n'k's' - f_nks) |
            -delta_n'>n ---------------------------------- |
                        hw + (eps_n'k's'-eps_nks) + ih eta |
                                                           /
        """
        # Some spin things
        return


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
