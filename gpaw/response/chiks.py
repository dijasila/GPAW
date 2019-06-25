import numpy as np

from ase.utils.timing import timer

from gpaw.utilities.blas import gemm
from gpaw.response.kslrf import PlaneWaveKSLRF
from gpaw.response.kspair import PlaneWavePairDensity


class ChiKS(PlaneWaveKSLRF):
    """Class calculating the four-component Kohn-Sham susceptibility tensor."""

    def __init__(self, *args, **kwargs):
        """Initialize the chiKS object in plane wave mode."""
        # Avoid any response ambiguity
        if 'response' in kwargs.keys():
            response = kwargs.pop('response')
            assert response == 'susceptibility'

        PlaneWaveKSLRF.__init__(self, *args, response='susceptibility',
                                **kwargs)

        # Susceptibilities use pair densities as matrix elements
        self.pme = PlaneWavePairDensity(self.kspair)

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

    def add_integrand(self, k_v, n1_t, n2_t, s1_t, s2_t, A_wGG):
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
        q_c = self.pd.kd.bzk_kc[0]
        kskptpairs = self.kspair.get_kpoint_pairs(n1_t, n2_t, k_c, k_c + q_c,
                                                  s1_t, s2_t)

        # Get (f_n'k's' - f_nks) and (eps_n'k's' - eps_nks)
        with self.timer('Get occupation differences and transition energies'):
            df_t = kskptpairs.get_occupation_differences()
            df_t[np.abs(df_t) <= 1e-20] = 0.0
            deps_t = kskptpairs.get_transition_energies()
        
        # Calculate the pair densities
        n_tG = self.pme(kskptpairs, self.pd)  # Should this include some extrapolate_q? XXX

        self._add_integrand(n1_t, n2_t, s1_t, s2_t,
                            df_t, deps_t, n_tG, A_wGG)

    @timer('Add integrand to chiks_wGG')
    def _add_integrand(self, n1_t, n2_t, s1_t, s2_t,
                       df_t, deps_t, n_tG, A_wGG):
        """In-place calculation of the integrand (depends on bandsummation)"""
        x_wt = self.get_temporal_part(n1_t, n2_t, s1_t, s2_t, df_t, deps_t)

        '''
        # old gemm version
        for x_t, A_GG in zip(x_wt, A_wGG):  # Why in a for-loop? XXX
            # Multiply temporal part with n_t(q+G'), divide summation in blocks
            nx_tG = n_tG[:, self.Ga:self.Gb] * x_t[:, np.newaxis]
            # Multiply with n_t*(q+G) and sum over transitions t:
            with self.timer('actual gemm step'):
                gemm(1.0, n_tG.conj(), np.ascontiguousarray(nx_tG.T), 1.0, A_GG)
        '''
        '''
        # Matmul numpy version to compare (about a factor 1.03 slower)
        for x_t, A_GG in zip(x_wt, A_wGG):
            nx_tG = n_tG[:, self.Ga:self.Gb] * x_t[:, np.newaxis]
            A_GG += np.matmul(nx_tG.T, n_tG.conj())
        '''
        '''
        # Don't repeat stuff too much - numpy (a factor 0.93 improvement)
        ncc_tG = n_tG.conj()
        nx_wGt = x_wt[:, np.newaxis, :]\
            * n_tG[:, self.Ga:self.Gb].T[np.newaxis, :, :]
        for x_t, nx_Gt, A_GG in zip(x_wt, nx_wGt, A_wGG):
            A_GG += np.matmul(nx_Gt, ncc_tG)
            # nx_tG = n_tG[:, self.Ga:self.Gb] * x_t[:, np.newaxis]
            # A_GG += np.matmul(nx_tG.T, ncc_tG)
        '''
        '''
        # Don't repeat stuff too much - gemm (a factor 0.91 improvement)
        with self.timer('set up nx and ncc'):
            ncc_tG = n_tG.conj()
            nx_wGt = np.ascontiguousarray(x_wt[:, np.newaxis, :]
                                          * n_tG[:, self.Ga:self.Gb].T[np.newaxis, :, :])
        with self.timer('gemm part'):
            for x_t, nx_Gt, A_GG in zip(x_wt, nx_wGt, A_wGG):
                gemm(1.0, ncc_tG, nx_Gt, 1.0, A_GG)
        '''
        # THIS CAN BE A 'MEMORY-SAFE' OPTION XXX
        # Don't repeat and contiguous on large arrays - gemm
        # Calc times (a factor 0.84 improvement):
        # gemm part: [61, 52, 62, 53, 55, 59, 63, 57, 60]    avg: 58
        # setup nx and ncc: [6, 7, 6, 6, 6, 7, 6, 7, 6]      avg: 6.3
        # total: [67, 59, 68, 59, 61, 66, 69, 64, 66]        avg: 64.3
        with self.timer('Set up ncc and nx'):
            ncc_tG = n_tG.conj()
            nx_wGt = x_wt[:, np.newaxis, :]\
                * np.ascontiguousarray(n_tG[:, self.Ga:self.Gb].T)[np.newaxis,
                                                                   :, :]
        with self.timer('Perform sum over t-transitions of ncc * nx'):
            for nx_Gt, A_GG in zip(nx_wGt, A_wGG):
                gemm(1.0, ncc_tG, nx_Gt, 1.0, A_GG)
        '''
        # Vectorized numpy version (about a factor 1.4 slower)
        nx_wGt = x_wt[:, np.newaxis, :]\
            * n_tG[:, self.Ga:self.Gb].T[np.newaxis, :, :]
        A_wGG += np.matmul(nx_wGt, n_tG.conj())
        '''
        '''
        # THIS VERSION SHOULD BE IMPLEMENTED AS DEFAULT OPTION XXX
        # Vectorized gemm version without contiguous on large arrays
        # Calc times (a factor 0.79 improvement):
        # gemm part: [50, 50, 54, 51, 49, 48, 53, 50, 49]    avg: 50.4
        # setup nx and ncc: [7, 12, 12, 11, 10, 8, 9, 11, 9] avg: 9.9
        # total: [57, 62, 66, 62, 59, 56, 62, 61, 58]        avg: 60.3
        with self.timer('array transpose'):
            # Fake a different array structure
            nw, nG1, nG2 = A_wGG.shape
            tmp_GwG = np.zeros((nG2, nw, nG1), dtype=complex)
            # Fake different input structure
            x_tw = np.ascontiguousarray(x_wt.T)
            n_Gt = np.ascontiguousarray(n_tG.T)
        with self.timer('set up nx and ncc'):
            nx_twG = x_tw[:, :, np.newaxis] * \
                np.ascontiguousarray(n_Gt[self.Ga:self.Gb, :].T)[:, np.newaxis, :]
            ncc_Gt = n_Gt.conj()
        with self.timer('gemm part'):
            gemm(1.0, nx_twG, ncc_Gt, 1.0, tmp_GwG)
        with self.timer('array transpose'):
            A_wGG += tmp_GwG.transpose((1, 2, 0))
        '''
        '''
        # Vectorized gemm version (about a factor 1.08 slower)
        with self.timer('array transpose'):
            nw, nG1, nG2 = A_wGG.shape
            tmp_GwG = np.zeros((nG2, nw, nG1), dtype=complex)
        with self.timer('set up nx and ncc'):
            nx_twG = np.ascontiguousarray(x_wt.T[:, :, np.newaxis]
                                          * n_tG[:, np.newaxis, self.Ga:self.Gb])
            # nx_twG = x_wt.T[:, :, np.newaxis]\
            #     * n_tG[:, np.newaxis, self.Ga:self.Gb]
            # ncc_Gt = n_tG.conj().T
            ncc_Gt = np.ascontiguousarray(n_tG.conj().T)
        with self.timer('gemm part'):
            gemm(1.0, nx_twG, ncc_Gt, 1.0, tmp_GwG)
        with self.timer('array transpose'):
            A_wGG += tmp_GwG.transpose((1, 2, 0))
        '''
        '''
        # Memory kamikaze - gemm version (about a factor 1.8 slower)
        ncc_tG = n_tG.conj()
        with self.timer('construct huge array'):
            n_tGG = ncc_tG[:, np.newaxis, :] * n_tG[:, self.Ga:self.Gb, np.newaxis]
        gemm(1.0, n_tGG, x_wt, 1.0, A_wGG)
        '''
        '''
        # Sum numpy version to compare (about a factor 22 slower...)
        for x_t, A_GG in zip(x_wt, A_wGG):
            nx_tG = n_tG[:, self.Ga:self.Gb] * x_t[:, np.newaxis]
            A_GG += np.sum(n_tG.conj()[:, np.newaxis, :]
                           * nx_tG[:, :, np.newaxis], axis=0)
        '''
        '''
        # Memory kamikaze - numpy version (about a factor 40 slower)
        n_tGG = n_tG[:, np.newaxis, :].conj()\
            * n_tG[:, self.Ga:self.Gb, np.newaxis]
        A_wGG += np.sum(x_wt[:, :, np.newaxis, np.newaxis]
                        * n_tGG[np.newaxis, :, :, :], axis=1)
        '''
        '''
        # Go for A_wGG sized arrays - numpy version (about  a factor 33 slower)
        for x_w, n_G in zip(x_wt.T, n_tG):
            A_wGG += x_w[:, np.newaxis, np.newaxis]\
                * n_G.conj()[np.newaxis, np.newaxis, :]\
                * n_G[np.newaxis, self.Ga:self.Gb, np.newaxis]
        '''

    @timer('Get temporal part')
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
        # Get the right spin components
        scomps_t = get_smat_components(self.spincomponent, s1_t, s2_t)
        # Calculate nominator
        nom_t = scomps_t * df_t
        # Calculate denominator
        denom_wt = self.omega_w[:, np.newaxis] - deps_t[np.newaxis, :]\
            + 1j * self.eta
        
        return nom_t[np.newaxis, :] / denom_wt

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
        # Dirac delta
        delta_t = np.ones(len(n1_t))
        delta_t[n2_t <= n1_t] = 0
        # Get the right spin components
        scomps1_t = get_smat_components(self.spincomponent, s1_t, s2_t)
        scomps2_t = get_smat_components(self.spincomponent, s2_t, s1_t)
        # Calculate nominators
        nom1_t = scomps1_t * df_t
        nom2_t = delta_t * scomps2_t * df_t
        # Calculate denominators
        denom1_wt = self.omega_w[:, np.newaxis] - deps_t[np.newaxis, :]\
            + 1j * self.eta
        denom2_wt = self.omega_w[:, np.newaxis] + deps_t[np.newaxis, :]\
            + 1j * self.eta
        
        return nom1_t[np.newaxis, :] / denom1_wt\
            - nom2_t[np.newaxis, :] / denom2_wt


def get_spin_rotation(spincomponent):
    """Get the spin rotation corresponding to the given spin component."""
    if spincomponent is None or spincomponent == '00':
        return '0'
    elif spincomponent in ['uu', 'dd', '+-', '-+']:
        return spincomponent[-1]
    else:
        raise ValueError(spincomponent)


def get_smat_components(spincomponent, s1_t, s2_t):
    """For s1=s and s2=s', get:
    smu_ss' snu_s's
    """
    if spincomponent is None:
        spincomponent = '00'

    smatmu = smat(spincomponent[0])
    smatnu = smat(spincomponent[1])

    return smatmu[s1_t, s2_t] * smatnu[s2_t, s1_t]


def smat(spinrot):
    if spinrot == '0':
        return np.array([[1, 0], [0, 1]])
    elif spinrot == 'u':
        return np.array([[1, 0], [0, 0]])
    elif spinrot == 'd':
        return np.array([[0, 0], [0, 1]])
    elif spinrot == '-':
        return np.array([[0, 0], [1, 0]])
    elif spinrot == '+':
        return np.array([[0, 1], [0, 0]])
    else:
        raise ValueError(spinrot)
