import numpy as np
from time import ctime

from gpaw.utilities import convert_string_to_fd
from ase.utils.timing import timer

from gpaw.utilities.blas import mmmx
from gpaw.response.kslrf import PlaneWaveKSLRF
from gpaw.response.kspair import PlaneWavePairDensity


class ChiKS(PlaneWaveKSLRF):
    r"""Class calculating the four-component Kohn-Sham susceptibility tensor,
    see [PRB 103, 245110 (2021)]. For collinear systems, the susceptibility
    tensor is defined as:
                        __  __   __
                     1  \   \    \
    chiKSmunu(q,w) = ‾  /   /    /   (f_nks - f_n'k+qs') smu_ss' snu_s's
                     V  ‾‾  ‾‾   ‾‾
                        k  n,s  n',s'
                                       n_nks,n'k+qs'(G+q) n_n'k+qs',nks(-G'-q)
                                     x ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                                         hw - (eps_n'k+qs'-eps_nks) + ih eta

    where the matrix elements

    n_nks,n'k+qs'(G+q) = <nks| e^-i(G+q)r |n'k+qs'>_V0

    are unit cell normalized plane wave pair densities of each transition.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the chiKS object in plane wave mode."""
        PlaneWaveKSLRF.__init__(self, *args, **kwargs)

        # Susceptibilities use pair densities as matrix elements
        self.pme = PlaneWavePairDensity(self.kspair)

        # The class is calculating one spin component at a time
        self.spincomponent = None

    @timer('Calculate the Kohn-Sham susceptibility')
    def calculate(self, q_c, frequencies, spincomponent='all',
                  A_x=None, txt=None):
        """Calculate a component of the susceptibility tensor.

        Parameters
        ----------
        spincomponent : str or int
            What susceptibility should be calculated?
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented.
            'all' is an alias for '00', kept for backwards compability
            Likewise 0 or 1, can be used for 'uu' or 'dd'
        """
        # Initiate new call-output file, if supplied
        if txt is not None:
            self.cfd = convert_string_to_fd(txt, self.world)
        # Print to output file(s)
        if str(self.fd) != str(self.cfd) or txt is not None:
            print('Calculating Kohn-Sham susceptibility with '
                  f'q_c={q_c} and spincomponent={spincomponent}',
                  file=self.fd)
            print(ctime(), file=self.fd)

        # Analyze the requested spin component
        self.spincomponent = spincomponent
        spinrot = get_spin_rotation(spincomponent)

        return PlaneWaveKSLRF.calculate(self, q_c, frequencies,
                                        spinrot=spinrot, A_x=A_x)

    @timer('Add integrand to chiks_wGG')
    def add_integrand(self, kskptpair, weight, A_x):
        r"""Use PairDensity object to calculate the integrand for all relevant
        transitions of the given k-point pair (k,k+q).

        Depending on the bandsummation parameter, the integrand of the
        collinear four-component Kohn-Sham susceptibility tensor is calculated
        as:

        bandsummation: double

                   __
                   \  smu_ss' snu_s's (f_nks - f_n'k's')
        (...)_k =  /  ---------------------------------- n_kt(G+q) n_kt^*(G'+q)
                   ‾‾ hw - (eps_n'k's'-eps_nks) + ih eta
                   t

        where n_kt(G+q) = n_nks,n'k+qs'(G+q) and

        bandsummation: pairwise (using spin-conserving time-reversal symmetry)

                    __ /
                    \  | smu_ss' snu_s's (f_nks - f_n'k's')
        (...)_k =   /  | ----------------------------------
                    ‾‾ | hw - (eps_n'k's'-eps_nks) + ih eta
                    t  \
                                                       \
                    smu_s's snu_ss' (f_nks - f_n'k's') |
        -delta_n'>n ---------------------------------- | n_kt(G+q) n_kt^*(G'+q)
                    hw + (eps_n'k's'-eps_nks) + ih eta |
                                                       /

        The integrand is added to the output array A_x multiplied with the
        supplied k-point weight.
        """
        # Get data, distributed in memory
        # Get bands and spins of the transitions
        n1_t, n2_t, s1_t, s2_t = kskptpair.get_transitions()
        # Get (f_n'k's' - f_nks), (eps_n'k's' - eps_nks) and the pair densities
        df_t, deps_t, n_tG = kskptpair.df_t, kskptpair.deps_t, kskptpair.n_tG

        x_wt = weight * self.get_temporal_part(n1_t, n2_t,
                                               s1_t, s2_t, df_t, deps_t)

        myslice = self.blocks1d.myslice

        if self.bundle_integrals:
            # Specify notation
            A_GwmyG = A_x

            x_tw = np.ascontiguousarray(x_wt.T)
            n_Gt = np.ascontiguousarray(n_tG.T)

            with self.timer('Set up ncc and nx'):
                ncc_Gt = n_Gt.conj()
                n_tmyG = n_tG[:, myslice]
                nx_twmyG = x_tw[:, :, np.newaxis] * n_tmyG[:, np.newaxis, :]

            with self.timer('Perform sum over t-transitions of ncc * nx'):
                mmmx(1.0, ncc_Gt, 'N', nx_twmyG, 'N',
                     1.0, A_GwmyG)  # slow step
        else:
            # Specify notation
            A_wmyGG = A_x

            with self.timer('Set up ncc and nx'):
                ncc_tG = n_tG.conj()
                n_myGt = np.ascontiguousarray(n_tG[:, myslice].T)
                nx_wmyGt = x_wt[:, np.newaxis, :] * n_myGt[np.newaxis, :, :]

            with self.timer('Perform sum over t-transitions of ncc * nx'):
                for nx_myGt, A_myGG in zip(nx_wmyGt, A_wmyGG):
                    mmmx(1.0, nx_myGt, 'N', ncc_tG, 'N',
                         1.0, A_myGG)  # slow step

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

               smu_ss' snu_s's (f_nks - f_n'k's')
        x_wt = ----------------------------------
               hw - (eps_n'k's'-eps_nks) + ih eta
        """
        # Get the right spin components
        scomps_t = get_smat_components(self.spincomponent, s1_t, s2_t)
        # Calculate nominator
        nom_t = - scomps_t * df_t  # df = f2 - f1
        # Calculate denominator
        denom_wt = self.wd.omega_w[:, np.newaxis] + 1j * self.eta\
            - deps_t[np.newaxis, :]  # de = e2 - e1

        return nom_t[np.newaxis, :] / denom_wt

    def get_pairwise_temporal_part(self, n1_t, n2_t, s1_t, s2_t, df_t, deps_t):
        """Get:
               /
               | smu_ss' snu_s's (f_nks - f_n'k's')
        x_wt = | ----------------------------------
               | hw - (eps_n'k's'-eps_nks) + ih eta
               \
                                                           \
                        smu_s's snu_ss' (f_nks - f_n'k's') |
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
        nom1_t = - scomps1_t * df_t  # df = f2 - f1
        nom2_t = - delta_t * scomps2_t * df_t
        # Calculate denominators
        denom1_wt = self.wd.omega_w[:, np.newaxis] + 1j * self.eta\
            - deps_t[np.newaxis, :]  # de = e2 - e1
        denom2_wt = self.wd.omega_w[:, np.newaxis] + 1j * self.eta\
            + deps_t[np.newaxis, :]

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
