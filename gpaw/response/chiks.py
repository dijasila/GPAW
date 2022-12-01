from functools import partial

import numpy as np

from ase.units import Hartree

from gpaw.utilities.blas import mmmx

from gpaw.response import ResponseContext, timer
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.pw_parallelization import (Blocks1D,
                                              PlaneWaveBlockDistributor)
# from gpaw.response.kslrf import PlaneWaveKSLRF, PairFunctionIntegrator
from gpaw.response.kslrf import PairFunctionIntegrator
from gpaw.response.kspair import PlaneWavePairDensity


class ChiKS:
    """Tmp. backwards compatibility."""

    def __init__(self, gs, context=None, nblocks=1,
                 disable_point_group=False, disable_time_reversal=False,
                 disable_non_symmorphic=True,
                 eta=0.2, ecut=50, gammacentered=False, nbands=None,
                 bundle_integrals=True, bandsummation='pairwise'):

        self.calc = ChiKSCalculator(
            gs, context=context, nblocks=nblocks,
            disable_point_group=disable_point_group,
            disable_time_reversal=disable_time_reversal,
            disable_non_symmorphic=disable_non_symmorphic)
        self.context = self.calc.context
        self.gs = self.calc.gs
        self.nblocks = self.calc.nblocks

        self.eta = eta
        self.ecut = ecut
        self.gammacentered = gammacentered
        self.nbands = nbands
        self.bundle_integrals = bundle_integrals
        self.bandsummation = bandsummation

        # Hard-coded, but expected properties
        self.kpointintegration = 'point integration'

    def calculate(self, q_c, frequencies, spincomponent='all', A_x=None):
        if isinstance(frequencies, FrequencyDescriptor):
            wd = frequencies
        else:
            wd = FrequencyDescriptor.from_array_or_dict(frequencies)
        self.wd = wd
        self.blockdist = PlaneWaveBlockDistributor(self.context.world,
                                                   self.calc.blockcomm,
                                                   self.calc.intrablockcomm)

        return self.calc.calculate(spincomponent, q_c, wd,
                                   eta=self.eta,
                                   ecut=self.ecut,
                                   gammacentered=self.gammacentered,
                                   nbands=self.nbands,
                                   bundle_integrals=self.bundle_integrals,
                                   bandsummation=self.bandsummation)

    def get_PWDescriptor(self, q_c):
        return self.calc.get_PWDescriptor(q_c, ecut=self.ecut,
                                          gammacentered=self.gammacentered)

    @timer('Distribute frequencies')
    def distribute_frequencies(self, chiks_wGG):
        return self.blockdist.distribute_frequencies(chiks_wGG, len(self.wd))


r'''
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
    def calculate(self, q_c, frequencies, spincomponent='all', A_x=None):
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

            with self.context.timer('Set up ncc and nx'):
                ncc_Gt = n_Gt.conj()
                n_tmyG = n_tG[:, myslice]
                nx_twmyG = x_tw[:, :, np.newaxis] * n_tmyG[:, np.newaxis, :]

            with self.context.timer('Perform sum over t-transitions '
                                    'of ncc * nx'):
                mmmx(1.0, ncc_Gt, 'N', nx_twmyG, 'N',
                     1.0, A_GwmyG)  # slow step
        else:
            # Specify notation
            A_wmyGG = A_x

            with self.context.timer('Set up ncc and nx'):
                ncc_tG = n_tG.conj()
                n_myGt = np.ascontiguousarray(n_tG[:, myslice].T)
                nx_wmyGt = x_wt[:, np.newaxis, :] * n_myGt[np.newaxis, :, :]

            with self.context.timer('Perform sum over t-transitions of '
                                    'ncc * nx'):
                for nx_myGt, A_myGG in zip(nx_wmyGt, A_wmyGG):
                    mmmx(1.0, nx_myGt, 'N', ncc_tG, 'N',
                         1.0, A_myGG)  # slow step

    @timer('Get temporal part')
    def get_temporal_part(self, n1_t, n2_t, s1_t, s2_t, df_t, deps_t):
        return get_temporal_part(self.spincomponent, self.wd.omega_w, self.eta,
                                 n1_t, n2_t, s1_t, s2_t, df_t, deps_t,
                                 self.bandsummation)
'''


class ChiKSCalculator(PairFunctionIntegrator):
    """
    Some documentation here!                                                   XXX
    """

    def __init__(self, gs, context=None, nblocks=1, **kwargs):
        """Contruct the ChiKSCalculator

        Parameters
        ----------
        gs : ResponseGroundStateAdapter
        context : ResponseContext
        nblocks : int
            Distribute the chiks_wGG array into nblocks (where nblocks is a
            divisor of context.world.size)
        kwargs : see gpaw.kslrf.PairFunctionIntegrator
        """
        if context is None:
            context = ResponseContext()
        assert isinstance(context, ResponseContext)

        super().__init__(gs, context, nblocks=nblocks, **kwargs)

        self.pair_density = PlaneWavePairDensity(self.kspair)

    def calculate(self, spincomponent, q_c, wd,
                  eta=0.2,
                  ecut=50,
                  gammacentered=False,
                  nbands=None,
                  chiks_x=None,
                  bundle_integrals=True,
                  bandsummation='pairwise'):
        r"""Calculate χ_KS,GG'^μν(q,ω+iη)

        Parameters
        ----------
        spincomponent : str
            Spin component (μν) of the Kohn-Sham susceptibility.
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented.
        q_c : list or np.array
            Wave vector in relative coordinates
        wd : FrequencyDescriptor
            (Real part ω) of the frequencies where χ_KS,GG'^μν(q,ω+iη) is
            evaluated
        eta : float
            Imaginary part η of the frequencies where χ_KS,GG'^μν(q,ω+iη) is
            evaluated
        ecut : float (or None)
            Plane-wave cutoff in eV
        gammacentered : bool
            Center the grid of plane waves around the Γ-point (or the q-vector)
        nbands : int
            Number of bands to include in the sum over states
        chiks_x : np.array
            Pre-existing integration buffer
        bandsummation : str
            Band summation strategy (does not change the result, but can affect
            the run-time).
            'pairwise': sum over pairs of bands
            'double': double sum over band indices.
        bundle_integrals : bool
            Do the k-point integrals (large matrix multiplications)
            simultaneously for all frequencies (does not change the result,
            but can affect the overall performance).

        Returns
        -------
        pd : PWDescriptor
        chiks_wGG : np.array
        """
        assert isinstance(wd, FrequencyDescriptor)

        # Set inputs on self, so that they can be accessed later
        self.spincomponent = spincomponent
        self.wd = wd
        self.eta = eta / Hartree  # eV -> Hartree
        self.bundle_integrals = bundle_integrals
        self.bandsummation = bandsummation

        # Set up the internal plane-wave descriptor
        ecut = None if ecut is None else ecut / Hartree  # eV to Hartree
        pdi = self._get_PWDescriptor(q_c, ecut=ecut,
                                     gammacentered=gammacentered)

        # Analyze the requested spin component
        spinrot = get_spin_rotation(spincomponent)

        # Prepare to sum over bands and spins
        n1_t, n2_t, s1_t, s2_t = self.get_band_and_spin_transitions_domain(
            spinrot, nbands=nbands, bandsummation=bandsummation)

        self.print_information(pdi, len(wd), eta,
                               spincomponent, nbands, len(n1_t))

        self.context.print('Initializing pair densities')
        self.pair_density.initialize(pdi)

        # Allocate array (or clean up existing buffer)
        blocks1d = Blocks1D(self.blockcomm, pdi.ngmax)
        chiks_x = self.set_up_array(len(wd), blocks1d, chiks_x=chiks_x)

        # Perform the actual integration
        analyzer = self._integrate(pdi, chiks_x, n1_t, n2_t, s1_t, s2_t)

        # Apply symmetries and map to output format
        pd, chiks_WgG = self.post_process(pdi, chiks_x, analyzer,
                                          ecut=ecut,
                                          gammacentered=gammacentered)

        return pd, chiks_WgG

    def set_up_array(self, nw, blocks1d, chiks_x=None):
        """Initialize the chiks_x array."""
        nG = blocks1d.N
        nGlocal = blocks1d.nlocal
        localsize = nw * nGlocal * nG

        if self.bundle_integrals:
            # Set up chiks_GWg
            shape = (nG, nw, nGlocal)
            if chiks_x is not None:
                chiks_GWg = chiks_x[:localsize].reshape(shape)
                chiks_GWg[:] = 0.0
            else:
                chiks_GWg = np.zeros(shape, complex)

            return chiks_GWg
        else:
            # Set up chiks_WgG
            shape = (nw, nGlocal, nG)
            if chiks_x is not None:
                chiks_WgG = chiks_x[:localsize].reshape(shape)
                chiks_WgG[:] = 0.0
            else:
                chiks_WgG = np.zeros(shape, complex)

            return chiks_WgG

    @timer('Add integrand to chiks_x')
    def add_integrand(self, kptpair, weight, pd, chiks_x):
        r"""Use the PlaneWavePairDensity object to calculate the integrand for
        all relevant transitions of the given k-point pair, k -> k + q.

        Depending on the bandsummation parameter, the integrand of the
        collinear four-component Kohn-Sham susceptibility tensor (in the
        absence of spin-orbit coupling) is calculated as:

        bandsummation: double

                   __
                   \  σ^μ_ss' σ^ν_s's (f_nks - f_n'k's')
        (...)_k =  /  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ n_kt(G+q) n_kt^*(G'+q)
                   ‾‾   ħω - (ε_n'k's' - ε_nks) + iħη
                   t

        where n_kt(G+q) = n_nks,n'k+qs'(G+q) and

        bandsummation: pairwise

                    __ /
                    \  | σ^μ_ss' σ^ν_s's (f_nks - f_n'k's')
        (...)_k =   /  | ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                    ‾‾ |   ħω - (ε_n'k's' - ε_nks) + iħη
                    t  \
                                                       \
                    σ^μ_s's σ^ν_ss' (f_nks - f_n'k's') |
           - δ_n'>n ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ | n_kt(G+q) n_kt^*(G'+q)
                      ħω + (ε_n'k's' - ε_nks) + iħη    |
                                                       /

        The integrand is added to the output array chiks_x multiplied with the
        supplied kptpair integral weight.
        """
        # Calculate the pair densities
        self.pair_density(kptpair, pd)

        # Extract the ingredients from the KohnShamKPointPair
        # Get bands and spins of the transitions
        n1_t, n2_t, s1_t, s2_t = kptpair.get_transitions()
        # Get (f_n'k's' - f_nks), (ε_n'k's' - ε_nks) as well as n_kt(G+q)
        df_t, deps_t, n_tG = kptpair.df_t, kptpair.deps_t, kptpair.n_tG

        # Calculate the frequency dependence of the integrand
        x_Wt = weight * get_temporal_part(self.spincomponent,
                                          self.wd.omega_w, self.eta,
                                          n1_t, n2_t, s1_t, s2_t,
                                          df_t, deps_t,
                                          self.bandsummation)

        # Let each process handle its own slice of integration
        blocks1d = Blocks1D(self.blockcomm, pd.ngmax)
        myslice = blocks1d.myslice

        if self.bundle_integrals:
            # Specify notation
            chiks_GWg = chiks_x

            x_tW = np.ascontiguousarray(x_Wt.T)
            n_Gt = np.ascontiguousarray(n_tG.T)

            with self.context.timer('Set up ncc and nx'):
                ncc_Gt = n_Gt.conj()
                n_tg = n_tG[:, myslice]
                nx_tWg = x_tW[:, :, np.newaxis] * n_tg[:, np.newaxis, :]

            with self.context.timer('Perform sum over t-transitions '
                                    'of ncc * nx'):
                mmmx(1.0, ncc_Gt, 'N', nx_tWg, 'N',
                     1.0, chiks_GWg)  # slow step
        else:
            # Specify notation
            chiks_WgG = chiks_x

            with self.context.timer('Set up ncc and nx'):
                ncc_tG = n_tG.conj()
                n_gt = np.ascontiguousarray(n_tG[:, myslice].T)
                nx_Wgt = x_Wt[:, np.newaxis, :] * n_gt[np.newaxis, :, :]

            with self.context.timer('Perform sum over t-transitions of '
                                    'ncc * nx'):
                for nx_gt, chiks_gG in zip(nx_Wgt, chiks_WgG):
                    mmmx(1.0, nx_gt, 'N', ncc_tG, 'N',
                         1.0, chiks_gG)  # slow step

    @timer('Post processing')
    def post_process(self, pdi, chiks_x, analyzer,
                     ecut=50 / Hartree, gammacentered=False):
        if self.bundle_integrals:
            # chiks_x = chiks_GWg
            chiks_WgG = chiks_x.transpose((1, 2, 0))
        else:
            chiks_WgG = chiks_x
        nw = chiks_WgG.shape[0]

        # Distribute over frequencies
        blockdist = PlaneWaveBlockDistributor(self.context.world,
                                              self.blockcomm,
                                              self.intrablockcomm)
        tmp_wGG = blockdist.distribute_as(chiks_WgG, nw, 'wGG')
        with self.context.timer('Symmetrizing chiks_wGG'):
            analyzer.symmetrize_wGG(tmp_wGG)
        # Distribute over plane waves
        chiks_WgG[:] = blockdist.distribute_as(tmp_wGG, nw, 'WgG')

        if gammacentered and not self.disable_symmetries:
            assert not pdi.gammacentered
            # Reduce the q-centered plane-wave basis used internally to the
            # gammacentered basis
            q_c = pdi.kd.bzk_kc[0]
            pd = self._get_PWDescriptor(q_c, ecut=ecut, gammacentered=True,
                                        internal=False)
            chiks_WgG = map_WgG_array_to_reduced_pd(pdi, pd,
                                                    blockdist, chiks_WgG)
        else:
            pd = pdi

        return pd, chiks_WgG

    def print_information(self, pd, nw, eta, spincomponent, nbands, nt):
        """Print information about the joint density of states calculation"""
        from gpaw.utilities.memory import maxrss

        q_c = pd.kd.bzk_kc[0]
        ecut = pd.ecut * Hartree
        Asize = nw * pd.ngmax**2 * 16. / 1024**2 / self.blockcomm.size
        
        p = partial(self.context.print, flush=False)

        p('Calculating the Kohn-Sham susceptibility with:')
        p('    Spin component: %s' % spincomponent)
        p('    q_c: [%f, %f, %f]' % (q_c[0], q_c[1], q_c[2]))
        p('    Number of frequency points: %d' % nw)
        p('    Broadening (eta): %f' % eta)
        if nbands is None:
            p('    Bands included: All')
        else:
            p('    Number of bands included: %d' % nbands)
        p('Resulting in:')
        p('    A total number of band and spin transitions of: %d' % nt)
        p('')

        self.print_basic_information()

        p('The Kohn-Sham susceptibility is calculated in a plane-wave basis:')
        p('    Planewave cutoff: %f' % ecut)
        p('    Number of planewaves: %d' % pd.ngmax)
        p('    Memory estimates:')
        p('        A_wGG: %f M / cpu' % Asize)
        p('        Memory usage before allocation: %f M / cpu' % (maxrss() /
                                                                  1024**2))
        self.context.print()


def map_WgG_array_to_reduced_pd(pdi, pd, blockdist, in_WgG):
    """Map an output array to a reduced plane wave basis which is
    completely contained within the original basis, that is, from pdi to
    pd."""
    from gpaw.pw.descriptor import PWMapping

    # Initialize the basis mapping
    pwmapping = PWMapping(pdi, pd)
    G2_GG = tuple(np.meshgrid(pwmapping.G2_G1, pwmapping.G2_G1,
                              indexing='ij'))
    G1_GG = tuple(np.meshgrid(pwmapping.G1, pwmapping.G1,
                              indexing='ij'))

    # Distribute over frequencies
    nw = in_WgG.shape[0]
    tmp_wGG = blockdist.distribute_as(in_WgG, nw, 'wGG')

    # Allocate array in the new basis
    nG = pd.ngmax
    new_tmp_shape = (tmp_wGG.shape[0], nG, nG)
    new_tmp_wGG = np.zeros(new_tmp_shape, complex)

    # Extract values in the global basis
    for w, tmp_GG in enumerate(tmp_wGG):
        new_tmp_wGG[w][G2_GG] = tmp_GG[G1_GG]

    # Distribute over plane waves
    out_WgG = blockdist.distribute_as(new_tmp_wGG, nw, 'WgG')

    return out_WgG


def get_temporal_part(spincomponent, omega_w, eta,
                      n1_t, n2_t, s1_t, s2_t, df_t, deps_t, bandsummation):
    """Get the temporal part of a (causal linear) susceptibility integrand."""
    _get_temporal_part = create_get_temporal_part(bandsummation)
    return _get_temporal_part(spincomponent, s1_t, s2_t,
                              df_t, deps_t, omega_w, eta,
                              n1_t, n2_t)


def create_get_temporal_part(bandsummation):
    """Creator component, deciding how to calculate the temporal part"""
    if bandsummation == 'double':
        return get_double_temporal_part
    elif bandsummation == 'pairwise':
        return get_pairwise_temporal_part
    raise ValueError(bandsummation)


def get_double_temporal_part(spincomponent, s1_t, s2_t,
                             df_t, deps_t, omega_w, eta,
                             *unused):
    r"""Get:

             σ^μ_ss' σ^ν_s's (f_nks - f_n'k's')
    Χ_t(ω) = ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
               ħω - (ε_n'k's' - ε_nks) + iħη
    """
    # Get the right spin components
    scomps_t = get_smat_components(spincomponent, s1_t, s2_t)
    # Calculate nominator
    nom_t = - scomps_t * df_t  # df = f2 - f1
    # Calculate denominator
    denom_wt = omega_w[:, np.newaxis] + 1j * eta\
        - deps_t[np.newaxis, :]  # de = e2 - e1

    return nom_t[np.newaxis, :] / denom_wt


def get_pairwise_temporal_part(spincomponent, s1_t, s2_t,
                               df_t, deps_t, omega_w, eta,
                               n1_t, n2_t):
    r"""Get:

             /
             | σ^μ_ss' σ^ν_s's (f_nks - f_n'k's')
    Χ_t(ω) = | ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
             |   ħω - (ε_n'k's' - ε_nks) + iħη
             \
                                                           \
                        σ^μ_s's σ^ν_ss' (f_nks - f_n'k's') |
               - δ_n'>n ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ |
                          ħω + (ε_n'k's' - ε_nks) + iħη    |
                                                           /
    """
    # Kroenecker delta
    delta_t = np.ones(len(n1_t))
    delta_t[n2_t <= n1_t] = 0
    # Get the right spin components
    scomps1_t = get_smat_components(spincomponent, s1_t, s2_t)
    scomps2_t = get_smat_components(spincomponent, s2_t, s1_t)
    # Calculate nominators
    nom1_t = - scomps1_t * df_t  # df = f2 - f1
    nom2_t = - delta_t * scomps2_t * df_t
    # Calculate denominators
    denom1_wt = omega_w[:, np.newaxis] + 1j * eta\
        - deps_t[np.newaxis, :]  # de = e2 - e1
    denom2_wt = omega_w[:, np.newaxis] + 1j * eta\
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
