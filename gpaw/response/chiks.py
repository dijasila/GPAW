import numpy as np
from time import ctime

from ase.units import Hartree

from gpaw.utilities.blas import mmmx

from gpaw.response import ResponseContext, timer
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.pw_parallelization import PlaneWaveBlockDistributor
from gpaw.response.kspair import PlaneWavePairDensity
from gpaw.response.pair_integrator import PairFunctionIntegrator
from gpaw.response.pair_functions import (SingleQPWDescriptor,
                                          LatticePeriodicPairFunction)


class ChiKS:
    """Temporary backwards compatibility layer."""

    def __init__(self, gs, context=None, nblocks=1,
                 disable_point_group=False, disable_time_reversal=False,
                 disable_non_symmorphic=True,
                 eta=0.2, ecut=50, gammacentered=False, nbands=None,
                 bundle_integrals=True, bandsummation='pairwise'):

        self.calc = ChiKSCalculator(
            gs, context=context, nblocks=nblocks,
            ecut=ecut, gammacentered=gammacentered,
            nbands=nbands,
            bundle_integrals=bundle_integrals, bandsummation=bandsummation,
            disable_point_group=disable_point_group,
            disable_time_reversal=disable_time_reversal,
            disable_non_symmorphic=disable_non_symmorphic)
        self.context = self.calc.context
        self.gs = self.calc.gs
        self.nblocks = self.calc.nblocks
        self.gammacentered = self.calc.gammacentered

        self.eta = eta

        # Hard-coded, but expected properties
        self.kpointintegration = 'point integration'

    def calculate(self, q_c, frequencies, spincomponent='all'):
        if isinstance(frequencies, FrequencyDescriptor):
            wd = frequencies
        else:
            wd = FrequencyDescriptor.from_array_or_dict(frequencies)
        self.wd = wd
        self.blockdist = PlaneWaveBlockDistributor(self.context.world,
                                                   self.calc.blockcomm,
                                                   self.calc.intrablockcomm)

        return self.calc.calculate(spincomponent, q_c, wd, eta=self.eta)

    def get_pw_descriptor(self, q_c):
        return self.calc.get_pw_descriptor(q_c)

    @timer('Distribute frequencies')
    def distribute_frequencies(self, chiks_wGG):
        return self.blockdist.distribute_frequencies(chiks_wGG, len(self.wd))


class ChiKSData(LatticePeriodicPairFunction):  # rename to ChiKS               XXX
    """
    Some documentation here!                                                   XXX
    """

    def __init__(self, spincomponent, pd, wd, eta,
                 blockdist, distribution='WgG'):
        """
        Document me!                                                           XXX
        """
        self.spincomponent = spincomponent
        self.eta = eta
        super().__init__(pd, wd, blockdist, distribution=distribution)


class ChiKSCalculator(PairFunctionIntegrator):
    r"""Calculator class for the four-component Kohn-Sham susceptibility tensor
    of collinear systems in absence of spin-orbit coupling,
    see [PRB 103, 245110 (2021)]:
                              __  __   __
                           1  \   \    \
    χ_KS,GG'^μν(q,ω+iη) =  ‾  /   /    /   σ^μ_ss' σ^ν_s's (f_nks - f_n'k+qs')
                           V  ‾‾  ‾‾   ‾‾
                              k   n,n' s,s'
                                        n_nks,n'k+qs'(G+q) n_n'k+qs',nks(-G'-q)
                                      x ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                                            ħω - (ε_n'k+qs' - ε_nks) + iħη

    where the matrix elements

    n_nks,n'k+qs'(G+q) = <nks| e^-i(G+q)r |n'k+qs'>_V0

    are the unit cell normalized plane-wave pair densities of each transition.
    """

    def __init__(self, gs, context=None, nblocks=1,
                 ecut=50, gammacentered=False,
                 nbands=None,
                 bundle_integrals=True, bandsummation='pairwise',
                 **kwargs):
        """Contruct the ChiKSCalculator

        Parameters
        ----------
        gs : ResponseGroundStateAdapter
        context : ResponseContext
        nblocks : int
            Distribute the chiks_wGG array into nblocks (where nblocks is a
            divisor of context.world.size)
        ecut : float (or None)
            Plane-wave cutoff in eV
        gammacentered : bool
            Center the grid of plane waves around the Γ-point (or the q-vector)
        nbands : int
            Number of bands to include in the sum over states
        bandsummation : str
            Band summation strategy (does not change the result, but can affect
            the run-time).
            'pairwise': sum over pairs of bands
            'double': double sum over band indices.
        bundle_integrals : bool
            Do the k-point integrals (large matrix multiplications)
            simultaneously for all frequencies (does not change the result,
            but can affect the overall performance).
        kwargs : see gpaw.response.pair_integrator.PairFunctionIntegrator
        """
        if context is None:
            context = ResponseContext()
        assert isinstance(context, ResponseContext)

        super().__init__(gs, context, nblocks=nblocks, **kwargs)

        self.ecut = None if ecut is None else ecut / Hartree  # eV to Hartree
        self.gammacentered = gammacentered
        self.nbands = nbands
        self.bundle_integrals = bundle_integrals
        self.bandsummation = bandsummation

        self.pair_density = PlaneWavePairDensity(self.kspair)

    def calculate(self, spincomponent, q_c, wd, eta=0.2):
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

        Returns
        -------
        pd : PWDescriptor
        chiks_wGG : np.array
        """
        assert isinstance(wd, FrequencyDescriptor)
        eta = eta / Hartree  # eV -> Hartree

        # Set up the internal plane-wave descriptor
        pdi = self.get_pw_descriptor(q_c, internal=True)

        # Analyze the requested spin component
        spinrot = get_spin_rotation(spincomponent)

        # Prepare to sum over bands and spins
        n1_t, n2_t, s1_t, s2_t = self.get_band_and_spin_transitions_domain(
            spinrot, nbands=self.nbands, bandsummation=self.bandsummation)

        self.context.print(self.get_information(
            pdi, len(wd), eta, spincomponent, self.nbands, len(n1_t)))

        self.context.print('Initializing pair densities')
        self.pair_density.initialize(pdi)

        # Create ChiKS data structure
        chiks = self.create_chiks(spincomponent, wd, pdi, eta)

        # Perform the actual integration
        analyzer = self._integrate(chiks, n1_t, n2_t, s1_t, s2_t)

        # Apply symmetries and map to output format
        # Return chiks data structure instead                                  XXX
        pd, chiks_WgG = self.post_process(chiks, analyzer)

        return pd, chiks_WgG

    def get_pw_descriptor(self, q_c, internal=False):
        """Get plane-wave descriptor for a calculation with wave vector q_c."""
        return self._get_pw_descriptor(q_c, ecut=self.ecut,
                                       gammacentered=self.gammacentered,
                                       internal=internal)

    def _get_pw_descriptor(self, q_c, ecut=50 / Hartree, gammacentered=False,
                           internal=True):
        """Get plane-wave descriptor for the wave vector q_c.

        Parameters
        ----------
        q_c : list or ndarray
            Wave vector in relative coordinates
        ecut : float (or None)
            Plane-wave cutoff in Hartree
        gammacentered : bool
            Center the grid of plane waves around the Γ-point (or the q-vector)
        internal : bool
            When using symmetries, the actual calculation of chiks must happen
            using a q-centered plane wave basis. If internal==True, as it is by
            default, the internal plane wave basis (used in the integration of
            chiks.array) is returned, otherwise the external descriptor is
            returned, corresponding to the requested chiks.
        """
        q_c = np.asarray(q_c, dtype=float)
        gd = self.gs.gd

        # Update to internal basis, if needed
        if internal and gammacentered and not self.disable_symmetries:
            # In order to make use of the symmetries of the system to reduce
            # the k-point integration, the internal code assumes a plane wave
            # basis which is centered at q in reciprocal space.
            gammacentered = False
            # If we want to compute the pair function on a plane wave grid
            # which is effectively centered in the gamma point instead of q, we
            # need to extend the internal ecut such that the q-centered grid
            # encompasses all reciprocal lattice points inside the gamma-
            # centered sphere.
            # The reduction to the global gamma-centered basis will then be
            # carried out as a post processing step.

            # Compute the extended internal ecut
            B_cv = 2.0 * np.pi * gd.icell_cv  # Reciprocal lattice vectors
            q_v = q_c @ B_cv
            ecut = get_ecut_to_encompass_centered_sphere(q_v, ecut)

        pd = SingleQPWDescriptor.from_q(q_c, ecut, gd,
                                        gammacentered=gammacentered)

        return pd

    def create_chiks(self, spincomponent, wd, pd, eta):
        """
        Some documentation here!                                               XXX
        """
        if self.bundle_integrals:
            distribution = 'GWg'
        else:
            distribution = 'WgG'
        blockdist = PlaneWaveBlockDistributor(self.context.world,
                                              self.blockcomm,
                                              self.intrablockcomm)

        return ChiKSData(spincomponent, pd, wd, eta,
                         blockdist, distribution=distribution)

    @timer('Add integrand to chiks')
    def add_integrand(self, kptpair, weight, chiks):
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
        self.pair_density(kptpair, chiks.pd)

        # Extract the ingredients from the KohnShamKPointPair
        # Get bands and spins of the transitions
        n1_t, n2_t, s1_t, s2_t = kptpair.get_transitions()
        # Get (f_n'k's' - f_nks), (ε_n'k's' - ε_nks) as well as n_kt(G+q)
        df_t, deps_t, n_tG = kptpair.df_t, kptpair.deps_t, kptpair.n_tG

        # Calculate the frequency dependence of the integrand
        if chiks.spincomponent in ['00', 'all'] and self.gs.nspins == 1:
            weight = 2 * weight
        x_Wt = weight * get_temporal_part(chiks.spincomponent,
                                          chiks.wd.omega_w, chiks.eta,
                                          n1_t, n2_t, s1_t, s2_t,
                                          df_t, deps_t,
                                          self.bandsummation)

        # Let each process handle its own slice of integration
        myslice = chiks.blocks1d.myslice

        if chiks.distribution == 'GWg':
            # Specify notation
            chiks_GWg = chiks.array

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
        elif chiks.distribution == 'WgG':
            # Specify notation
            chiks_WgG = chiks.array

            with self.context.timer('Set up ncc and nx'):
                ncc_tG = n_tG.conj()
                n_gt = np.ascontiguousarray(n_tG[:, myslice].T)
                nx_Wgt = x_Wt[:, np.newaxis, :] * n_gt[np.newaxis, :, :]

            with self.context.timer('Perform sum over t-transitions of '
                                    'ncc * nx'):
                for nx_gt, chiks_gG in zip(nx_Wgt, chiks_WgG):
                    mmmx(1.0, nx_gt, 'N', ncc_tG, 'N',
                         1.0, chiks_gG)  # slow step
        else:
            raise ValueError(f'Invalid distribution {chiks.distribution}')

    @timer('Post processing')
    def post_process(self, chiks, analyzer):
        if chiks.distribution == 'GWg':
            # chiks_GWg = chiks.array
            chiks_WgG = chiks.array.transpose((1, 2, 0))
        else:
            chiks_WgG = chiks.array
        nw = len(chiks.wd)

        # Distribute over frequencies
        tmp_wGG = chiks.blockdist.distribute_as(chiks_WgG, nw, 'wGG')
        with self.context.timer('Symmetrizing chiks_wGG'):
            analyzer.symmetrize_wGG(tmp_wGG)
        # Distribute over plane waves
        chiks_WgG[:] = chiks.blockdist.distribute_as(tmp_wGG, nw, 'WgG')

        if self.gammacentered and not self.disable_symmetries:
            pdi = chiks.pd  # internal pd
            assert not pdi.gammacentered
            # Reduce the q-centered plane-wave basis used internally to the
            # gammacentered basis
            q_c = pdi.kd.bzk_kc[0]
            pd = self.get_pw_descriptor(q_c)
            chiks_WgG = map_WgG_array_to_reduced_pd(pdi, pd,
                                                    chiks.blockdist, chiks_WgG)
        else:
            pd = chiks.pd

        return pd, chiks_WgG

    def get_information(self, pd, nw, eta, spincomponent, nbands, nt):
        r"""Get information about the χ_KS,GG'^μν(q,ω+iη) calculation"""
        from gpaw.utilities.memory import maxrss

        q_c = pd.kd.bzk_kc[0]
        ecut = pd.ecut * Hartree
        Asize = nw * pd.ngmax**2 * 16. / 1024**2 / self.blockcomm.size
        cmem = maxrss() / 1024**2

        s = '\n'

        s += 'Calculating the Kohn-Sham susceptibility with:\n'
        s += '    Spin component: %s\n' % spincomponent
        s += '    q_c: [%f, %f, %f]\n' % (q_c[0], q_c[1], q_c[2])
        s += '    Number of frequency points: %d\n' % nw
        s += '    Broadening (eta): %f\n' % eta
        if nbands is None:
            s += '    Bands included: All\n'
        else:
            s += '    Number of bands included: %d\n' % nbands
        s += 'Resulting in:\n'
        s += '    A total number of band and spin transitions of: %d\n' % nt
        s += '\n'

        s += self.get_basic_information()
        s += '\n'

        s += 'Plane-wave basis of the Kohn-Sham susceptibility:\n'
        s += '    Planewave cutoff: %f\n' % ecut
        s += '    Number of planewaves: %d\n' % pd.ngmax
        s += '    Memory estimates:\n'
        s += '        A_wGG: %f M / cpu\n' % Asize
        s += '        Memory usage before allocation: %f M / cpu\n' % cmem
        s += '\n'
        s += '%s\n' % ctime()

        return s


def get_ecut_to_encompass_centered_sphere(q_v, ecut):
    """Calculate the minimal ecut which results in a q-centered plane wave
    basis containing all the reciprocal lattice vectors G, which lie inside a
    specific gamma-centered sphere:

    |G|^2 < 2 * ecut
    """
    q = np.linalg.norm(q_v)
    ecut = ecut + q * (np.sqrt(2 * ecut) + q / 2)

    return ecut


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
