import numpy as np
from abc import ABC, abstractmethod

from ase.units import Hartree

from gpaw.utilities.progressbar import ProgressBar

from gpaw.response import timer
from gpaw.response.kspair import KohnShamPair
from gpaw.response.pw_parallelization import block_partition
from gpaw.response.pair_functions import SingleQPWDescriptor


class PairFunctionIntegrator(ABC):
    r"""Baseclass for computing pair functions in the Kohn-Sham system of
    collinear periodic crystals in absence of spin-orbit coupling.

    A pair function is understood as any function, which can be constructed as
    a sum over transitions between Kohn-Sham eigenstates at k and k + q,
                  __  __  __                            __
               1  \   \   \                          1  \
    pf(q,z) =  ‾  /   /   /   pf_nks,n'k+qs'(q,z) =  ‾  /  pf_T(q,z)
               V  ‾‾  ‾‾  ‾‾                         V  ‾‾
                  k  n,n' s,s'                          T

    where z is decodes any additional variables (usually this will be some sort
    of complex frequency). In the notation used here, V is the crystal volume
    and T is a composit index encoding all relevant transitions:

    T: (n, k, s) -> (n', k + q, s')

    The sum over transitions can be split into two steps: (1) an integral over
    k-points k inside the 1st Brillouin Zone and (2) a sum over band and spin
    transitions t:

    t (composit transition index): (n, s) -> (n', s')
                  __                 __  __                  __
               1  \               1  \   \                1  \
    pf(q,z) =  ‾  /  pf_T(q,z) =  ‾  /   /  pf_kt(q,z) =  ‾  /  (...)_k
               V  ‾‾              V  ‾‾  ‾‾               V  ‾‾
                  T                  k   t                   k

    In the code, the k-point integral is handled by a KPointPairIntegral
    object, while the sum over band and spin transitions t is carried out in
    the self.add_integrand() method, which also defines the specific pair
    function in question.

    KPointPairIntegral:
       __
    1  \
    ‾  /  (...)_k
    V  ‾‾
       k

    self.add_integrand():
                __                __   __
                \                 \    \
    (...)_k  =  /  pf_kt(q,z)  =  /    /   pf_nks,n'k+qs'(q,z)
                ‾‾                ‾‾   ‾‾
                t                 n,n' s,s'

    In practise, the integration is carried out by letting the
    KPointPairIntegral extract individual KohnShamKPointPair objects, which
    contain all relevant information about the Kohn-Sham eigenstates at k and
    k + q for a number of specified spin and band transitions t.
    KPointPairIntegral.weighted_kpoint_pairs() generates these kptpairs along
    with their integral weights such that self._integrate() can construct the
    pair functions in a flexible, yet general manner.
    
    NB: Although it is not a fundamental limitation to pair functions as
    described above, the current implementation is based on a plane-wave
    represenation of spatial coordinates. This means that symmetries are
    analyzed with a plane-wave basis in mind, leaving room for further
    generalization in the future.
    """

    def __init__(self, gs, context, nblocks=1,
                 disable_point_group=False,
                 disable_time_reversal=False,
                 disable_non_symmorphic=True):
        """Construct the PairFunctionIntegrator

        Parameters
        ----------
        gs : ResponseGroundStateAdapter
        context : ResponseContext
        nblocks : int
            Distribute the pair function into nblocks. Useful when the pair
            function itself becomes a large array (read: memory limiting).
        disable_point_group : bool
            Do not use the point group symmetry operators.
        disable_time_reversal : bool
            Do not use time reversal symmetry.
        disable_non_symmorphic : bool
            Do no use non symmorphic symmetry operators.
        """
        self.gs = gs
        self.context = context

        # Communicators for distribution of memory and work
        (self.blockcomm,
         self.intrablockcomm) = self.create_communicators(nblocks)
        self.nblocks = self.blockcomm.size

        # The KohnShamPair class handles extraction of k-point pairs from the
        # ground state
        self.kspair = KohnShamPair(self.gs, self.context,
                                   # Distribution of work.
                                   # t-transitions are distributed through
                                   # blockcomm, k-points through
                                   # intrablockcomm.
                                   transitionblockscomm=self.blockcomm,
                                   kptblockcomm=self.intrablockcomm)

        # Symmetry flags
        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal
        self.disable_non_symmorphic = disable_non_symmorphic
        if (disable_time_reversal and disable_point_group
            and disable_non_symmorphic):
            self.disable_symmetries = True
        else:
            self.disable_symmetries = False

    @timer('Integrate pair function')
    def _integrate(self, out, n1_t, n2_t, s1_t, s2_t):
        """In-place pair function integration

        Parameters
        ----------
        out : PairFunction
            Output data structure
        n1_t : np.array
            Band index of k-point k for each transition t.
        n2_t : np.array
            Band index of k-point k + q for each transition t.
        s1_t : np.array
            Spin index of k-point k for each transition t.
        s2_t : np.array
            Spin index of k-point k + q for each transition t.

        Returns
        -------
        analyzer : PWSymmetryAnalyzer
        """
        # Initialize the plane-wave symmetry analyzer
        analyzer = self.get_pw_symmetry_analyzer(out.qpd)
        
        # Perform the actual integral as a point integral over k-point pairs
        integral = KPointPairPointIntegral(self.kspair, analyzer)
        weighted_kptpairs = integral.weighted_kpoint_pairs(n1_t, n2_t,
                                                           s1_t, s2_t)
        pb = ProgressBar(self.context.fd)  # pb with a generator is awkward
        for _, _ in pb.enumerate([None] * integral.ni):
            kptpair, weight = next(weighted_kptpairs)
            if weight is not None:
                assert kptpair is not None
                self.add_integrand(kptpair, weight, out)

        # Sum over the k-points, which have been distributed between processes
        with self.context.timer('Sum over distributed k-points'):
            self.intrablockcomm.sum(out.array)

        # Because the symmetry analyzer is used both to generate the k-point
        # integral domain *and* to symmetrize pair functions after the
        # integration, we have to return it. It would be good to split up these
        # two tasks, so that we don't need to pass the analyzer object around
        # in the code like this...
        return analyzer

    @abstractmethod
    def add_integrand(self, kptpair, weight, out):
        """Add the relevant integrand of the outer k-point integral to the
        output data structure 'out', weighted by 'weight' and constructed
        from the provided KohnShamKPointPair 'kptpair'.

        This method effectively defines the pair function in question.
        """

    def create_communicators(self, nblocks):
        """Create MPI communicators to distribute the memory needed to store
        large arrays and parallelize calculations when possible.

        Parameters
        ----------
        nblocks : int
            Separate large arrays into n different blocks. Each process
            allocates memory for the large arrays. By allocating only a
            fraction/block of the total arrays, the memory requirements are
            eased.

        Returns
        -------
        blockcomm : gpaw.mpi.Communicator
            Communicate between processes belonging to different memory blocks.
            In every communicator, there is one process for each block of
            memory, so that all blocks are represented.
            If nblocks < world.size, there will be size // nblocks different
            processes that allocate memory for the same block of the large
            arrays. Thus, there will be also size // nblocks different block
            communicators, grouping the processes into sets that allocate the
            entire arrays between them.
        intrablockcomm : gpaw.mpi.Communicator
            Communicate between processes belonging to the same memory block.
            There will be size // nblocks processes per memory block.
        """
        world = self.context.world
        blockcomm, intrablockcomm = block_partition(world, nblocks)

        return blockcomm, intrablockcomm

    def get_pw_descriptor(self, q_c, ecut=50, gammacentered=False):
        q_c = np.asarray(q_c, dtype=float)
        ecut = None if ecut is None else ecut / Hartree  # eV to Hartree
        gd = self.gs.gd

        qpd = SingleQPWDescriptor.from_q(q_c, ecut, gd,
                                        gammacentered=gammacentered)

        return qpd

    def get_pw_symmetry_analyzer(self, qpd):
        from gpaw.response.symmetry import PWSymmetryAnalyzer

        return PWSymmetryAnalyzer(
            self.gs.kd, qpd, self.context,
            disable_point_group=self.disable_point_group,
            disable_time_reversal=self.disable_time_reversal,
            disable_non_symmorphic=self.disable_non_symmorphic)

    def get_band_and_spin_transitions_domain(self, spinrot, nbands=None,
                                             bandsummation='pairwise'):
        """Generate all allowed band and spin transitions (transitions from
        occupied to occupied and from unoccupied to unoccupied are not
        allowed).

        Parameters
        ----------
        spinrot : str
            Spin rotation from k to k + q.
            Choices: 'u', 'd', '0' (= 'u' + 'd'), '-' and '+'.
            All rotations are included for spinrot=None ('0' + '+' + '-').
        nbands : int
            Maximum band index to include.
        bandsummation : str
            Band (and spin) summation for pairs of Kohn-Sham orbitals
            'pairwise': sum over pairs of bands (and spins)
            'double': double sum over band (and spin) indices.
        """
        # Include all bands, if nbands is None
        nspins = self.gs.nspins
        nbands = nbands or self.gs.bd.nbands
        assert nbands <= self.gs.bd.nbands
        nocc1 = self.kspair.nocc1
        nocc2 = self.kspair.nocc2

        n1_M, n2_M = get_band_transitions_domain(bandsummation, nbands,
                                                 nocc1=nocc1,
                                                 nocc2=nocc2)
        s1_S, s2_S = get_spin_transitions_domain(bandsummation,
                                                 spinrot, nspins)

        n1_t, n2_t, s1_t, s2_t = transitions_in_composite_index(n1_M, n2_M,
                                                                s1_S, s2_S)

        return n1_t, n2_t, s1_t, s2_t

    def get_basic_information(self):
        """Get basic information about the ground state and parallelization."""
        nspins = self.gs.nspins
        nbands = self.gs.bd.nbands
        nocc1 = self.kspair.nocc1
        nocc2 = self.kspair.nocc2
        nk = self.gs.kd.nbzkpts
        nik = self.gs.kd.nibzkpts

        wsize = self.context.world.size
        knsize = self.intrablockcomm.size
        bsize = self.blockcomm.size

        s = ''

        s += 'The pair function integration is based on a ground state with:\n'
        s += '    Number of spins: %d\n' % nspins
        s += '    Number of bands: %d\n' % nbands
        s += '    Number of completely occupied bands: %d\n' % nocc1
        s += '    Number of partially occupied bands: %d\n' % nocc2
        s += '    Number of kpoints: %d\n' % nk
        s += '    Number of irredicible kpoints: %d\n' % nik
        s += '\n'
        s += 'The pair function integration is performed in parallel with:\n'
        s += '    world.size: %d\n' % wsize
        s += '    intrablockcomm.size: %d\n' % knsize
        s += '    blockcomm.size: %d\n' % bsize

        return s


def get_band_transitions_domain(bandsummation, nbands, nocc1=None, nocc2=None):
    """Get all pairs of bands to sum over

    Parameters
    ----------
    bandsummation : str
        Band summation method
    nbands : int
        number of bands
    nocc1 : int
        number of completely filled bands
    nocc2 : int
        number of non-empty bands

    Returns
    -------
    n1_M : ndarray
        band index 1, M = (n1, n2) composite index
    n2_M : ndarray
        band index 2, M = (n1, n2) composite index
    """
    _get_band_transitions_domain =\
        create_get_band_transitions_domain(bandsummation)
    n1_M, n2_M = _get_band_transitions_domain(nbands)

    return remove_null_transitions(n1_M, n2_M, nocc1=nocc1, nocc2=nocc2)


def create_get_band_transitions_domain(bandsummation):
    """Creator component deciding how to carry out band summation."""
    if bandsummation == 'pairwise':
        return get_pairwise_band_transitions_domain
    elif bandsummation == 'double':
        return get_double_band_transitions_domain
    raise ValueError(bandsummation)


def get_double_band_transitions_domain(nbands):
    """Make a simple double sum"""
    n_n = np.arange(0, nbands)
    m_m = np.arange(0, nbands)
    n_nm, m_nm = np.meshgrid(n_n, m_m)
    n_M, m_M = n_nm.flatten(), m_nm.flatten()

    return n_M, m_M


def get_pairwise_band_transitions_domain(nbands):
    """Make a sum over all pairs"""
    n_n = range(0, nbands)
    n_M = []
    m_M = []
    for n in n_n:
        m_m = range(n, nbands)
        n_M += [n] * len(m_m)
        m_M += m_m

    return np.array(n_M), np.array(m_M)


def remove_null_transitions(n1_M, n2_M, nocc1=None, nocc2=None):
    """Remove pairs of bands, between which transitions are impossible"""
    n1_newM = []
    n2_newM = []
    for n1, n2 in zip(n1_M, n2_M):
        if nocc1 is not None and (n1 < nocc1 and n2 < nocc1):
            continue  # both bands are fully occupied
        elif nocc2 is not None and (n1 >= nocc2 and n2 >= nocc2):
            continue  # both bands are completely unoccupied
        n1_newM.append(n1)
        n2_newM.append(n2)

    return np.array(n1_newM), np.array(n2_newM)


def get_spin_transitions_domain(bandsummation, spinrot, nspins):
    """Get structure of the sum over spins

    Parameters
    ----------
    bandsummation : str
        Band summation method
    spinrot : str
        spin rotation
    nspins : int
        number of spin channels in ground state calculation

    Returns
    -------
    s1_s : ndarray
        spin index 1, S = (s1, s2) composite index
    s2_S : ndarray
        spin index 2, S = (s1, s2) composite index
    """
    _get_spin_transitions_domain =\
        create_get_spin_transitions_domain(bandsummation)
    return _get_spin_transitions_domain(spinrot, nspins)


def create_get_spin_transitions_domain(bandsummation):
    """Creator component deciding how to carry out spin summation."""
    if bandsummation == 'pairwise':
        return get_pairwise_spin_transitions_domain
    elif bandsummation == 'double':
        return get_double_spin_transitions_domain
    raise ValueError(bandsummation)


def get_double_spin_transitions_domain(spinrot, nspins):
    """Usual spin rotations forward in time"""
    if nspins == 1:
        if spinrot is None or spinrot == '0':
            s1_S = [0]
            s2_S = [0]
        else:
            raise ValueError(spinrot, nspins)
    else:
        if spinrot is None:
            s1_S = [0, 0, 1, 1]
            s2_S = [0, 1, 0, 1]
        elif spinrot == '0':
            s1_S = [0, 1]
            s2_S = [0, 1]
        elif spinrot == 'u':
            s1_S = [0]
            s2_S = [0]
        elif spinrot == 'd':
            s1_S = [1]
            s2_S = [1]
        elif spinrot == '-':
            s1_S = [0]  # spin up
            s2_S = [1]  # spin down
        elif spinrot == '+':
            s1_S = [1]  # spin down
            s2_S = [0]  # spin up
        else:
            raise ValueError(spinrot)

    return np.array(s1_S), np.array(s2_S)


def get_pairwise_spin_transitions_domain(spinrot, nspins):
    """In a sum over pairs, transitions including a spin rotation may have to
    include terms, propagating backwards in time."""
    if spinrot in ['+', '-']:
        assert nspins == 2
        return np.array([0, 1]), np.array([1, 0])
    else:
        return get_double_spin_transitions_domain(spinrot, nspins)


def transitions_in_composite_index(n1_M, n2_M, s1_S, s2_S):
    """Use a composite index t for transitions (n, s) -> (n', s')."""
    n1_MS, s1_MS = np.meshgrid(n1_M, s1_S)
    n2_MS, s2_MS = np.meshgrid(n2_M, s2_S)
    return n1_MS.flatten(), n2_MS.flatten(), s1_MS.flatten(), s2_MS.flatten()


class KPointPairIntegral(ABC):
    r"""Baseclass for reciprocal space integrals of the first Brillouin Zone,
    where the integrand is a sum over transitions between any number of states
    at the wave vectors k and k + q (referred to as a k-point pair).

    Definition (V is the total crystal hypervolume and D is the dimension of
    the crystal):
       __
    1  \                 1    /
    ‾  /  (...)_k  =  ‾‾‾‾‾‾  |dk (...)_k
    V  ‾‾             (2π)^D  /
       k

    NB: In the current implementation, the dimension is fixed to 3. This is
    sensible for pair functions which are functions of position (such as the
    four-component Kohn-Sham susceptibility tensor), and in most circumstances
    a change in dimensionality can be accomplished simply by adding an extra
    prefactor to the integral elsewhere.
    NB: The current implementation is running on backbone functionality to
    analyze symmetries within a plane wave representation of real-space
    coordinates. This could be generalized further in the future. See the
    PWSymmetryAnalyzer in gpaw.response.symmetry.
    """

    def __init__(self, kspair, analyzer):
        """Construct a KPointPairIntegral corresponding to a given q-point.

        Parameters
        ----------
        kspair : KohnShamPair
            Object responsible for extracting all relevant information about
            the k-point pairs from the underlying ground state calculation.
        analyzer : PWSymmetryAnalyzer
            Object responsible for analyzing the symmetries of the q-point in
            question, for which the k-point pair integral is constructed.
        """
        self.gs = kspair.gs
        self.kspair = kspair
        self.q_c = analyzer.qpd.q_c

        # Prepare the k-point pair integral
        bzk_kc, weight_k = self.get_kpoint_domain(analyzer)
        bzk_ipc, weight_i = self.slice_kpoint_domain(bzk_kc, weight_k)
        self._domain = (bzk_ipc, weight_i)
        self.ni = len(weight_i)

    def weighted_kpoint_pairs(self, n1_t, n2_t, s1_t, s2_t):
        r"""Generate all k-point pairs in the integral along with their
        integral weights.

        The reciprocal space integral is estimated as the sum over a discrete
        k-point domain. The domain will genererally depend on the integration
        method as well as the symmetry of the crystal.

        Definition:
                                        __
           1    /            ~     1    \   (2π)^D
        ‾‾‾‾‾‾  |dk (...)_k  =  ‾‾‾‾‾‾  /   ‾‾‾‾‾‾ w_kr (...)_kr
        (2π)^D  /               (2π)^D  ‾‾  Nk V0
                                        kr
                                __
                             ~  \
                             =  /  iw_kr (...)_kr
                                ‾‾
                                kr

        The sum over kr denotes the reduced k-point domain specified by the
        integration method (a reduced selection of Nkr points from the ground
        state k-point grid of Nk total points in the entire 1BZ). Each point
        is weighted by its k-point volume in the ground state k-point grid

                      (2π)^D
        kpointvol  =  ‾‾‾‾‾‾,
                      Nk V0

        and an additional individual k-point weight w_kr specific to the
        integration method (V0 denotes the cell volume). Together with the
        integral prefactor, these make up the integral weight

                   1
        iw_kr = ‾‾‾‾‾‾ kpointvol w_kr
                (2π)^D

        Parameters
        ----------
        n1_t : np.array
            Band index of k-point k for each transition t.
        n2_t : np.array
            Band index of k-point k + q for each transition t.
        s1_t : np.array
            Spin index of k-point k for each transition t.
        s2_t : np.array
            Spin index of k-point k + q for each transition t.
        """
        # Calculate prefactors
        outer_prefactor = 1 / (2 * np.pi)**3
        V = self.crystal_volume()  # V = Nk * V0
        kpointvol = (2 * np.pi)**3 / V
        prefactor = outer_prefactor * kpointvol

        # Generate k-point pairs
        for k_pc, weight in zip(*self._domain):
            if weight is None:
                integral_weight = None
            else:
                integral_weight = prefactor * weight
            kptpair = self.kspair.get_kpoint_pairs(n1_t, n2_t,
                                                   k_pc, k_pc + self.q_c,
                                                   s1_t, s2_t)
            yield kptpair, integral_weight

    @abstractmethod
    def get_kpoint_domain(self, analyzer):
        """Use the PWSymmetryAnalyzer to define and weight the k-point domain.

        Returns
        -------
        bzk_kc : np.array
            k-points to integrate in relative coordinates.
        weight_k : np.array
            Integral weight of each k-point in the integral.
        """

    def slice_kpoint_domain(self, bzk_kc, weight_k):
        """When integrating over k-points, slice the domain in pieces with one
        k-point per process each.

        Returns
        -------
        bzk_ipc : nd.array
            k-points (relative) coordinates for each process for each iteration
        """
        comm = self.kspair.kptblockcomm
        rank, size = comm.rank, comm.size

        nk = bzk_kc.shape[0]
        ni = (nk + size - 1) // size
        bzk_ipc = [bzk_kc[i * size:(i + 1) * size] for i in range(ni)]

        # Extract the weight corresponding to the process' own k-point pair
        weight_ip = [weight_k[i * size:(i + 1) * size] for i in range(ni)]
        weight_i = [None] * len(weight_ip)
        for i, w_p in enumerate(weight_ip):
            if rank in range(len(w_p)):
                weight_i[i] = w_p[rank]

        return bzk_ipc, weight_i

    def crystal_volume(self):
        """Calculate the total crystal volume, V = Nk * V0, corresponding to
        the ground state k-point grid."""
        return self.gs.kd.nbzkpts * self.gs.volume


class KPointPairPointIntegral(KPointPairIntegral):
    r"""Reciprocal space integral of k-point pairs in the first Brillouin Zone,
    estimated as a point integral over all k-points of the ground state k-point
    grid.

    Definition:
                                   __
       1    /           ~     1    \   (2π)^D
    ‾‾‾‾‾‾  |dk (...)_k =  ‾‾‾‾‾‾  /   ‾‾‾‾‾‾ (...)_k
    (2π)^D  /              (2π)^D  ‾‾  Nk V0
                                   k

    """

    def get_kpoint_domain(self, analyzer):
        # Generate k-point domain in relative coordinates
        K_gK = analyzer.group_kpoints()  # What is g? XXX
        bzk_kc = np.array([self.gs.kd.bzk_kc[K_K[0]] for
                           K_K in K_gK])  # Why only K=0? XXX

        # Get the k-point weights from the symmetry analyzer
        weight_k = np.array([analyzer.get_kpoint_weight(k_c)
                             for k_c in bzk_kc])

        return bzk_kc, weight_k
