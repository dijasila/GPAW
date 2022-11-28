import numpy as np
from functools import partial
from time import ctime
from abc import ABC, abstractmethod

from ase.units import Hartree

import gpaw
from gpaw.utilities.memory import maxrss
from gpaw.utilities.progressbar import ProgressBar

from gpaw.response import ResponseGroundStateAdapter, ResponseContext, timer
from gpaw.response.kspair import KohnShamPair
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.pw_parallelization import (block_partition, Blocks1D,
                                              PlaneWaveBlockDistributor)


class KohnShamLinearResponseFunction:  # Future PairFunctionIntegrator? XXX
    r"""Class calculating linear response functions in the Kohn-Sham system of
    a periodic crystal.

    In the Lehmann representation (frequency domain), linear response functions
    are written as a sum over transitions between the ground and excited energy
    eigenstates with poles at the transition energies. In the Kohn-Sham system
    such a sum can be evaluated explicitly, as only excited states where a
    single electron has been moved from an occupied single-particle Kohn-Sham
    orbital to an unoccupied one contribute.

    Resultantly, any linear response function in the Kohn-Sham system can be
    written as a sum over transitions between pairs of occupied and unoccupied
    Kohn-Sham orbitals.

    Furthermore, for periodic systems the response is diagonal in the reduced
    wave vector q (confined to the 1st Brillouin Zone), meaning that one can
    treat each momentum transfer (hbar q) independently.

    Currently, only collinear Kohn-Sham systems are supported. That is, all
    relevant transitions can be written in terms of band indexes, k-points and
    spins for a given wave vector q, leading to the following definition of the
    Kohn-Sham linear response function,
                   __  __  __                         __
                1  \   \   \                       1  \
    chi(q,w) =  ‾  /   /   /   f_nks,n'k+qs'(w) =  ‾  /  f_T(q,w)
                V  ‾‾  ‾‾  ‾‾                      V  ‾‾
                   k  n,n' s,s'                       T

    where V is the crystal volume and,

    T (composit transition index): (n, k, s) -> (n', k + q, s')


    The sum over transitions can be split into two steps: (1) an integral over
    k-points k inside the 1st Brillouin Zone and (2) a sum over band and spin
    transitions t:

    t (composit transition index): (n, s) -> (n', s')
                   __                __  __                  __
                1  \              1  \   \                1  \
    chi(q,w) =  ‾  /  f_T(q,w) =  ‾  /   /  f_k,t(q,w) =  ‾  /  (...)_k
                V  ‾‾             V  ‾‾  ‾‾               V  ‾‾
                   T                 k   t                   k

    In the code, the k-point integral is handled by the Integator object, and
    the sum over band and spin transitions t is carried out in the
    self.add_integrand() method, which also defines the specific response
    function.

    Integrator:
       __
    1  \
    ‾  /  (...)_k
    V  ‾‾
       k

    self.add_integrand():
                __                __   __
                \                 \    \
    (...)_k  =  /  f_k,t(q,w)  =  /    /   f_nks,n'k+qs'(w)
                ‾‾                ‾‾   ‾‾
                t                 n,n' s,s'

    In practise, the Integrator supplies an individual k-point weight wk, for
    the self.add_integrand() method to multiply each integrand with, so that
    add_integrand adds wk (...)_k to the output array for each k-point.
    """

    def __init__(self, gs, context=None, mode=None,
                 bandsummation='pairwise', nbands=None,
                 kpointintegration=None, nblocks=1):
        """Construct the KSLRF object

        Parameters
        ----------
        gs : ResponseGroundStateAdapter
        context : ResponseContext
        mode: str
            Calculation mode.
            Currently, only a plane wave mode is implemented.
        bandsummation : str
            Band summation for pairs of Kohn-Sham orbitals
            'pairwise': sum over pairs of bands
            'double': double sum over band indices.
        nbands : int
            Maximum band index to include.
        kpointintegration : str
            Brillouin Zone integration for the Kohn-Sham orbital wave vector.
            Currently, only point integration is supported
        nblocks : int
            Divide the response function storage into nblocks. Useful when the
            response function is large and memory requirements are restrictive.

        Attributes
        ----------
        kspair : gpaw.response.pair.KohnShamPair instance
            Class for handling pairs of Kohn-Sham orbitals
        pme : gpaw.response.pair.PairMatrixElement instance
            Class for calculating transition matrix elements for pairs of
            Kohn-Sham orbitals
        integrator : Integrator instance
            The integrator class is a general class for Brillouin Zone
            integration. The user defined integrand is integrated over k-points
            and summed over a given band and spin domain.

        Callables
        ---------
        self.add_integrand(kskptpair, weight, tmp_x, *args, **kwargs) : func
            Add the integrand for a given part of the domain to output array
        self.calculate(*args, **kwargs) : func
            Runs the calculation, returning the response function.
            Returned format can varry depending on response and mode.
        """
        assert isinstance(gs, ResponseGroundStateAdapter)
        self.gs = gs
        if context is None:
            self.context = ResponseContext()
        else:
            assert isinstance(context, ResponseContext)
            self.context = context

        self.context.print('Initializing KohnShamLinearResponseFunction')

        # Communicators for parallelization
        self.blockcomm = None
        self.intrablockcomm = None
        self.initialize_communicators(nblocks)
        self.nblocks = self.blockcomm.size

        # The KohnShamPair class handles data extraction from ground state
        self.kspair = KohnShamPair(gs, self.context,
                                   # Let each process handle slow steps only
                                   # for a fraction of all transitions.
                                   # t-transitions are distributed through
                                   # blockcomm, k-points through
                                   # intrablockcomm.
                                   transitionblockscomm=self.blockcomm,
                                   kptblockcomm=self.intrablockcomm)

        self.mode = mode

        self.bandsummation = bandsummation
        self.nbands = nbands or self.gs.bd.nbands
        assert self.nbands <= self.gs.bd.nbands
        self.nocc1 = self.kspair.nocc1  # number of completely filled bands
        self.nocc2 = self.kspair.nocc2  # number of non-empty bands

        self.kpointintegration = kpointintegration
        self.integrator = create_integrator(self)
        # Each integrator might take some extra input kwargs
        self.extraintargs = {}

        # Attributes related to the specific response function
        self.pme = None

    def initialize_communicators(self, nblocks):
        """Set up MPI communicators to distribute the memory needed to store
        large arrays and parallelize calculations when possible.

        Parameters
        ----------
        nblocks : int
            Separate large arrays into n different blocks. Each process
            allocates memory for the large arrays. By allocating only a
            fraction/block of the total arrays, the memory requirements are
            eased.

        Sets
        ----
        blockcomm : gpaw.mpi.Communicator
            Communicate between processes belonging to different memory blocks.
            In every communicator, there is one process for each block of
            memory, so that all blocks are represented.
            If nblocks < world.size, there will be size // nblocks different
            processes that allocate memory for the same block of the large
            arrays. Thus, there will be also size // nblocks different inter
            block communicators, grouping the processes into sets that allocate
            the entire arrays between them.
        intrablockcomm : gpaw.mpi.Communicator
            Communicate between processes belonging to the same memory block.
            There will be size // nblocks processes per memory block.
        """
        world = self.context.world
        self.blockcomm, self.intrablockcomm = block_partition(world,
                                                              nblocks)

        self.context.print('Number of blocks:', nblocks)

    @timer('Calculate Kohn-Sham linear response function')
    def calculate(self, spinrot=None, A_x=None):
        return self._calculate(spinrot, A_x)

    def _calculate(self, spinrot, A_x):
        """In-place calculation of the response function

        Parameters
        ----------
        spinrot : str
            Select spin rotation.
            Choices: 'u', 'd', '0' (= 'u' + 'd'), '-'= and '+'
            All rotations are included for spinrot=None ('0' + '+' + '-').
        A_x : ndarray
            Output array. If None, the output array is created.
        """
        self.spinrot = spinrot
        # Prepare to sum over bands and spins
        n1_t, n2_t, s1_t, s2_t = self.get_band_spin_transitions_domain()

        # Print information about the prepared calculation
        self.print_information(len(n1_t))
        if gpaw.dry_run:  # Exit after setting up
            self.context.print('    Dry run exit')
            raise SystemExit

        self.context.print('\n----------', flush=False)
        self.context.print('Initializing PairMatrixElement')
        self.initialize_pme()

        A_x = self.setup_output_array(A_x)

        self.integrator.integrate(n1_t, n2_t, s1_t, s2_t, A_x,
                                  **self.extraintargs)

        # Different calculation modes might want the response function output
        # in different formats
        out = self.post_process(A_x)

        return out

    def get_band_spin_transitions_domain(self):
        """Generate all allowed band and spin transitions.

        If only a subset of possible spin rotations are considered
        (examples: s1 = s2 or s2 = 1 - s1), do not include others
        in the sum over transitions.
        """
        n1_M, n2_M = get_band_transitions_domain(self.bandsummation,
                                                 self.nbands,
                                                 nocc1=self.nocc1,
                                                 nocc2=self.nocc2)
        s1_S, s2_S = get_spin_transitions_domain(self.bandsummation,
                                                 self.spinrot,
                                                 self.gs.nspins)

        return transitions_in_composite_index(n1_M, n2_M, s1_S, s2_S)

    def setup_output_array(self, A_x):
        raise NotImplementedError('Output array depends on mode')

    def get_ks_kpoint_pairs(self, k_pv, *args, **kwargs):
        raise NotImplementedError('Integrated pairs of states depend on '
                                  'response and mode')

    def initialize_pme(self, *args, **kwargs):
        raise NotImplementedError('Calculator method for matrix elements '
                                  'depend on response and mode')

    def calculate_pme(self, kskptpair, *args, **kwargs):
        raise NotImplementedError('Calculator method for matrix elements '
                                  'depend on response and mode')

    def add_integrand(self, kskptpair, weight, tmp_x, *args, **kwargs):
        raise NotImplementedError('Integrand depends on response and mode')

    def post_process(self, A_x):
        raise NotImplementedError('Post processing depends on mode')

    def print_information(self, nt):
        """Basic information about the input ground state, parallelization
        and sum over states"""
        ns = self.gs.nspins
        nbands = self.nbands
        nocc = self.nocc1
        npocc = self.nocc2
        nk = self.gs.kd.nbzkpts
        nik = self.gs.kd.nibzkpts

        if gpaw.dry_run:
            from gpaw.mpi import DryRunCommunicator
            size = gpaw.dry_run
            world = DryRunCommunicator(size)
        else:
            world = self.context.world
        wsize = world.size
        knsize = self.intrablockcomm.size
        bsize = self.blockcomm.size

        spinrot = self.spinrot

        p = partial(self.context.print, flush=False)

        p('Called a response.kslrf.KohnShamLinearResponseFunction.calculate()')
        p('%s' % ctime())
        p('Using a Kohn-Sham ground state with:')
        p('    Number of spins: %d' % ns)
        p('    Number of bands: %d' % nbands)
        p('    Number of completely occupied states: %d' % nocc)
        p('    Number of partially occupied states: %d' % npocc)
        p('    Number of kpoints: %d' % nk)
        p('    Number of irredicible kpoints: %d' % nik)
        p('')
        p('The response function calculation is performed in parallel with:')
        p('    world.size: %d' % wsize)
        p('    intrablockcomm.size: %d' % knsize)
        p('    blockcomm.size: %d' % bsize)
        p('')
        p('The sum over band and spin transitions is performed using:')
        p('    Spin rotation: %s' % spinrot)
        p('    Total number of composite band and spin transitions: %d' % nt)
        self.context.print('')


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


class PlaneWaveKSLRF(KohnShamLinearResponseFunction):
    """Class for doing KS-LRF calculations in plane wave mode"""

    def __init__(self, *args, eta=0.2, ecut=50, gammacentered=False,
                 disable_point_group=False, disable_time_reversal=False,
                 disable_non_symmorphic=True, bundle_integrals=True,
                 kpointintegration='point integration', **kwargs):
        """Initialize the plane wave calculator mode.
        In plane wave mode, the linear response function is calculated for a
        given set of frequencies. The spatial part is expanded in plane waves
        for a given momentum transfer q within the first Brillouin Zone.

        Parameters
        ----------
        eta : float
            Energy broadening of spectra.
        ecut : float
            Energy cutoff for the plane wave representation.
        gammacentered : bool
            Center the grid of plane waves around the gamma point or q-vector.
        disable_point_group : bool
            Do not use the point group symmetry operators.
        disable_time_reversal : bool
            Do not use time reversal symmetry.
        disable_non_symmorphic : bool
            Do no use non symmorphic symmetry operators.
        bundle_integrals : bool
            Do the k-point integrals (large matrix multiplications)
            simultaneously for all frequencies.
            Can be switched of, if this step forces calculations out of memory.
        """

        # Avoid any mode ambiguity
        if 'mode' in kwargs.keys():
            mode = kwargs.pop('mode')
            assert mode == 'pw'

        KSLRF = KohnShamLinearResponseFunction
        KSLRF.__init__(self, *args, mode='pw',
                       kpointintegration=kpointintegration, **kwargs)

        self.eta = eta / Hartree
        self.ecut = None if ecut is None else ecut / Hartree
        self.gammacentered = gammacentered

        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal
        self.disable_non_symmorphic = disable_non_symmorphic
        if (disable_time_reversal and disable_point_group
            and disable_non_symmorphic):
            self.disable_symmetries = True
        else:
            self.disable_symmetries = False

        self.bundle_integrals = bundle_integrals

        # Attributes related to specific q, given to self.calculate()
        self.pd = None  # Plane wave descriptor for given momentum transfer q
        self.pwsa = None  # Plane wave symmetry analyzer for given q
        self.wd = None  # Frequency descriptor for the given frequencies
        self.blocks1d = None  # Plane wave block parallelization descriptor
        self.blockdist = None  # Plane wave block distributor

    @timer('Calculate Kohn-Sham linear response function in plane wave mode')
    def calculate(self, q_c, frequencies, spinrot=None, A_x=None):
        """
        Parameters
        ----------
        q_c : list or ndarray
            Wave vector
        frequencies : ndarray, dict or FrequencyDescriptor
            Array of frequencies to evaluate the response function at,
            dictionary of parameters for build-in frequency grids or a
            descriptor of those frequencies.

        Returns
        -------
        pd : Planewave descriptor
            Planewave descriptor for q_c.
        A_wGG : ndarray
            The linear response function.
        """
        # Set up internal plane wave description with the given wave vector q
        self.pd = self._get_PWDescriptor(q_c)
        self.pwsa = self.get_PWSymmetryAnalyzer(self.pd)

        # Set up frequency descriptor for the given frequencies
        self.wd = self.get_FrequencyDescriptor(frequencies)

        # Set up block parallelization
        self.blocks1d = Blocks1D(self.blockcomm, self.pd.ngmax)
        self.blockdist = PlaneWaveBlockDistributor(self.context.world,
                                                   self.blockcomm,
                                                   self.intrablockcomm)

        # In-place calculation
        return self._calculate(spinrot, A_x)

    def get_PWDescriptor(self, q_c):
        """Get the resulting plane wave description for a calculation
        with wave vector q_c."""
        return self._get_PWDescriptor(q_c, internal=False)

    def _get_PWDescriptor(self, q_c, internal=True):
        """Get plane wave descriptor for the wave vector q_c.

        Parameters
        ----------
        q_c : list or ndarray
            Wave vector
        internal : bool
            When using symmetries, the actual calculation of chiks_wGG must
            happen using a q-centered plane wave basis. If internal==True,
            as it is by default, the internal plane wave basis used in the
            evaluation of chiks_wGG is returned, otherwise the external
            descriptor is returned, corresponding to the requested chiks_wGG.
        """
        gd = self.gs.gd

        # Fall back to ecut of requested plane wave basis
        ecut = self.ecut
        gammacentered = self.gammacentered

        # Update to internal basis, if needed
        if internal and gammacentered and not self.disable_symmetries:
            # In order to make use of the symmetries of the system to reduce
            # the k-point integration, the internal code assumes a plane wave
            # basis which is centered at q in reciprocal space.
            gammacentered = False
            # If we want to compute the linear response function on a plane
            # wave grid which is effectively centered in the gamma point
            # instead of q, we need to extend the internal ecut such that the
            # q-centered grid encompasses all reciprocal lattice points inside
            # the gamma-centered sphere.
            # The reduction to the global gamma-centered basis will then be
            # carried out as a post processing step.

            # Compute the extended internal ecut
            q_c = np.asarray(q_c, dtype=float)
            B_cv = 2.0 * np.pi * gd.icell_cv  # Reciprocal lattice vectors
            q_v = q_c @ B_cv
            ecut = get_ecut_to_encompass_centered_sphere(q_v, ecut)

        pd = get_PWDescriptor(ecut, gd, q_c, gammacentered=gammacentered)

        return pd

    @timer('Get PW symmetry analyser')
    def get_PWSymmetryAnalyzer(self, pd):
        from gpaw.response.symmetry import PWSymmetryAnalyzer

        return PWSymmetryAnalyzer(
            self.gs.kd, pd, self.context,
            disable_point_group=self.disable_point_group,
            disable_time_reversal=self.disable_time_reversal,
            disable_non_symmorphic=self.disable_non_symmorphic)

    def get_FrequencyDescriptor(self, frequencies):
        """Get the frequency descriptor for a certain input frequencies."""
        if isinstance(frequencies, FrequencyDescriptor):
            return frequencies
        else:
            return FrequencyDescriptor.from_array_or_dict(frequencies)

    def print_information(self, nt):
        """Basic information about the input ground state, parallelization,
        sum over states and calculated response function array."""
        KohnShamLinearResponseFunction.print_information(self, nt)

        pd = self.pd
        q_c = pd.kd.bzk_kc[0]
        nw = len(self.wd)
        eta = self.eta * Hartree
        ecut = pd.ecut * Hartree
        ngmax = pd.ngmax
        Asize = nw * pd.ngmax**2 * 16. / 1024**2 / self.blockcomm.size

        p = partial(self.context.print, flush=False)

        p('The response function is calculated in the PlaneWave mode, using:')
        p('    q_c: [%f, %f, %f]' % (q_c[0], q_c[1], q_c[2]))
        p('    Number of frequency points: %d' % nw)
        p('    Broadening (eta): %f' % eta)
        p('    Planewave cutoff: %f' % ecut)
        p('    Number of planewaves: %d' % ngmax)
        p('')
        p('    Memory estimates:')
        p('        A_wGG: %f M / cpu' % Asize)
        p('        Memory usage before allocation: %f M / cpu' % (maxrss() /
                                                                  1024**2))
        self.context.print('')

    def setup_output_array(self, A_x=None):
        """Initialize the output array in blocks"""
        # Could use some more documentation XXX
        nG = self.blocks1d.N
        nw = len(self.wd)

        nGlocal = self.blocks1d.nlocal
        localsize = nw * nGlocal * nG
        # if self.blockcomm.rank == 0:
        #     assert self.Gb - self.Ga >= 3
        # assert mynG * (self.blockcomm.size - 1) < nG
        if self.bundle_integrals:
            # Setup A_GwG
            shape = (nG, nw, nGlocal)
            if A_x is not None:
                A_GwG = A_x[:localsize].reshape(shape)
                A_GwG[:] = 0.0
            else:
                A_GwG = np.zeros(shape, complex)

            return A_GwG
        else:
            # Setup A_wGG
            shape = (nw, nGlocal, nG)
            if A_x is not None:
                A_wGG = A_x[:localsize].reshape(shape)
                A_wGG[:] = 0.0
            else:
                A_wGG = np.zeros(shape, complex)

            return A_wGG

    def get_ks_kpoint_pairs(self, k_pv, n1_t, n2_t, s1_t, s2_t):
        """Get all pairs of Kohn-Sham transitions:

        (n1_t, k_c, s1_t) -> (n2_t, k_c + q_c, s2_t)

        for each process with its own k-point.
        """
        k_pc = np.array([np.dot(self.pd.gd.cell_cv, k_v) / (2 * np.pi)
                         for k_v in k_pv])
        q_c = self.pd.kd.bzk_kc[0]
        return self.kspair.get_kpoint_pairs(n1_t, n2_t, k_pc, k_pc + q_c,
                                            s1_t, s2_t)

    def initialize_pme(self):
        self.pme.initialize(self.pd)

    def calculate_pme(self, kskptpair):
        self.pme(kskptpair, self.pd)

    def add_integrand(self, kskptpair, weight, tmp_x, **kwargs):
        raise NotImplementedError('Integrand depends on response')

    @timer('Post processing')
    def post_process(self, A_x):
        if self.bundle_integrals:
            # A_x = A_GwG
            A_wGG = A_x.transpose((1, 2, 0))
        else:
            A_wGG = A_x

        # distribute over frequencies
        tmpA_wGG = self.distribute_as(A_wGG, 'wGG')
        with self.context.timer('Symmetrizing Kohn-Sham linear '
                                'response function'):
            self.pwsa.symmetrize_wGG(tmpA_wGG)
        # distribute over plane waves
        A_wGG[:] = self.distribute_as(tmpA_wGG, 'WgG')

        if self.gammacentered and not self.disable_symmetries:
            # Reduce the q-specific basis to the global basis
            q_c = self.pd.kd.bzk_kc[0]
            pd = self.get_PWDescriptor(q_c)
            A_wGG = self.map_to(pd, A_wGG)

        return self.pd, A_wGG

    def map_to(self, pd, A_wGG):
        """Map the output array to a reduced plane wave basis (which will
        be adopted as the global basis)."""
        from gpaw.pw.descriptor import PWMapping

        # Initialize the basis mapping
        pwmapping = PWMapping(self.pd, pd)
        G2_GG = tuple(np.meshgrid(pwmapping.G2_G1, pwmapping.G2_G1,
                                  indexing='ij'))
        G1_GG = tuple(np.meshgrid(pwmapping.G1, pwmapping.G1,
                                  indexing='ij'))

        # Distribute over frequencies
        tmpA_wGG = self.distribute_as(A_wGG, 'wGG')

        # Change relevant properties on self to support the global basis
        # from this point onwards
        self.pd = pd
        self.blocks1d = Blocks1D(self.blockcomm, self.pd.ngmax)

        # Allocate array in the new basis
        nG = self.blocks1d.N
        tmpshape = (tmpA_wGG.shape[0], nG, nG)
        newtmpA_wGG = np.zeros(tmpshape, complex)

        # Extract values in the global basis
        for w, tmpA_GG in enumerate(tmpA_wGG):
            newtmpA_wGG[w][G2_GG] = tmpA_GG[G1_GG]

        # Distribute over plane waves
        newA_wGG = self.distribute_as(newtmpA_wGG, 'WgG')

        return newA_wGG

    @timer('Redistribute memory')
    def distribute_as(self, in_wGG, out_dist):
        return self.blockdist.distribute_as(in_wGG, len(self.wd), out_dist)

    @timer('Distribute frequencies')
    def distribute_frequencies(self, chiks_wGG):
        return self.blockdist.distribute_frequencies(chiks_wGG, len(self.wd))


def get_ecut_to_encompass_centered_sphere(q_v, ecut):
    """Calculate the minimal ecut which results in a q-centered plane wave
    basis containing all the reciprocal lattice vectors G, which lie inside a
    specific gamma-centered sphere:

    |G|^2 < 2 * ecut
    """
    q = np.linalg.norm(q_v)
    ecut = ecut + q * (np.sqrt(2 * ecut) + q / 2)

    return ecut


def get_PWDescriptor(ecut, gd, q_c, gammacentered=False):
    """Get the plane wave descriptor for a specific wave vector q_c."""
    from gpaw.kpt_descriptor import KPointDescriptor
    from gpaw.pw.descriptor import PWDescriptor

    q_c = np.asarray(q_c, dtype=float)
    qd = KPointDescriptor([q_c])

    pd = PWDescriptor(ecut, gd, complex, qd,
                      gammacentered=gammacentered)

    return pd


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
    a change in dimensionality can be acomplished simply by adding an extra
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
        self.q_c = analyzer.pd.kd.bzk_kc[0]

        # Prepare the k-point pair integral
        bzk_kc, weight_k = self.get_kpoint_domain(analyzer)
        bzk_ipc, weight_i = self.slice_kpoint_domain(bzk_kc, weight_k)
        self._domain = (bzk_ipc, weight_i)
        self.ni = len(weight_i)

    def weighted_kpoint_pairs(self, n1_t, n2_t, s1_t, s2_t):
        r"""Generate all k-point pairs in the integral along with their integral
        weights.

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
        """
        # Calculate prefactors
        outer_prefactor = 1 / (2 * np.pi)**3
        V = self.crystal_volume()  # V = Nk * V0
        kpointvol = (2 * np.pi)**3 / V
        prefactor = outer_prefactor * kpointvol

        # Generate k-point pairs
        for k_pc, weight in self._domain:
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


class Integrator:  # --> KPointPairIntegrator in the future? XXX
    r"""Baseclass for reciprocal space integrals of the first Brillouin Zone,
    where the integrand is a sum over transitions in bands and spin.

    Definition (V is the total crystal volume and D is the dimension of the
    crystal):
       __
    1  \                 1     /
    ‾  /  (...)_k  =  ‾‾‾‾‾‾‾  |dk (...)_k
    V  ‾‾             (2pi)^D  /
       k

    NB: In the current implementation, the dimension is fixed to 3. This is
    sensible for pair functions which are a function of position (such as the
    susceptibility), but not for e.g. a joint density of states of a lower
    dimensional crystal.
    """
    def __init__(self, kslrf):  # Make independent of kslrf in the future? XXX
        """
        Parameters
        ----------
        kslrf : KohnShamLinearResponseFunction instance
        """
        self.kslrf = kslrf
        self.context = self.kslrf.context

    @timer('Integrate response function')
    def integrate(self, n1_t, n2_t, s1_t, s2_t, out_x, **kwargs):
        r"""Estimate the reciprocal space integral as the sum over a discrete
        k-point domain. The domain will genererally depend on the integration
        method as well as the symmetry of the crystal.

        Definition:
                                          __
           1     /            ~     A     \   (2pi)^D
        ‾‾‾‾‾‾‾  |dk (...)_k  =  ‾‾‾‾‾‾‾  /   ‾‾‾‾‾‾‾  wkr (...)_kr
        (2pi)^D  /               (2pi)^D  ‾‾   Nk V0
                                          kr
        The sum over kr denotes the reduced k-point domain specified by the
        integration method (a reduced selection of Nkr points from the ground
        state k-point grid of Nk total points in the entire 1BZ). Each point
        is weighted by its k-point volume on the ground state k-point grid
                      (2pi)^D
        kpointvol  =  ‾‾‾‾‾‾‾,
                       Nk V0
        and an additional individual k-point weight wkr specific to the
        integration method (V0 denotes the cell volume). Furthermore, the
        integration method may define an extra integration prefactor A.
        """
        bzk_kv, weight_k = self.get_kpoint_domain()

        # Calculate prefactors
        A = self.calculate_bzint_prefactor(bzk_kv)
        outer_prefactor = A / (2 * np.pi)**3
        V = self.calculate_crystal_volume()  # V = Nk * V0
        kpointvol = (2 * np.pi)**3 / V
        prefactor = outer_prefactor * kpointvol

        # Perform the sum over the k-point domain w.o. prefactors
        tmp_x = np.zeros_like(out_x)
        self._integrate(bzk_kv, weight_k,
                        n1_t, n2_t, s1_t, s2_t, tmp_x, **kwargs)

        # Add integrated response function to the output with prefactors
        out_x /= prefactor
        out_x += tmp_x
        out_x *= prefactor

        return out_x

    def get_kpoint_domain(self):
        raise NotImplementedError('Domain depends on integration method')

    def calculate_bzint_prefactor(self, bzk_kv):
        raise NotImplementedError('Prefactor depends on integration method')

    def calculate_crystal_volume(self):
        """Calculate the total crystal volume, V = Nk * V0, corresponding to
        the ground state k-point grid."""
        # Get the total number of k-points on the ground state k-point grid
        gs = self.kslrf.gs
        return gs.kd.nbzkpts * gs.volume

    def _integrate(self, bzk_kv, weight_k,
                   n1_t, n2_t, s1_t, s2_t, tmp_x, **kwargs):
        r"""Do the actual reciprocal space integral as a simple weighted sum
        over the k-point domain, where the integrand is calculated externally
        as a sum over transitions in bands and spin.

        Definition (kr denotes the k-point domain and wkr the weights):
        __
        \
        /   wkr (...)_kr
        ‾‾
        kr
        """
        # tmp_x should be zero prior to the in-place integration
        assert np.allclose(tmp_x, 0.)

        # Slice domain
        bzk_ipv, weight_i = self.slice_kpoint_domain(bzk_kv, weight_k)

        # Perform sum over k-points
        pb = ProgressBar(self.context.fd)
        # Each process will do its own k-points, but it has to follow the
        # others, as it may have to send them information about its
        # partition of the ground state
        self.context.print('\nIntegrating response function')
        for i, k_pv in pb.enumerate(bzk_ipv):
            kskptpair = self.kslrf.get_ks_kpoint_pairs(k_pv, n1_t, n2_t,
                                                       s1_t, s2_t)
            if kskptpair is not None:
                weight = weight_i[i]
                assert weight is not None
                self.kslrf.calculate_pme(kskptpair)
                self.kslrf.add_integrand(kskptpair, weight,
                                         tmp_x, **kwargs)

        # Sum over the k-points that have been distributed between processes
        with self.context.timer('Sum over distributed k-points'):
            self.kslrf.intrablockcomm.sum(tmp_x)

    def slice_kpoint_domain(self, bzk_kv, weight_k):
        """When integrating over k-points, slice the domain in pieces with one
        k-point per process each.

        Returns
        -------
        bzk_ipv : nd.array
            k-points coordinates for each process for each iteration
        """
        nk = bzk_kv.shape[0]
        size = self.kslrf.intrablockcomm.size
        ni = (nk + size - 1) // size
        bzk_ipv = [bzk_kv[i * size:(i + 1) * size] for i in range(ni)]

        # Extract the weight corresponding to the process' own k-point pair
        weight_ip = [weight_k[i * size:(i + 1) * size] for i in range(ni)]
        weight_i = [None] * len(weight_ip)
        krank = self.kslrf.intrablockcomm.rank
        for i, w_p in enumerate(weight_ip):
            if krank in range(len(w_p)):
                weight_i[i] = w_p[krank]

        return bzk_ipv, weight_i


class PWPointIntegrator(Integrator):
    r"""A simple point integrator for the plane wave mode, estimating the
    k-point integral as a simple sum over all k-points of the ground state
    k-point grid:
                                      __
       1     /           ~  2/nspins  \   (2pi)^D
    ‾‾‾‾‾‾‾  |dk (...)_k =  ‾‾‾‾‾‾‾‾  /   ‾‾‾‾‾‾‾ (...)_k
    (2pi)^D  /              (2pi)^D   ‾‾   Nk V0
                                      k

    Using the PWSymmetryAnalyzer, the k-point sum is reduced according to the
    symmetries of the crystal.
    """

    @timer('Get k-point domain')
    def get_kpoint_domain(self):
        """Use the PWSymmetryAnalyzer to define and weight the k-point domain
        based on the ground state k-point grid.

        NB: We could use some more documentation, see XXX below.
        """
        # Generate k-point domain in relative coordinates
        K_gK = self.kslrf.pwsa.group_kpoints()  # What is g? XXX
        bzk_kc = np.array([self.kslrf.gs.kd.bzk_kc[K_K[0]] for
                           K_K in K_gK])  # Why only K=0? XXX
        # Compute actual k-points in absolute reciprocal space coordinates
        bzk_kv = np.dot(bzk_kc, self.kslrf.pd.gd.icell_cv) * 2 * np.pi

        # Get the k-point weights from the symmetry analyzer
        weight_k = [self.kslrf.pwsa.get_kpoint_weight(k_c) for k_c in bzk_kc]

        return bzk_kv, weight_k

    def calculate_bzint_prefactor(self, bzk_kv):
        """Calculate the k-point intregral prefactor A."""
        # The spin prefactor does not naturally belong to the k-point pair
        # integrator. Move to "add_integrand" functionality, when Integrator is
        # made independent of kslrf XXX.
        sfrac = 2 / self.kslrf.gs.nspins

        A = sfrac

        return A


def create_integrator(kslrf):
    """Creator component for the integrator"""
    if kslrf.mode == 'pw':
        if kslrf.kpointintegration is None or \
           kslrf.kpointintegration == 'point integration':
            return PWPointIntegrator(kslrf)

    raise ValueError(kslrf.mode, kslrf.kpointintegration)
