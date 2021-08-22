import numpy as np
from functools import partial
from time import ctime

from ase.units import Hartree
from gpaw.utilities import convert_string_to_fd
from ase.utils.timing import Timer, timer

import gpaw
import gpaw.mpi as mpi
from gpaw.blacs import (BlacsGrid, BlacsDescriptor, Redistributor)
from gpaw.utilities.memory import maxrss
from gpaw.utilities.progressbar import ProgressBar
from gpaw.response.kspair import KohnShamPair, get_calc


class KohnShamLinearResponseFunction:
    r"""Class calculating linear response functions in the Kohn-Sham system

    Any linear response function can be calculated as a sum over transitions
    between the ground state and excited energy eigenstates.

    In the Kohn-Sham system this approach is particularly simple, as only
    excited states, for which a single electron has been moved from an occupied
    single-particle Kohn-Sham orbital to an unoccupied one, contribute.

    Resultantly, any linear response function in the Kohn-Sham system can be
    written as a sum over transitions between pairs of occupied and unoccupied
    Kohn-Sham orbitals.

    Currently, only collinear Kohn-Sham systems are supported. That is, all
    transitions can be written in terms of band indexes, k-points and spins:

    T (composit transition index): (n, k, s) -> (n', k', s')

    The sum over transitions is an integral over k-points in the 1st Brillouin
    Zone and a sum over all bands and spins. Sums over bands and spins can be
    handled together:

    t (composit transition index): (n, s) -> (n', s')

    __               __   __               __
    \      //        \    \      //        \
    /   =  ||dk dk'  /    /   =  ||dk dk'  /
    ‾‾     //        ‾‾   ‾‾     //        ‾‾
    T               n,n' s,s'              t
    """

    def __init__(self, gs, response=None, mode=None,
                 bandsummation='pairwise', nbands=None, kpointintegration=None,
                 world=mpi.world, nblocks=1, txt='-', timer=None):
        """Construct the KSLRF object

        Parameters
        ----------
        gs : str
            The groundstate calculation file that the linear response
            calculation is based on.
        response : str
            Type of response function.
            Currently, only susceptibilities are supported.
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
        world : obj
            MPI communicator.
        nblocks : int
            Divide the response function storage into nblocks. Useful when the
            response function is large and memory requirements are restrictive.
        txt : str
            Output file.
        timer : func
            gpaw.utilities.timing.timer wrapper instance

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
        # Output .txt filehandle
        self.fd = convert_string_to_fd(txt, world)
        self.cfd = self.fd
        print('Initializing KohnShamLinearResponseFunction', file=self.fd)

        # Communicators for parallelization
        self.world = world
        self.interblockcomm = None
        self.intrablockcomm = None
        self.initialize_communicators(nblocks)
        self.nblocks = self.interblockcomm.size

        # Timer
        self.timer = timer or Timer()

        # Load ground state calculation
        self.calc = get_calc(gs, fd=self.fd, timer=self.timer)

        # The KohnShamPair class handles data extraction from ground state
        self.kspair = KohnShamPair(self.calc, world=world,
                                   # Let each process handle slow steps only
                                   # for a fraction of all transitions.
                                   # t-transitions are distributed through
                                   # interblockcomm, k-points through
                                   # intrablockcomm.
                                   transitionblockscomm=self.interblockcomm,
                                   kptblockcomm=self.intrablockcomm,
                                   txt=self.fd, timer=self.timer)

        self.response = response
        self.mode = mode

        self.bandsummation = bandsummation
        self.nbands = nbands or self.calc.wfs.bd.nbands
        assert self.nbands <= self.calc.wfs.bd.nbands
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
        interblockcomm : gpaw.mpi.Communicator
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
        world = self.world
        if nblocks == 1:
            self.interblockcomm = world.new_communicator([world.rank])
            self.intrablockcomm = world
        else:
            assert world.size % nblocks == 0, world.size
            rank1 = world.rank // nblocks * nblocks
            rank2 = rank1 + nblocks
            self.interblockcomm = world.new_communicator(range(rank1, rank2))
            ranks = range(world.rank % nblocks, world.size, nblocks)
            self.intrablockcomm = world.new_communicator(ranks)
        print('Number of blocks:', nblocks, file=self.fd)

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
            print('    Dry run exit', file=self.fd)
            raise SystemExit

        A_x = self.setup_output_array(A_x)

        self.integrator.integrate(n1_t, n2_t, s1_t, s2_t,
                                  out_x=A_x, **self.extraintargs)

        # Different calculation modes might want the response function output
        # in different formats
        out = self.post_process(A_x)

        print('', file=self.cfd)

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
                                                 self.calc.wfs.nspins)

        return transitions_in_composite_index(n1_M, n2_M, s1_S, s2_S)

    def setup_output_array(self, A_x):
        raise NotImplementedError('Output array depends on mode')

    def get_ks_kpoint_pairs(self, k_pv, *args, **kwargs):
        raise NotImplementedError('Integrated pairs of states depend on'
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
        ns = self.calc.wfs.nspins
        nbands = self.nbands
        nocc = self.nocc1
        npocc = self.nocc2
        nk = self.calc.wfs.kd.nbzkpts
        nik = self.calc.wfs.kd.nibzkpts

        if gpaw.dry_run:
            from gpaw.mpi import DryRunCommunicator
            size = gpaw.dry_run
            world = DryRunCommunicator(size)
        else:
            world = self.world
        wsize = world.size
        knsize = self.intrablockcomm.size
        bsize = self.interblockcomm.size

        spinrot = self.spinrot

        p = partial(print, file=self.cfd)

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
        p('    interblockcomm.size: %d' % bsize)
        p('')
        p('The sum over band and spin transitions is performed using:')
        p('    Spin rotation: %s' % spinrot)
        p('    Total number of composite band and spin transitions: %d' % nt)
        p('')


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
                 disable_point_group=True, disable_time_reversal=True,
                 disable_non_symmorphic=True, bundle_integrals=True,
                 kpointintegration='point integration', bundle_kptpairs=False,
                 **kwargs):
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
        bundle_kptpairs : bool
            Extract the k-point pairs simultaneously, so no process has to wait
            for the others in the middle of the response function integration.
            [Only relevant in the case where ground state is distributed]
            Can be switched of, if this step forces calculations out of memory.
        """
        from gpaw.mpi import SerialCommunicator

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

        self.bundle_integrals = bundle_integrals

        # Bundle kptpairs if specified and using a distributed ground state
        cworld = self.calc.world
        if isinstance(cworld, SerialCommunicator) or cworld.size == 1:
            bundle_kptpairs = False
        self.bundle_kptpairs = bundle_kptpairs

        # Attributes related to specific q, given to self.calculate()
        self.pd = None  # Plane wave descriptor for given momentum transfer q
        self.pwsa = None  # Plane wave symmetry analyzer for given q
        self.wd = None  # Frequency descriptor for the given frequencies
        self.omega_w = None  # Frequencies in code units

    @timer('Calculate Kohn-Sham linear response function in plane wave mode')
    def calculate(self, q_c, frequencies, spinrot=None, A_x=None):
        """
        Parameters
        ----------
        q_c : list or ndarray or PWDescriptor
            Momentum transfer (and possibly plane wave basis)
        frequencies : ndarray or FrequencyDescriptor
            Array of frequencies to evaluate the response function at or
            descriptor of those frequencies.

        Returns
        -------
        pd : Planewave descriptor
            Planewave descriptor for q_c.
        A_wGG : ndarray
            The linear response function.
        """
        # Set up plane wave description with the gived momentum transfer q
        self.pd = self.get_PWDescriptor(q_c)
        self.pwsa = self.get_PWSymmetryAnalyzer(self.pd)

        # Set up frequency descriptor for the given frequencies
        self.wd = self.get_FreqDescriptor(frequencies)
        self.omega_w = self.wd.get_data()

        # In-place calculation
        return self._calculate(spinrot, A_x)

    def get_PWDescriptor(self, q_c):
        """Get the planewave descriptor for a certain momentum transfer q_c."""
        from gpaw.pw.descriptor import PWDescriptor
        if isinstance(q_c, PWDescriptor):
            return q_c
        else:
            from gpaw.kpt_descriptor import KPointDescriptor
            q_c = np.asarray(q_c, dtype=float)
            qd = KPointDescriptor([q_c])
            pd = PWDescriptor(self.ecut, self.calc.wfs.gd,
                              complex, qd, gammacentered=self.gammacentered)
            return pd

    @timer('Get PW symmetry analyser')
    def get_PWSymmetryAnalyzer(self, pd):
        from gpaw.response.pair import PWSymmetryAnalyzer as PWSA

        return PWSA(self.calc.wfs.kd, pd,
                    timer=self.timer, txt=self.fd,
                    disable_point_group=self.disable_point_group,
                    disable_time_reversal=self.disable_time_reversal,
                    disable_non_symmorphic=self.disable_non_symmorphic)

    def get_FreqDescriptor(self, frequencies):
        """Get the frequency descriptor for a certain input frequencies."""
        if isinstance(frequencies, FrequencyDescriptor):
            return frequencies
        else:
            return FrequencyDescriptor(np.asarray(frequencies) / Hartree)

    def print_information(self, nt):
        """Basic information about the input ground state, parallelization,
        sum over states and calculated response function array."""
        KohnShamLinearResponseFunction.print_information(self, nt)

        pd = self.pd
        q_c = pd.kd.bzk_kc[0]
        nw = len(self.omega_w)
        eta = self.eta * Hartree
        ecut = self.ecut * Hartree
        ngmax = pd.ngmax
        Asize = nw * pd.ngmax**2 * 16. / 1024**2 / self.interblockcomm.size

        p = partial(print, file=self.cfd)

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
        p('')

    def setup_output_array(self, A_x=None):
        """Initialize the output array in blocks"""
        # Could use some more documentation XXX
        nG = self.pd.ngmax
        nw = len(self.omega_w)
        mynG = (nG + self.interblockcomm.size - 1) // self.interblockcomm.size
        self.Ga = min(self.interblockcomm.rank * mynG, nG)
        self.Gb = min(self.Ga + mynG, nG)
        # if self.interblockcomm.rank == 0:
        #     assert self.Gb - self.Ga >= 3
        # assert mynG * (self.interblockcomm.size - 1) < nG
        if self.bundle_integrals:
            # Setup A_GwG
            if A_x is not None:
                nx = nw * (self.Gb - self.Ga) * nG
                A_GwG = A_x[:nx].reshape((nG, nw, self.Gb - self.Ga))
                A_GwG[:] = 0.0
            else:
                A_GwG = np.zeros((nG, nw, self.Gb - self.Ga), complex)

            return A_GwG
        else:
            # Setup A_wGG
            if A_x is not None:
                nx = nw * (self.Gb - self.Ga) * nG
                A_wGG = A_x[:nx].reshape((nw, self.Gb - self.Ga, nG))
                A_wGG[:] = 0.0
            else:
                A_wGG = np.zeros((nw, self.Gb - self.Ga, nG), complex)

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

        tmpA_wGG = self.redistribute(A_wGG)  # distribute over frequencies
        with self.timer('Symmetrizing Kohn-Sham linear response function'):
            self.pwsa.symmetrize_wGG(tmpA_wGG)
        self.redistribute(tmpA_wGG, A_wGG)

        return self.pd, A_wGG

    @timer('Redistribute memory')
    def redistribute(self, in_wGG, out_x=None):
        """Redistribute array.

        Switch between two kinds of parallel distributions:

        1) parallel over G-vectors (second dimension of in_wGG)
        2) parallel over frequency (first dimension of in_wGG)

        Returns new array using the memory in the 1-d array out_x.
        """

        comm = self.interblockcomm

        if comm.size == 1:
            return in_wGG

        nw = len(self.omega_w)
        nG = in_wGG.shape[2]
        mynw = (nw + comm.size - 1) // comm.size
        mynG = (nG + comm.size - 1) // comm.size

        bg1 = BlacsGrid(comm, comm.size, 1)
        bg2 = BlacsGrid(comm, 1, comm.size)
        md1 = BlacsDescriptor(bg1, nw, nG**2, mynw, nG**2)
        md2 = BlacsDescriptor(bg2, nw, nG**2, nw, mynG * nG)

        if len(in_wGG) == nw:
            mdin = md2
            mdout = md1
        else:
            mdin = md1
            mdout = md2

        r = Redistributor(comm, mdin, mdout)

        outshape = (mdout.shape[0], mdout.shape[1] // nG, nG)
        if out_x is None:
            out_wGG = np.empty(outshape, complex)
        else:
            out_wGG = out_x[:np.product(outshape)].reshape(outshape)

        r.redistribute(in_wGG.reshape(mdin.shape),
                       out_wGG.reshape(mdout.shape))

        return out_wGG


class FrequencyDescriptor:
    """Describes a one-dimensional array of frequencies."""

    def __init__(self, data_x):
        self.data_x = np.array(np.sort(data_x))
        self._data_len = len(data_x)

    def __len__(self):
        return self._data_len

    def get_data(self):
        return self.data_x


class Integrator:
    """Baseclass for integrating over k-points in the first Brillouin Zone
    and summing over bands and spin.
    """
    def __init__(self, kslrf):
        """
        Parameters
        ----------
        kslrf : KohnShamLinearResponseFunction instance
        """
        self.kslrf = kslrf
        self.timer = self.kslrf.timer

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
        bzk_ipv = np.array([bzk_kv[i * size:(i + 1) * size]
                            for i in range(ni)])

        # Extract the weight corresponding to the process' own k-point pair
        weight_ip = np.array([weight_k[i * size:(i + 1) * size]
                              for i in range(ni)])
        weight_i = [None] * len(weight_ip)
        krank = self.kslrf.intrablockcomm.rank
        for i, w_p in enumerate(weight_ip):
            if krank in range(len(w_p)):
                weight_i[i] = w_p[krank]

        return bzk_ipv, weight_i

    @timer('Integrate response function')
    def integrate(self, n1_t, n2_t, s1_t, s2_t,
                  out_x=None, **kwargs):
        if out_x is None:
            raise NotImplementedError

        bzk_kv, weight_k = self.get_kpoint_domain()
        prefactor = self.calculate_bzint_prefactor(bzk_kv)
        out_x /= prefactor
        self._integrate(bzk_kv, weight_k,
                        n1_t, n2_t, s1_t, s2_t, out_x, **kwargs)
        out_x *= prefactor

        return out_x

    def get_kpoint_domain(self):
        raise NotImplementedError('Domain depends on integration method')

    def calculate_bzint_prefactor(self, bzk_kv):
        raise NotImplementedError('Prefactor depends on integration method')

    def _integrate(self, bzk_kv, weight_k,
                   n1_t, n2_t, s1_t, s2_t, out_x, **kwargs):
        raise NotImplementedError('Integration method is defined by subclass')


class PWPointIntegrator(Integrator):
    """A simple point integrator for the plane wave mode."""

    @timer('Get k-point domain')
    def get_kpoint_domain(self):
        # Could use some more documentation XXX
        K_gK = self.kslrf.pwsa.group_kpoints()
        bzk_kc = np.array([self.kslrf.calc.wfs.kd.bzk_kc[K_K[0]] for
                           K_K in K_gK])

        bzk_kv = np.dot(bzk_kc, self.kslrf.pd.gd.icell_cv) * 2 * np.pi

        nsym = self.kslrf.pwsa.how_many_symmetries()
        weight_k = [self.kslrf.pwsa.get_kpoint_weight(k_c) / nsym
                    for k_c in bzk_kc]

        return bzk_kv, weight_k

    def calculate_bzint_prefactor(self, bzk_kv):
        # Could use some more documentation XXX
        if self.kslrf.calc.wfs.kd.refine_info is not None:
            nbzkpts = self.kslrf.calc.wfs.kd.refine_info.mhnbzkpts
        else:
            nbzkpts = self.kslrf.calc.wfs.kd.nbzkpts
        frac = len(bzk_kv) / nbzkpts

        return (2 * frac * self.kslrf.pwsa.how_many_symmetries() /
                (self.kslrf.calc.wfs.nspins * (2 * np.pi)**3))

    def _integrate(self, bzk_kv, weight_k,
                   n1_t, n2_t, s1_t, s2_t, out_x, **kwargs):
        """Do a simple sum over k-points in the first Brillouin Zone,
        adding the integrand (summed over bands and spin) for each k-point."""

        nk = bzk_kv.shape[0]
        vol = abs(np.linalg.det(self.kslrf.calc.wfs.gd.cell_cv))

        kpointvol = (2 * np.pi)**3 / vol / nk
        out_x /= kpointvol

        # Initialize pme
        print('----------',
              file=self.kslrf.cfd)
        print('Initializing PairMatrixElement',
              file=self.kslrf.cfd, flush=True)
        self.kslrf.initialize_pme()

        # Slice domain
        bzk_ipv, weight_i = self.slice_kpoint_domain(bzk_kv, weight_k)

        # Perform sum over k-points
        tmp_x = np.zeros_like(out_x)
        pb = ProgressBar(self.kslrf.cfd)
        if self.kslrf.bundle_kptpairs:
            # Extract all the process' kptpairs at once, then do integration
            print('\nExtracting Kohn-Sham k-point pairs',
                  file=self.kslrf.cfd, flush=True)
            kskptpair_i = []
            for k_pv in bzk_ipv:
                kskptpair_i.append(self.kslrf.get_ks_kpoint_pairs(k_pv,
                                                                  n1_t, n2_t,
                                                                  s1_t, s2_t))

            print('\nIntegrating response function',
                  file=self.kslrf.cfd, flush=True)
            # Carry out sum
            for i, kskptpair in pb.enumerate(kskptpair_i):
                if kskptpair is not None:
                    weight = weight_i[i]
                    assert weight is not None
                    self.kslrf.calculate_pme(kskptpair)
                    self.kslrf.add_integrand(kskptpair, weight,
                                             tmp_x, **kwargs)
        else:
            # Each process will do its own k-points, but it has to follow the
            # others, as it may have to send them information about its
            # partition of the ground state
            print('\nIntegrating response function',
                  file=self.kslrf.cfd, flush=True)
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
        with self.timer('Sum over distributed k-points'):
            self.kslrf.intrablockcomm.sum(tmp_x)

        out_x += tmp_x
        out_x *= kpointvol


def create_integrator(kslrf):
    """Creator component for the integrator"""
    if kslrf.mode == 'pw':
        if kslrf.kpointintegration is None or \
           kslrf.kpointintegration == 'point integration':
            return PWPointIntegrator(kslrf)

    raise ValueError(kslrf.mode, kslrf.kpointintegration)
