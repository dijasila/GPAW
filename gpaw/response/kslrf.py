import numpy as np
from functools import partial
from time import ctime

from ase.units import Hartree
from ase.utils.timing import timer

from gpaw import extra_parameters
import gpaw.mpi as mpi
from gpaw.blacs import (BlacsGrid, BlacsDescriptor, Redistributor)
from gpaw.utilities.memory import maxrss


class KohnShamLinearResponseFunction:
    """Class calculating linear response functions in the Kohn-Sham system

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
        KSPair : gpaw.response.pair.KohnShamPair instance
            Class for handling pairs of Kohn-Sham orbitals
        PME : gpaw.response.pair.PairMatrixElement instance
            Class for calculating transition matrix elements for pairs of
            Kohn-Sham orbitals
        integrator : gpaw.response.integrators.Integrator instance
            The integrator class is a general class for Brillouin Zone
            integration. The user defined integrand is integrated over k-points
            and a user defined sum over possible band and spin domains is
            carried out.

        Callables
        ---------
        self.calculate(*args, **kwargs) : func
            Runs the calculation, returning the response function.
            Returned format can varry depending on response and mode.
        """

        self.KSPair = KohnShamPair(gs, world=world, txt=txt, timer=timer)  # KohnShamPair object missing XXX
        self.calc = self.KSPair.calc

        self.response = response
        self.mode = mode

        self.bandsummation = bandsummation
        self.nbands = nbands or self.calc.wfs.bd.nbands
        self.nocc1 = self.KSPair.nocc1  # number of completely filled bands
        self.nocc2 = self.KSPair.nocc2  # number of non-empty bands

        self.kpointintegration = kpointintegration
        self.integrator = None  # Mode specific (kpoint) Integrator class
        # Each integrator might support different integration strategies:
        self.integration_kind = None
        # Each integrator might take some extra input kwargs
        self.extraintargs = {}

        self.initialize_distributed_memory(nblocks)
        self.nblocks = nblocks

        # Extract world, timer and filehandle for output
        self.world = self.KSPair.world
        self.timer = self.KSPair.timer
        self.fd = self.KSPair.fd

        # Attributes related to the specific response function
        self.PME = None
    
    def initialize_distributed_memory(self, nblocks):
        """Set up MPI communicators to allow each process to store
        only a fraction (a block) of the response function."""
        if nblocks == 1:
            self.blockcomm = self.world.new_communicator([self.world.rank])
            self.kncomm = self.world
        else:
            assert self.world.size % nblocks == 0, self.world.size
            rank1 = self.world.rank // nblocks * nblocks
            rank2 = rank1 + nblocks
            self.blockcomm = self.world.new_communicator(range(rank1, rank2))
            ranks = range(self.world.rank % nblocks, self.world.size, nblocks)
            self.kncomm = self.world.new_communicator(ranks)

    def calculate(self, spinrot=None, A_x=None):
        return self._calculate(spinrot, A_x)

    @timer('Calculate Kohn-Sham linear response function')
    def _calculate(self, spinrot, A_x):
        """In-place calculation of the response function

        Parameters
        ----------
        spinrot : str
            Select spin rotation.
            Choices: 'uu', 'dd', 'I' (= 'uu' + 'dd'), '-'= and '+'
            All rotations are included for spinrot=None ('I' + '+' + '-').
        A_x : ndarray
            Output array. If None, the output array is created.
        """
        self.spinrot = spinrot
        # Prepare to sum over bands and spins
        n1_t, n2_t, s1_t, s2_t = self.get_band_spin_transitions_domain()

        # Print information about the prepared calculation
        self.print_information(len(n1_t))
        if extra_parameters.get('df_dry_run'):  # Exit after setting up
            print('    Dry run exit', file=self.fd)
            raise SystemExit

        A_x = self.setup_output_array(A_x)

        self.integrator.integrate(kind=self.integration_kind,
                                  bsdomain=(n1_t, n2_t, s1_t, s2_t),
                                  get_integrand=self.get_integrand,
                                  out_x=A_x,
                                  **self.extraintargs)

        return self.post_process(A_x)

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

    def get_integrand(self, *args, **kwargs):
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

        if extra_parameters.get('df_dry_run'):
            from gpaw.mpi import DryRunCommunicator
            size = extra_parameters['df_dry_run']
            world = DryRunCommunicator(size)
        else:
            world = self.world
        wsize = world.size
        knsize = self.kncomm.size
        bsize = self.blockcomm.size

        spinrot = self.spinrot

        p = partial(print, file=self.fd)

        p('%s' % ctime())
        p('Called a response.lrf.KohnShamLinearResponseFunction.calculate()')
        p('using a Kohn-Sham ground state with:')
        p('    Number of spins: %d' % ns)
        p('    Number of bands: %d' % nbands)
        p('    Number of completely occupied states: %d' % nocc)
        p('    Number of partially occupied states: %d' % npocc)
        p('    Number of kpoints: %d' % nk)
        p('    Number of irredicible kpoints: %d' % nik)
        p('')
        p('The response function calculation is performed in parallel with:')
        p('    world.size: %d' % wsize)
        p('    kncomm.size: %d' % knsize)
        p('    blockcomm.size: %d' % bsize)
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
        if spinrot is None or spinrot == 'I':
            s1_S = [0]
            s2_S = [0]
        else:
            raise ValueError(spinrot, nspins)
    else:
        if spinrot is None:
            s1_S = [0, 0, 1, 1]
            s2_S = [0, 1, 0, 1]
        elif spinrot == 'I':
            s1_S = [0, 1]
            s2_S = [0, 1]
        elif spinrot == 'uu':
            s1_S = [0]
            s2_S = [0]
        elif spinrot == 'dd':
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

    def __init__(self, *args, frequencies=None, eta=0.2,
                 ecut=50, gammacentered=False, disable_point_group=True,
                 disable_time_reversal=True, disable_non_symmorphic=True,
                 kpointintegration='point integration', **kwargs):
        """Initialize the plane wave calculator mode.
        In plane wave mode, the linear response function is calculated for a
        given set of frequencies. The spatial part is expanded in plane waves
        for a given momentum transfer q within the first Brillouin Zone.

        Parameters
        ----------
        frequencies : ndarray or None
            Array of frequencies to evaluate the response function at.
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
        """
        # Avoid any mode ambiguity
        if 'mode' in kwargs.keys():
            mode = kwargs.pop('mode')
            assert mode == 'pw'

        KSLRF = KohnShamLinearResponseFunction
        KSLRF.__init__(self, *args, mode='pw',
                       kpointintegration=kpointintegration, **kwargs)

        self.wd = FrequencyDescriptor(np.asarray(frequencies) / Hartree)
        self.omega_w = self.wd.get_data()
        self.eta = eta / Hartree

        self.ecut = None if ecut is None else ecut / Hartree
        self.gammacentered = gammacentered

        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal
        self.disable_non_symmorphic = disable_non_symmorphic

        self.integrator = create_integrator(self.mode, self.kpointintegration)  # Write me XXX

        # Attributes related to specific q, given to self.calculate()
        self.pd = None  # Plane wave descriptor for given momentum transfer q
        self.PWSA = None  # Plane wave symmetry analyzer for given q

    def calculate(self, q_c, spinrot=None, A_x=None):
        """
        Parameters
        ----------
        q_c : list or ndarray
            Momentum transfer.

        Returns
        -------
        pd : Planewave descriptor
            Planewave descriptor for q_c.
        A_wGG : ndarray
            The linear response function.
        """
        # Set up plane wave description with the gived momentum transfer q
        q_c = np.asarray(q_c, dtype=float)
        self.pd = self.get_PWDescriptor(q_c)
        self.PWSA = self.get_PWSymmetryAnalyzer(self.pd)
        self.extraintargs['PWSA'] = self.PWSA

        # In-place calculation
        return self._calculate(spinrot, A_x)

    def get_PWDescriptor(self, q_c):
        """Get the planewave descriptor for a certain momentum transfer q_c."""
        from gpaw.kpt_descriptor import KPointDescriptor
        from gpaw.wavefunctions.pw import PWDescriptor
        
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(self.ecut, self.calc.wfs.gd,
                          complex, qd, gammacentered=self.gammacentered)
        return pd

    def get_PWSymmetryAnalyzer(self, pd):
        from gpaw.response.pair import PWSymmetryAnalyzer as PWSA
        
        return PWSA(self.calc.wfs.kd, pd,
                    timer=self.timer, txt=self.fd,
                    disable_point_group=self.disable_point_group,
                    disable_time_reversal=self.disable_time_reversal,
                    disable_non_symmorphic=self.disable_non_symmorphic)

    def print_information(self, nt):
        """Basic information about the input ground state, parallelization,
        sum over states and calculated response function array."""
        KohnShamLinearResponseFunction.print_information(self, nt)

        q_c = self.pd.kd.bzk_kc[0]
        nw = len(self.omega_w)
        eta = self.eta * Hartree
        ecut = self.ecut * Hartree
        ngmax = self.pd.ngmax
        Asize = nw * self.pd.ngmax**2 * 16. / 1024**2 / self.blockcomm.size

        p = partial(print, file=self.fd)

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

    def setup_output_array(self, nG, A_x=None):
        """Initialize the output array in blocks"""
        # Could use some more documentation XXX
        nw = len(self.omega_w)
        mynG = (nG + self.blockcomm.size - 1) // self.blockcomm.size
        self.Ga = min(self.blockcomm.rank * mynG, nG)
        self.Gb = min(self.Ga + mynG, nG)
        # if self.blockcomm.rank == 0:
        #     assert self.Gb - self.Ga >= 3
        # assert mynG * (self.blockcomm.size - 1) < nG
        if A_x is not None:
            nx = nw * (self.Gb - self.Ga) * nG
            A_wGG = A_x[:nx].reshape((nw, self.Gb - self.Ga, nG))
            A_wGG[:] = 0.0
        else:
            A_wGG = np.zeros((nw, self.Gb - self.Ga, nG), complex)

        return A_wGG

    def post_process(self, A_wGG):
        tmpA_wGG = self.redistribute(A_wGG)  # distribute over frequencies
        self.PWSA.symmetrize_wGG(tmpA_wGG)
        self.redistribute(tmpA_wGG, A_wGG)

        return self.pd, A_wGG

    @timer('redist')
    def redistribute(self, in_wGG, out_x=None):
        """Redistribute array.

        Switch between two kinds of parallel distributions:

        1) parallel over G-vectors (second dimension of in_wGG)
        2) parallel over frequency (first dimension of in_wGG)

        Returns new array using the memory in the 1-d array out_x.
        """

        comm = self.blockcomm

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


# These thing should be moved to integrator XXX
def get_kpoint_pointint_domain(pd, PWSA, calc):
    # Could use documentation XXX
    K_gK = PWSA.group_kpoints()
    bzk_kc = np.array([calc.wfs.kd.bzk_kc[K_K[0]] for
                       K_K in K_gK])

    return bzk_kc


def calculate_kpoint_pointint_prefactor(calc, PWSA, bzk_kv):
    # Could use documentation XXX
    if calc.wfs.kd.refine_info is not None:
        nbzkpts = calc.wfs.kd.refine_info.mhnbzkpts
    else:
        nbzkpts = calc.wfs.kd.nbzkpts
    frac = len(bzk_kv) / nbzkpts
    return (2 * frac * PWSA.how_many_symmetries() /
            (calc.wfs.nspins * (2 * np.pi)**3))
