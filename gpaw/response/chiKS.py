import numpy as np
from functools import partial
from time import ctime

from ase.units import Hartree
from ase.utils.timing import timer

from gpaw import extra_parameters
import gpaw.mpi as mpi
from gpaw.blacs import (BlacsGrid, BlacsDescriptor, Redistributor,
                        DryRunBlacsGrid)
from gpaw.utilities.memory import maxrss
from gpaw.bztools import convex_hull_volume


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

    def calculate(self, spinrot=None):
        """
        Parameters
        ----------
        spinrot : str
            Select spin rotation.
            Choices: 'uu', 'dd', 'I' (= 'uu' + 'dd'), '-'= and '+'
            All rotations are included for spinrot=None ('I' + '+' + '-').
        """
        self.spinrot = spinrot
        # Prepare to sum over bands and spins
        n1_t, n2_t, s1_t, s2_t = self.get_transitions_sum_domain(spinrot)

        # Print information about the prepared calculation
        self.print_information(len(n1_t))
        if extra_parameters.get('df_dry_run'):  # Exit after setting up
            print('    Dry run exit', file=self.fd)
            raise SystemExit

        return self._calculate(n1_t, n2_t, s1_t, s2_t)

    def get_transitions_sum_domain(self, spinrot=None):
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
                                                 spinrot, self.calc.wfs.nspins)

        return get_bandspin_transitions_domain(n1_M, n2_M, s1_S, s2_S)

    def _calculate(self, n1_t, n2_t, s1_t, s2_t):
        raise NotImplementedError('This is a parent class')

    def print_information(self, nt):
        """Basic information about input ground state and parallelization"""
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
        p('The sum over band and spin transitions is perform using:')
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


def get_spin_summation_domain(bandsummation, spin, nspins):
    """Get structure of the sum over spins

    Parameters
    ----------
    bandsummation : str
        Band summation method
    spin : str
        spin transition in chiKS standard
    nspins : int
        number of spin channels in ground state calculation

    Returns
    -------
    spins : list
        list of start spins (s)
    flip : bool
        does the transition flip the spin (s' = 1 - s)?
    """
    _get_spin_sum_domain = create_get_spin_sum_domain(bandsummation)
    spins, flip = _get_spin_sum_domain(spin, nspins)

    if flip:
        assert nspins == 2
    else:
        assert max(spins) < nspins

    return spins, flip


def create_get_spin_sum_domain(bandsummation):
    """Creator component deciding how to carry out spin summation."""
    if bandsummation == 'pairwise':
        return get_pairwise_spin_sum_domain
    elif bandsummation == 'double':
        return get_double_spin_sum_domain
    raise ValueError(bandsummation)


def get_double_spin_sum_domain(spin, nspins):
    """Transitions forward in time"""
    if spin == '00':
        spins = range(nspins)
        flip = False
    elif spin == 'uu':
        spins = [0]
        flip = False
    elif spin == 'dd':
        spins = [1]
        flip = False
    elif spin == '+-':
        spins = [0]
        flip = True
    elif spin == '-+':
        spins = [1]
        flip = True
    else:
        raise ValueError(spin)

    return spins, flip


def get_pairwise_spin_sum_domain(spin, nspins):
    """Transitions forward in time"""
    if spin == '00':
        spins = range(nspins)
        flip = False
    elif spin == 'uu':
        spins = [0]
        flip = False
    elif spin == 'dd':
        spins = [1]
        flip = False
    elif spin == '+-':
        spins = [0, 1]
        flip = True
    elif spin == '-+':
        spins = [0, 1]
        flip = True
    else:
        raise ValueError(spin)

    return spins, flip


class PlaneWaveKSLRF(KohnShamLinearResponseFunction):
    """Class for doing KS-LRF calculations in plane wave mode"""

    def __init__(self, *args, frequencies=None, eta=0.2,
                 ecut=50, gammacentered=False, disable_point_group=True,
                 disable_time_reversal=True, disable_non_symmorphic=True,
                 kpointintegration='point integration', **kwargs):
        """Initialize the plane wave calculator mode.
        In plane wave mode, the linear response function is calculated
        in the frequency domain. The spatial part is expanded in plane waves
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

    def calculate(self, q_c, A_x=None):
        """
        Parameters
        ----------
        q_c : list or ndarray
            Momentum transfer.
        A_x : ndarray
            Output array. If None, the output array is created.

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

        # Print information about the prepared calculation
        self.print_information()  # Print more/less information XXX
        if extra_parameters.get('df_dry_run'):  # Exit after setting up
            print('    Dry run exit', file=self.fd)
            raise SystemExit

        A_wGG = self.setup_output_array(A_x)

        n_M, m_M, s1_M, s2_M = self.get_summation_domain()

        return self._calculate(A_wGG, n_M, m_M, s1_S, s2_S)

    def _calculate(self, A_wGG, n_M, m_M, s1_S, s2_S):
        raise NotImplementedError('This is a parent class')

    def get_PWDescriptor(self, q_c):
        """Get the planewave descriptor for a certain momentum transfer q_c."""
        from gpaw.kpt_descriptor import KPointDescriptor
        from gpaw.wavefunctions.pw import PWDescriptor
        
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(self.ecut, self.calc.wfs.gd,
                          complex, qd, gammacentered=self.gammacentered)
        return pd

    def get_PWSymmetryAnalyzer(self, pd):
        from gpaw.response.integrators import PWSymmetryAnalyzer as PWSA  # Write me XXX
        
        return PWSA(self.calc.wfs.kd, pd,
                    timer=self.timer, txt=self.fd,
                    disable_point_group=self.disable_point_group,
                    disable_time_reversal=self.disable_time_reversal,
                    disable_non_symmorphic=self.disable_non_symmorphic)

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

    def get_summation_domain(self):
        return

    def print_information(self):
        calc = self.calc
        gd = calc.wfs.gd

        if extra_parameters.get('df_dry_run'):
            from gpaw.mpi import DryRunCommunicator
            size = extra_parameters['df_dry_run']
            world = DryRunCommunicator(size)
        else:
            world = self.world

        q_c = self.pd.kd.bzk_kc[0]
        nw = len(self.omega_w)
        ecut = self.ecut * Hartree
        ns = calc.wfs.nspins
        nbands = self.nbands
        nk = calc.wfs.kd.nbzkpts
        nik = calc.wfs.kd.nibzkpts
        ngmax = self.pd.ngmax
        eta = self.eta * Hartree
        wsize = world.size
        knsize = self.kncomm.size
        nocc = self.nocc1
        npocc = self.nocc2
        ngridpoints = gd.N_c[0] * gd.N_c[1] * gd.N_c[2]
        nstat = (ns * npocc + world.size - 1) // world.size
        occsize = nstat * ngridpoints * 16. / 1024**2
        bsize = self.blockcomm.size
        chisize = nw * self.pd.ngmax**2 * 16. / 1024**2 / bsize

        p = partial(print, file=self.fd)

        p('%s' % ctime())
        p('Called response.lrf.PlaneWaveKSLRF.calculate() with')
        p('    q_c: [%f, %f, %f]' % (q_c[0], q_c[1], q_c[2]))
        p('    Number of frequency points: %d' % nw)
        p('    Planewave cutoff: %f' % ecut)
        p('    Number of spins: %d' % ns)
        p('    Number of bands: %d' % nbands)
        p('    Number of kpoints: %d' % nk)
        p('    Number of irredicible kpoints: %d' % nik)
        p('    Number of planewaves: %d' % ngmax)
        p('    Broadening (eta): %f' % eta)
        p('    world.size: %d' % wsize)
        p('    kncomm.size: %d' % knsize)
        p('    blockcomm.size: %d' % bsize)
        p('    Number of completely occupied states: %d' % nocc)
        p('    Number of partially occupied states: %d' % npocc)
        p()
        p('    Memory estimate of potentially large arrays:')
        p('        A_wGG: %f M / cpu' % chisize)
        p('        Occupied states: %f M / cpu' % occsize)
        p('        Memory usage before allocation: %f M / cpu' % (maxrss() /
                                                                  1024**2))
        p()


class chiKS(PlaneWaveKSLRF):
    """Class calculating the four-component Kohn-Sham susceptibility tensor."""

    def __init__(self, *args, **kwargs):
        """Initialize the chiKS object in plane wave mode.
        
        INSERT: Description of the matrix elements XXX
        """
        PlaneWaveKSLRF.__init__(self, *args, **kwargs)

        # INSERT: some attachment of PairDensity object XXX

        # The class is calculating one spin component at a time
        self.spin = None
        # PAW correction might change with spin component and momentum transfer
        self.Q_aGii = None

    def calculate(self, q_c, spin='all', A_x=None):
        """Calculate spin susceptibility in plane wave mode.

        Parameters
        ----------
        q_c : list or ndarray
            Momentum transfer.
        spin : str or int
            What susceptibility should be calculated?
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented
            'all' is an alias for '00', kept for backwards compability
            Likewise 0 or 1, can be used for 'uu' or 'dd'
        A_x : ndarray
            Output array. If None, the output array is created.

        Returns
        -------
        pd : Planewave descriptor
            Planewave descriptor for q_c.
        chi_wGG : ndarray
            The response function.
        """
        # Set up plane wave description with the gived momentum transfer q
        q_c = np.asarray(q_c, dtype=float)
        self.pd = self.get_PWDescriptor(q_c)
        self.PWSA = self.get_PWSymmetryAnalyzer(self.pd)
        self.Q_aGii = self.pairdensity.initialize_paw_corrections(self.pd)  # PairDensity object missing XXX

        # Analyze the requested spin component
        self.spin = get_unique_spin_component(spin)

        # Print information about the set up calculation
        self.print_information()  # Print more/less information XXX
        if extra_parameters.get('df_dry_run'):  # Exit after setting up
            print('    Dry run exit', file=self.fd)
            raise SystemExit

        chi_wGG = self.setup_output_array(A_x)

        n_M, m_M, bzk_kv, spins, flip = self.get_summation_domain()

        return self._pw_calculate_susceptibility(chi_wGG,
                                                 n_M, m_M, bzk_kv, spins, flip)

    @timer('Calculate Kohn-Sham susceptibility')
    def _pw_calculate_susceptibility(self, chi_wGG,
                                     n_M, m_M, bzk_kv, spins, flip):
        """In-place calculation of the KS susceptibility in plane wave mode."""

        prefactor = self._calculate_kpoint_integration_prefactor(bzk_kv)
        chi_wGG /= prefactor

        self.integrator.integrate(kind=self.integration_kind,
                                  banddomain=(n_M, m_M),
                                  ksdomain=(bzk_kv, spins),
                                  integrand=(self._get_matrix_element,
                                             self._get_eigenvalues),
                                  x=self.wd,  # Frequency Descriptor
                                  # Arguments for integrand functions
                                  out_wxx=chi_wGG,  # Output array
                                  **self.extraintargs)

        # The response function is integrated only over the IBZ. The
        # chi calculated above must therefore be extended to include the
        # response from the full BZ. This extension can be performed as a
        # simple post processing of the response function that makes
        # sure that the response function fulfills the symmetries of the little
        # group of q. Due to the specific details of the implementation the chi
        # calculated above is normalized by the number of symmetries (as seen
        # below) and then symmetrized.
        chi_wGG *= prefactor
        tmpchi_wGG = self.redistribute(chi_wGG)  # distribute over frequencies
        self.PWSA.symmetrize_wGG(tmpchi_wGG)
        self.redistribute(tmpchi_wGG, chi_wGG)

        return self.pd, chi_wGG

    def setup_kpoint_integration(self, integrationmode, pd,
                                 disable_point_group,
                                 disable_time_reversal,
                                 disable_non_symmorphic, **extraargs):
        if integrationmode is None:
            self.integrationmode == 'point integration'
        else:
            self.integrationmode = integrationmode

        # The integration domain is reduced to the irreducible zone
        # of the little group of q.
        PWSA = PWSymmetryAnalyzer
        PWSA = PWSA(self.calc.wfs.kd, pd,
                    timer=self.timer, txt=self.fd,
                    disable_point_group=disable_point_group,
                    disable_time_reversal=disable_time_reversal,
                    disable_non_symmorphic=disable_non_symmorphic)

        self._setup_integrator = self.create_setup_integrator()
        self._setup_integrator(**extraargs)

        return PWSA

    def create_setup_integrator(self):
        """Creator component deciding how to set up kpoint integrator.
        The integrator class is a general class for brillouin zone
        integration that can integrate user defined functions over user
        defined domains and sum over bands.
        """
        
        if self.integrationmode == 'point integration':
            print('Using integration method: PointIntegrator',
                  file=self.fd)
            return self.setup_point_integrator

        elif self.integrationmode == 'tetrahedron integration':
            print('Using integration method: TetrahedronIntegrator',
                  file=self.fd)
            return self.setup_tetrahedron_integrator

        raise ValueError(self.integrationmode)

    def setup_point_integrator(self, **unused):
        self._get_kpoint_int_domain = partial(get_kpoint_pointint_domain,
                                              calc=self.calc)
        self.integrator = PointIntegrator(self.calc.wfs.gd.cell_cv,
                                          response=self.response,  # should be removed XXX
                                          comm=self.world,
                                          timer=self.timer,
                                          txt=self.fd,
                                          eshift=self.eshift,
                                          nblocks=self.nblocks)
        self._calculate_kpoint_integration_prefactor =\
            calculate_kpoint_pointint_prefactor

    def setup_tetrahedron_integrator(self, pbc=None, **unused):
        """
        Parameters
        ----------
        pbc : list
            Periodic directions of the system. Defaults to [True, True, True].
        """
        self.setup_pbc(pbc)
        self._get_kpoint_int_domain = partial(get_kpoint_tetrahint_domain,
                                              pbc=self.pbc)
        self.integrator = TetrahedronIntegrator(self.calc.wfs.gd.cell_cv,
                                                response=self.response,  # should be removed XXX
                                                comm=self.world,
                                                timer=self.timer,
                                                eshift=self.eshift,
                                                txt=self.fd,
                                                nblocks=self.nblocks)
        self._calculate_kpoint_integration_prefactor =\
            calculate_kpoint_tetrahint_prefactor

    def setup_pbc(self, pbc):
        if pbc is not None:
            self.pbc = np.array(pbc)
        else:
            self.pbc = np.array([True, True, True])

        if self.pbc is not None and (~self.pbc).any():
            assert np.sum((~self.pbc).astype(int)) == 1, \
                print('Only one non-periodic direction supported atm.')
            print('Nonperiodic BC\'s: ', (~self.pbc),
                  file=self.fd)

    def get_summation_domain(self):
        """Find the relevant (n, k, s) -> (m, k + q, s') domain to sum over"""
        n_M, m_M = get_band_summation_domain(self.bandsummation, self.nbands,
                                             nocc1=self.nocc1,
                                             nocc2=self.nocc2)
        spins, flip = get_spin_summation_domain(self.bandsummation, self.spin,
                                                self.calc.wfs.nspins)

        bzk_kv = self.get_kpoint_integration_domain(self.pd)

        return n_M, m_M, bzk_kv, spins, flip

    @timer('Get kpoint integration domain')
    def get_kpoint_integration_domain(self):
        bzk_kc = self._get_kpoint_int_domain(self.pd, self.PWSA)
        bzk_kv = np.dot(bzk_kc, self.pd.gd.icell_cv) * 2 * np.pi
        return bzk_kv

    def _get_kpoint_int_domain(pd):
        raise NotImplementedError('Please set up integrator before calling')

    @timer('Get pair density')
    def get_pw_pair_density(self, k_v, s, n_M, m_M, block=True):
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

    @timer('dist freq')
    def distribute_frequencies(self, chi0_wGG):
        """Distribute frequencies to all cores."""

        world = self.world
        comm = self.blockcomm

        if world.size == 1:
            return chi0_wGG

        nw = len(self.omega_w)
        nG = chi0_wGG.shape[2]
        mynw = (nw + world.size - 1) // world.size
        mynG = (nG + comm.size - 1) // comm.size

        wa = min(world.rank * mynw, nw)
        wb = min(wa + mynw, nw)

        if self.blockcomm.size == 1:
            return chi0_wGG[wa:wb].copy()

        if self.kncomm.rank == 0:
            bg1 = BlacsGrid(comm, 1, comm.size)
            in_wGG = chi0_wGG.reshape((nw, -1))
        else:
            bg1 = DryRunBlacsGrid(mpi.serial_comm, 1, 1)
            in_wGG = np.zeros((0, 0), complex)
        md1 = BlacsDescriptor(bg1, nw, nG**2, nw, mynG * nG)

        bg2 = BlacsGrid(world, world.size, 1)
        md2 = BlacsDescriptor(bg2, nw, nG**2, mynw, nG**2)

        r = Redistributor(world, md1, md2)
        shape = (wb - wa, nG, nG)
        out_wGG = np.empty(shape, complex)
        r.redistribute(in_wGG, out_wGG.reshape((wb - wa, nG**2)))

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


def get_unique_spin_component(spin):
    """Convert all supported input to chiKS standard."""
    if isinstance(spin, str):
        if spin in ['00', 'uu', 'dd', '+-', '-+']:
            return spin
        elif spin == 'all':
            return '00'
    elif isinstance(spin, int):
        if spin == 0:
            return 'uu'
        elif spin == 1:
            return 'dd'
    raise ValueError(spin)


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


def get_kpoint_tetrahint_domain(pd, PWSA, pbc):
    # Could use documentation XXX
    bzk_kc = PWSA.get_reduced_kd(pbc_c=pbc).bzk_kc
    if (~pbc).any():
        bzk_kc = np.append(bzk_kc,
                           bzk_kc + (~pbc).astype(int),
                           axis=0)

    return bzk_kc


def calculate_kpoint_tetrahint_prefactor(calc, PWSA, bzk_kv):
    """If there are non-periodic directions it is possible that the
    integration domain is not compatible with the symmetry operations
    which essentially means that too large domains will be
    integrated. We normalize by vol(BZ) / vol(domain) to make
    sure that to fix this.
    """
    vol = abs(np.linalg.det(calc.wfs.gd.cell_cv))
    domainvol = convex_hull_volume(bzk_kv) * PWSA.how_many_symmetries()
    bzvol = (2 * np.pi)**3 / vol
    frac = bzvol / domainvol
    return (2 * frac * PWSA.how_many_symmetries() /
            (calc.wfs.nspins * (2 * np.pi)**3))

