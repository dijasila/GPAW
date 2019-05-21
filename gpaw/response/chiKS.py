import numbers
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
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.bztools import convex_hull_volume
from gpaw.response.pair import PWSymmetryAnalyzer
from gpaw.response.pair import create_PairMatrixElement
from gpaw.response.integrators import PointIntegrator, TetrahedronIntegrator


class ChiKS:
    """For calculating linear response functions in the Kohn-Sham system"""

    def __init__(self, gs, response='susceptibility', mode='pw',
                 world=mpi.world, txt='-', timer=None, **kwargs):
        """Construct the ChiKS object

        Currently only collinear Kohn-Sham systems are supported.

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
        world : obj
            MPI communicator.
        txt : str
            Output file.
        timer : func
            gpaw.utilities.timing.timer wrapper instance

        Note: Any parameter you want to pass to PairMatrixElement (ex.
              gate_voltage to PairDensity) or to set up a specific response
              or calculation mode (ex. frequencies to setup_pw_calculator),
              you can pass as a kwarg.
              

        Callables
        ---------
        self.calculate(*args, **kwargs) : func
            Runs the ChiKS calculation, returning the response function.
            Returned format can varry depending on response and mode.
        """

        self.response = response
        self.mode = mode
        self.world = world

        # Initialize the PairMatrixElement object
        PME = create_PairMatrixElement(response)
        self.pair = PME(gs, world=self.world, txt=txt, timer=timer, **kwargs)

        # Extract ground state calculator, timer and filehandle for output
        calc = self.pair.calc
        self.calc = calc
        self.timer = self.pair.timer
        self.fd = self.pair.fd

        self.setup_calculator(**kwargs)

    def setup_calculator(self, **kwargs):
        (self._setup_calculator,
         self.calculate, self._calculate) = self.create_calculator()
        self._setup_calculator(**kwargs)

    def create_calculator(self):
        """Creator component deciding how and what to calculate."""
        if self.mode == 'pw' and self.response == 'susceptibility':
            return (self.setup_pw_calculator,
                    self.pw_calculate_susceptibility,
                    self._pw_calculate_susceptibility)
        else:
            raise ValueError(self.response, self.mode)

    def setup_pw_calculator(self, frequencies=None, eta=0.2,
                            ecut=50, gammacentered=False,
                            nbands=None, bandsummation='pairwise',
                            nblocks=1, **extraargs):
        """Initialize the plane wave calculator mode.

        Parameters
        ----------
        frequencies : ndarray or None
            Array of frequencies to evaluate the response function at.
            If None, a nonlinear frequency grid is used:
            domega0, omega2, omegamax : float
                Input parameters for nonlinear frequency grid.
                Passed through kwargs
        eta : float
            Energy broadening of spectra.
        ecut : float
            Energy cutoff for the plane wave representation.
        gammacentered : bool
            Center the grid of plane waves around the gamma point or q-vector.
        nbands : int
            Maximum band index to include.
        bandsummation : str
            Band summation for transition matrix elements.
            'pairwise': sum over pairs of bands (uses spin-conserving
            time-reversal symmetry to reduce number of matrix elements).
            'double': double sum over band indices.
        nblocks : int
            Divide the response function storage into nblocks. Useful when the
            response function is large and memory requirements are restrictive.

        extraargs : dict
            Extra arguments, that are not used for all spin susceptibilities
        """

        self.nbands = nbands or self.calc.wfs.bd.nbands

        self.frequencies = frequencies
        self.wd = self.get_pw_frequencies(frequencies, **extraargs)
        self.omega_w = self.wd.get_data()
        self.eta = eta / Hartree
        
        self.ecut = None if ecut is None else ecut / Hartree
        self.gammacentered = gammacentered
        
        self.bandsummation = bandsummation
        
        self.setup_memory_distribution(nblocks)
        self.nblocks = nblocks

        self.nocc1 = self.pair.nocc1  # number of completely filled bands
        self.nocc2 = self.pair.nocc2  # number of non-empty bands

        self.extraintargs = {}  # Extra arguments for the integration method.

        # Properties of plane wave susceptibility, given to self.calculate()
        self.pd = None  # Plane wave descriptor for given momentum transfer q
        self.spin = None  # Given spin rotation matrix
        self.PWSA = None  # Plane wave symmetry analyzer for given q
        self.Q_aGii = None  # PAW corrections for given q

        self.extraargs = extraargs

    def get_pw_frequencies(self, frequencies, **kwargs):
        """Frequencies to evaluate response function at in GPAW units"""
        if frequencies is None:
            return get_nonlinear_frequency_grid(self.calc, self.nbands,
                                                fd=self.fd, **kwargs)
        else:
            return ArrayDescriptor(np.asarray(frequencies) / Hartree)

    def setup_memory_distribution(self, nblocks):
        """Set up distribution of memory"""
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

    def pw_calculate_susceptibility(self, q_c, spin='all', A_x=None):
        """Calculate spin susceptibility in plane wave mode.

        Parameters
        ----------
        q_c : list or ndarray
            Momentum vector.
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
        q_c = np.asarray(q_c, dtype=float)
        self.pd = get_PWDescriptor(self.ecut, self.calc.wfs.gd, q_c,
                                   gammacentered=self.gammacentered)

        self.spin = get_unique_spin_str(spin)

        # Set up stuff that depends on q_c and spin
        self.PWSA = self.pw_setup_susceptibility(self.pd, self.spin)
        # Reset PAW correction
        self.Q_aGii = self.pair.initialize_paw_corrections(self.pd)

        # Print information about the set up calculation
        self.print_pw_chi()
        if extra_parameters.get('df_dry_run'):  # Exit after setting up
            print('    Dry run exit', file=self.fd)
            raise SystemExit

        chi_wGG = self.setup_chi_wGG(A_x)  # set up output array

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

    def pw_setup_susceptibility(self, pd, spin):
        """Set up chiKS to calculate a spicific spin susceptibility."""
        assert spin in ['00', 'uu', 'dd', '+-', '-+']
        self._pw_setup_sus = self.create_pw_setup_susceptibility(spin)
        self._pw_setup_sus(pd, **self.extraargs)

    def create_pw_setup_susceptibility(self, spin):
        """Creator component deciding how to set up spin susceptibility."""
        if spin == '00':
            return self.pw_setup_chi00
        else:
            return self.pw_setup_chimunu
    
    def pw_setup_chi00(self, pd, hilbert=True, timeordered=False,
                       intraband=True, integrationmode=None,
                       disable_point_group=False,
                       disable_time_reversal=False,
                       disable_non_symmorphic=True,
                       rate=0.0, eshift=0.0, **extraargs):
        """Set up additional parameters for plane wave chi00 calculation
        The chi00 calculation could use further refactorization XXX

        Parameters
        ----------
        hilbert : bool
            Switch for hilbert transform. If True, the full density
            response is determined from a hilbert transform of its spectral
            function. This is typically much faster, but does not work for
            imaginary frequencies.
        timeordered : bool
            Calculate the time ordered density response function.
            In this case the hilbert transform cannot be used.
        intraband : bool
            Switch for including the intraband contribution to the density
            response function.
        disable_point_group : bool
            Do not use the point group symmetry operators.
        disable_time_reversal : bool
            Do not use time reversal symmetry.
        disable_non_symmorphic : bool
            Do no use non symmorphic symmetry operators.
        integrationmode : str
            Integrator for the kpoint integration.
        rate : float
            Unknown parameter for intraband calculation in optical limit
        eshift : float
            Shift unoccupied bands
        """

        self.hilbert = hilbert
        self.timeordered = bool(timeordered)
            
        if self.eta == 0.0:  # Refacor using factory method XXX
            # If eta is 0 then we must be working with imaginary frequencies.
            # In this case chi is hermitian and it is therefore possible to
            # reduce the computational costs by a only computing half of the
            # response function.
            assert not self.hilbert
            assert not self.timeordered
            assert not self.omega_w.real.any()
            
            self.integration_kind = 'hermitian response function'
        elif self.hilbert:
            # The spectral function integrator assumes that the form of the
            # integrand is a function (a matrix element) multiplied by
            # a delta function and should return a function of at user defined
            # x's (frequencies). Thus the integrand is tuple of two functions
            # and takes an additional argument (x).
            assert self.frequencies is None
            self.integration_kind = 'spectral function'
        else:
            self.extraintargs['eta'] = self.eta
            self.extraintargs['timeordered'] = self.timeordered
            self.integration_kind = 'response function'

        self.include_intraband = intraband
        
        PWSA = self.setup_kpoint_integration(integrationmode, pd,
                                             disable_point_group,
                                             disable_time_reversal,
                                             disable_non_symmorphic,
                                             **extraargs)
        
        if rate == 'eta':
            self.rate = self.eta
        else:
            self.rate = rate / Hartree

        self.eshift = eshift / Hartree

        return PWSA

    def pw_setup_chimunu(self, pd, **unused):
        """Disable stuff, that has been developed for chi00 only"""

        self.extraintargs['eta'] = self.eta
        self.extraintargs['timeordered'] = False
        self.integration_kind = 'response function'

        PWSA = self.setup_kpoint_integration('point integration', pd,
                                             True, True, True,  # disable sym
                                             **unused)
        self.eshift = 0.0

        return PWSA

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
        self.integrator = PointIntegrator(self.pair.calc.wfs.gd.cell_cv,
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
        self.integrator = TetrahedronIntegrator(self.pair.calc.wfs.gd.cell_cv,
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

    def setup_chi_wGG(self, nG, A_x=None):
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
            chi_wGG = A_x[:nx].reshape((nw, self.Gb - self.Ga, nG))
            chi_wGG[:] = 0.0
        else:
            chi_wGG = np.zeros((nw, self.Gb - self.Ga, nG), complex)

        return chi_wGG

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
            K1 = self.pair.find_kpoint(k_c)
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
            self.Q_aGii = self.pair.initialize_paw_corrections(pd)

        kptpair = self.pair.get_kpoint_pair(pd, s, k_c, block=block)

        n_MG = self.pair.get_pair_density(pd, kptpair, n_M, m_M,
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
    
    def print_pw_chi(self):
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
        p('Called response.chi.calculate with')
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
        p('        chi_wGG: %f M / cpu' % chisize)
        p('        Occupied states: %f M / cpu' % occsize)
        p('        Memory usage before allocation: %f M / cpu' % (maxrss() /
                                                                  1024**2))
        p()


class ArrayDescriptor:
    """Describes a single dimensional array."""

    def __init__(self, data_x):
        self.data_x = np.array(np.sort(data_x))
        self._data_len = len(data_x)

    def __len__(self):
        return self._data_len

    def get_data(self):
        return self.data_x

    def get_closest_index(self, scalars_w):
        """Get closest index.

        Get closest index approximating scalars from below."""
        diff_xw = self.data_x[:, np.newaxis] - scalars_w[np.newaxis]
        return np.argmin(diff_xw, axis=0)

    def get_index_range(self, lim1_m, lim2_m):
        """Get index range. """

        i0_m = np.zeros(len(lim1_m), int)
        i1_m = np.zeros(len(lim2_m), int)

        for m, (lim1, lim2) in enumerate(zip(lim1_m, lim2_m)):
            i_x = np.logical_and(lim1 <= self.data_x,
                                 lim2 >= self.data_x)
            if i_x.any():
                inds = np.argwhere(i_x)
                i0_m[m] = inds.min()
                i1_m[m] = inds.max() + 1

        return i0_m, i1_m


class FrequencyDescriptor(ArrayDescriptor):

    def __init__(self, domega0, omega2, omegamax):
        beta = (2**0.5 - 1) * domega0 / omega2
        wmax = int(omegamax / (domega0 + beta * omegamax))
        w = np.arange(wmax + 2)  # + 2 is for buffer
        omega_w = w * domega0 / (1 - beta * w)

        ArrayDescriptor.__init__(self, omega_w)

        self.domega0 = domega0
        self.omega2 = omega2
        self.omegamax = omegamax
        self.omegamin = 0

        self.beta = beta
        self.wmax = wmax
        self.omega_w = omega_w
        self.wmax = wmax
        self.nw = len(omega_w)

    def get_closest_index(self, o_m):
        beta = self.beta
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        if isinstance(w_m, np.ndarray):
            w_m[w_m >= self.wmax] = self.wmax - 1
        elif isinstance(w_m, numbers.Integral):
            if w_m >= self.wmax:
                w_m = self.wmax - 1
        else:
            raise TypeError
        return w_m

    def get_index_range(self, omega1_m, omega2_m):
        omega1_m = omega1_m.copy()
        omega2_m = omega2_m.copy()
        omega1_m[omega1_m < 0] = 0
        omega2_m[omega2_m < 0] = 0
        w1_m = self.get_closest_index(omega1_m)
        w2_m = self.get_closest_index(omega2_m)
        o1_m = self.omega_w[w1_m]
        o2_m = self.omega_w[w2_m]
        w1_m[o1_m < omega1_m] += 1
        w2_m[o2_m < omega2_m] += 1
        return w1_m, w2_m


def find_maximum_frequency(calc, nbands, fd=None):
    """Determine the maximum electron-hole pair transition energy."""
    epsmin = 10000.0
    epsmax = -10000.0
    for kpt in calc.wfs.kpt_u:
        epsmin = min(epsmin, kpt.eps_n[0])
        epsmax = max(epsmax, kpt.eps_n[nbands - 1])

    if fd is not None:
        print('Minimum eigenvalue: %10.3f eV' % (epsmin * Hartree), file=fd)
        print('Maximum eigenvalue: %10.3f eV' % (epsmax * Hartree), file=fd)

        return epsmax - epsmin


def get_nonlinear_frequency_grid(calc, nbands, fd=None,
                                 domega0=0.1, omega2=10.0, omegamax=None,
                                 **unused):
    domega0 = domega0 / Hartree
    omega2 = omega2 / Hartree
    omegamax = None if omegamax is None else omegamax / Hartree
    if omegamax is None:
        omegamax = find_maximum_frequency(calc, nbands)

    if fd is not None:
        print('Using nonlinear frequency grid from 0 to %.3f eV' %
              (omegamax * Hartree), file=fd)
    return FrequencyDescriptor(domega0, omega2, omegamax)


def get_unique_spin_str(spin):
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


def get_PWDescriptor(ecut, gd, q_c, gammacentered=False):
    """Get the planewave descriptor of q_c."""
    qd = KPointDescriptor([q_c])
    pd = PWDescriptor(ecut, gd,
                      complex, qd, gammacentered=gammacentered)
    return pd


def get_band_summation_domain(bandsummation, nbands, nocc1=None, nocc2=None):
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
    n_M : ndarray
        band index 1, M = combined index
    m_M : ndarray
        band index 2, M = combined index
    """
    _get_band_sum_domain = create_get_band_sum_domain(bandsummation)
    n_M, m_M = _get_band_sum_domain(nbands)
    
    return remove_null_transitions(n_M, m_M, nocc1=nocc1, nocc2=nocc2)


def create_get_band_sum_domain(bandsummation):
    """Creator component deciding how to carry out band summation."""
    if bandsummation == 'pairwise':
        return get_pairwise_band_sum_domain
    elif bandsummation == 'double':
        return get_double_band_sum_domain
    raise ValueError(bandsummation)


def remove_null_transitions(n_M, m_M, nocc1=None, nocc2=None):
    """Remove pairs of bands, between which transitions are impossible"""
    n_newM = []
    m_newM = []
    for n, m in zip(n_M, m_M):
        if nocc1 is not None and (n < nocc1 and m < nocc1):
            continue  # both bands are fully occupied
        elif nocc2 is not None and (n >= nocc2 and m >= nocc2):
            continue  # both bands are completely unoccupied
        n_newM.append(n)
        m_newM.append(m)

    return np.array(n_newM), np.array(m_newM)


def get_double_band_sum_domain(nbands):
    """Make a simple double sum"""
    n_n = np.arange(0, nbands)
    m_m = np.arange(0, nbands)
    n_nm, m_nm = np.meshgrid(n_n, m_m)
    n_M, m_M = n_nm.flatten(), m_nm.flatten()

    return n_M, m_M


def get_pairwise_band_sum_domain(nbands):
    """Make a sum over all pairs"""
    n_n = range(0, nbands)
    n_M = []
    m_M = []
    for n in n_n:
        m_m = range(n, nbands)
        n_M += [n] * len(m_m)
        m_M += m_m

    return np.array(n_M), np.array(n_M)


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
