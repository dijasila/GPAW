import numbers
import numpy as np

from ase.units import Hartree

from gpaw import extra_parameters
import gpaw.mpi as mpi
from gpaw.response.pair import get_PairMatrixElement


class ChiKS:
    """Class for calculating response functions in the Kohn-Sham system"""

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

        Note: Any parameter you want to pass to PairMatrixElement,
              ex. gate_voltage to PairDensity, you can pass as a kwarg.

        Callables
        ---------
        self.calculate(**kwargs) : func
            Runs the ChiKS calculation, returning the response function.
            Returned format can varry depending on response and mode.
        """

        self.response = response
        self.mode = mode
        self.world = world

        # Initialize the PairMatrixElement object
        PME = get_PairMatrixElement(response)
        self.pair = PME(gs, world=self.world, txt=txt, timer=timer, **kwargs)

        # Extract ground state calculator, timer and filehandle for output
        calc = self.pair.calc
        self.calc = calc
        self.timer = self.pair.timer
        self.fd = self.pair.fd

        self.setup_calculator(**kwargs)

    def setup_calculator(self, **kwargs):
        self._setup_calculator, self.calculate = self.get_calculator()
        self._setup_calculator(**kwargs)

    def get_calculator(self):
        """Creator component deciding how to calculate chi"""
        if self.mode == 'pw':
            return self.setup_pw_calculator, self.pw_calculate
        else:
            ValueError(self.mode)

    def setup_pw_calculator(self, frequencies=None, eta=0.2,
                            ecut=50, gammacentered=False, pbc=None,
                            nbands=None, bandsummation='pairwise',
                            nblocks=1, **kwargs):
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
        pbc : list
            Periodic directions of the system. Defaults to [True, True, True].
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
        """

        self.nbands = nbands or self.calc.wfs.bd.nbands

        self.frequencies = frequencies
        self.wd = self.get_pw_frequencies(frequencies, **kwargs)
        self.omega_w = self.wd.get_data()
        self.eta = eta / Hartree
        
        self.ecut = None if ecut is None else ecut / Hartree
        self.gammacentered = gammacentered

        self.setup_pbc(pbc)
        
        self.bandsummation = bandsummation
        
        self.setup_memory_distribution(nblocks)
        self.nblocks = nblocks

        self.vol = abs(np.linalg.det(self.calc.wfs.gd.cell_cv))
        self.Q_aGii = None

    def pw_calculate(self, q_c, spin='all', A_x=None):
        """Calculate the response function in plane wave mode.

        Parameters
        ----------
        q_c : list or ndarray
            Momentum vector.
        spin : str or int
            If 'all' then include all spins.
            If 0 or 1, only include this specific spin.
            (not used in transverse response functions)
        A_x : ndarray
            Output array. If None, the output array is created.

        Returns
        -------
        pd : Planewave descriptor
            Planewave descriptor for q_c.
        chi0_wGG : ndarray
            The response function.
        chi0_wxvG : ndarray or None
            (Only in optical limit) Wings of the density response function.
        chi0_wvv : ndarray or None
            (Only in optical limit) Head of the density response function.

        """
        # Goes to some band summation method XXX
        self.nocc1 = self.pair.nocc1  # number of completely filled bands
        self.nocc2 = self.pair.nocc2  # number of non-empty bands

        wfs = self.calc.wfs

        # Get spins to be used for s1  # XXX should be defined by domain?
        if self.response == 'density':
            if spin == 'all':
                spins = range(wfs.nspins)
            else:
                assert spin in range(wfs.nspins)
                spins = [spin]
        else:
            assert self.response in ['+-', '-+']
            if self.disable_spincons_time_reversal:
                if self.response == '+-':
                    spins = [0]
                else:
                    spins = [1]
            else:
                spins = [0, 1]

        q_c = np.asarray(q_c, dtype=float)
        optical_limit = np.allclose(q_c, 0.0) and self.response == 'density'

        pd = self.get_PWDescriptor(q_c, self.gammacentered)

        self.print_chi(pd)

        if extra_parameters.get('df_dry_run'):
            print('    Dry run exit', file=self.fd)
            raise SystemExit

    def get_pw_frequencies(self, frequencies, **kwargs):
        """Frequencies to evaluate response function at in GPAW units"""
        if frequencies is None:
            return get_nonlinear_frequency_grid(self.calc, self.nbands,
                                                fd=self.fd, **kwargs)
        else:
            return ArrayDescriptor(np.asarray(frequencies) / Hartree)

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

    def setup_pw_chi00(self, hilbert=True, timeordered=False, intraband=True,
                       disable_point_group=False,
                       disable_time_reversal=False,
                       disable_non_symmorphic=True,
                       integrationmode=None,
                       rate=0.0, eshift=0.0, **unused):
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
            If == 'tetrahedron integration' then the kpoint integral is
            performed using the linear tetrahedron method.
        rate : float
            Unknown parameter for intraband calculation in optical limit
        eshift : float
            Shift unoccupied bands
        """

        self.hilbert = hilbert
        if self.frequencies is not None:
            assert not hilbert

        self.timeordered = bool(timeordered)
            
        if self.eta == 0.0:
            assert not self.hilbert
            assert not self.timeordered
            assert not self.omega_w.real.any()

        self.include_intraband = intraband
        
        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal
        self.disable_non_symmorphic = disable_non_symmorphic
        
        self.integrationmode = integrationmode
        if self.integrationmode is not None:
            print('Using integration method: ' + self.integrationmode,
                  file=self.fd)
        else:
            print('Using integration method: PointIntegrator', file=self.fd)
        
        if rate == 'eta':
            self.rate = self.eta
        else:
            self.rate = rate / Hartree

        self.eshift = eshift / Hartree


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
