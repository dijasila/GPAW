import numpy as np
import numpy.linalg as la

from ase.utils.timing import timer, Timer
import ase.units as units

from gpaw import GPAW
from gpaw.kpt_descriptor import KPointDescriptor
import gpaw.mpi as mpi
from gpaw.symmetry import Symmetry

from gpaw.dfpt2.dynamicalmatrix import DynamicalMatrix
from gpaw.dfpt2.phononperturbation import PhononPerturbation
from gpaw.dfpt2.poisson import FFTPoissonSolver
from gpaw.dfpt2.responsecalculator import ResponseCalculator
from gpaw.dfpt2.wavefunctions import WaveFunctions


class PhononCalculator:
    """This class defines the interface for phonon calculations."""
    def __init__(self, calc, dispersion=False, symmetry=False,
                 q_c=[0., 0., 0.], world=mpi.world):
        """Inititialize class with a list of atoms.

        The atoms object must contain a converged ground-state calculation.

        Parameters
        ----------
        calc: str or Calculator
            Calculator containing a ground-state calculation.
        dispersion: bool
            If true, calculates dynamcial matrix on a grid of q-points. Else,
            use  a single q-point. Not implemented.
        symmetry: bool
            Use symmetries to reduce the q-vectors of the dynamcial matrix
            (None, False or True). None=off, False=time_reversal only. True
            isn't implemented yet.
        q_c: array
            If not dispersion, give q-vector for phonon calculation. Default is
            Gamma point.
        world: Communicator
            Communicator for parallelization over k-points and real-space
            domain.
        """

        # Asserts for unfinished and non features
        assert symmetry in [None, False], "Spatial symmetries not allowed yet"
        assert not dispersion, "Phonon dispersion not implemented yet"

        self.timer = Timer()

        if isinstance(calc, str):
            self.calc = GPAW(calc, communicator=mpi.serial_comm, txt=None)
        else:
            self.calc = calc

        # Make sure localized functions are initialized
        self.calc.set_positions()
        # Note that this under some circumstances (e.g. when called twice)
        # allocates a new array for the P_ani coefficients !!

        # Store useful objects
        self.atoms = self.calc.get_atoms()
        # Get rid of ``calc`` attribute
        self.atoms.calc = None

        # Boundary conditions
        pbc_c = self.atoms.get_pbc()

        # Not supporting finite systems at the moment.
        assert not np.all(pbc_c is False), "Only extended systems, please"

        # Check, whether q vector is the Gamma point
        self.q_c = np.array(q_c)
        if np.allclose(self.q_c, [0., 0., 0.], 1e-5):
            self.dtype = float
            self.gamma = True
        else:
            self.dtype = complex
            self.gamma = False

        # Initialize FFT Poisson solver object for extended systems
        poisson_solver = FFTPoissonSolver(dtype=self.dtype)

        # Set symmetry object
        cell_cv = self.calc.atoms.get_cell()
        # XXX - no clue how to get magmom - ignore it for the moment
        # XXX should add assert, that we don't use magnetic calculation
        # m_av = magmom_av.round(decimals=3)  # round off
        # id_a = zip(setups.id_a, *m_av.T)
        id_a = self.calc.wfs.setups.id_a

        if symmetry is None:
            self.symmetry = Symmetry(id_a, cell_cv, point_group=False,
                                     time_reversal=False)
        else:
            self.symmetry = Symmetry(id_a, cell_cv, point_group=False,
                                     time_reversal=True)

        del cell_cv, id_a

        # XXX So, for historic reasons there is a k-point descriptor for the
        # q-points. Now, that shouldn't be necessary. Try to get rid of it.

        # K-point descriptor for the q-vector(s) of the dynamical matrix
        # Note, no explicit parallelization here.
        # DEPRECATED
        self.kd = KPointDescriptor([q_c, ], 1)
        self.kd.set_symmetry(self.atoms, self.symmetry)
        self.kd.set_communicator(mpi.serial_comm)

        # Number of occupied bands
        nvalence = self.calc.wfs.nvalence
        nbands = nvalence / 2 + nvalence % 2
        del nvalence
        assert nbands <= self.calc.wfs.bd.nbands

        # Extract other useful objects
        # Ground-state k-point descriptor - used for the k-points in the
        # ResponseCalculator
        # XXX replace communicators when ready to parallelize
        kd_gs = self.calc.wfs.kd
        gd = self.calc.density.gd
        kpt_u = self.calc.wfs.kpt_u
        setups = self.calc.wfs.setups
        dtype_gs = self.calc.wfs.dtype

        # WaveFunctions
        spos_ac = self.atoms.get_scaled_positions()
        wfs = WaveFunctions(nbands, kpt_u, setups, kd_gs, gd, spos_ac,
                            dtype=dtype_gs)
        del kd_gs, gd, kpt_u, setups, dtype_gs

        # Linear response calculator
        self.response_calc = ResponseCalculator(self.calc, wfs,
                                                dtype=self.dtype)

        # Phonon perturbation
        self.perturbation = PhononPerturbation(self.calc, self.q_c,
                                               poisson_solver,
                                               kd=self.kd,  # DEPRECATED
                                               dtype=self.dtype)

        # Dynamical matrix
        self.dyn = DynamicalMatrix(self.atoms, self.kd,  # DEPRECATED
                                   dtype=self.dtype, timer=self.timer)

        # Initialization flag
        self.initialized = False

        # Parallel communicator for parallelization over kpts and domain
        self.comm = world

    def initialize(self):
        """Initialize response calculator and perturbation."""

        # Get scaled atomic positions
        spos_ac = self.atoms.get_scaled_positions()

        self.perturbation.initialize(spos_ac)
        self.response_calc.initialize(spos_ac)

        self.initialized = True

    def get_phonons(self, modes=False):
        """Interface routine to calculate phonon energies for a given
        q-vector.

        At the moment this is a trivial routine. In the end it should decide
        whether to calculate the dynamcial matrix, or to read it from a file
        before diagonalising it.
        Possibly one could also read in real space force constants. However,
        calculating the force constant matrix should be done using a different
        interface.

        Parameters
        ----------
        modes: bool
            Returns both frequencies and modes (mass scaled) when True.
       """

        dynmat = self.calculate_dynamicalmatrix()
        energies = self.diagonalize_dynamicalmatrix(dynmat, modes=False)

        self.timer.write()

        return energies

    @timer('PHONON calculate dynamical matrix')
    def calculate_dynamicalmatrix(self):
        """Run calculation for atomic displacements and construct the dynamcial
        matrix.

        Parameters
        ----------

        """

        if not self.initialized:
            self.initialize()

        # XXX Make a single ground_state_contributions member function
        # Ground-state contributions to the force constants
        self.dyn.density_ground_state(self.calc)

        # Calculate linear response wrt q-vector and displacements of atoms

        if not self.gamma:
            # HACK
            self.perturbation.set_q(0)

        components = ['x', 'y', 'z']
        symbols = self.atoms.get_chemical_symbols()

        # First-order contributions to the force constants
        for a in self.dyn.indices:
            for v in [0, 1, 2]:
                print("Atom index: %i" % a)
                print("Atomic symbol: %s" % symbols[a])
                print("Component: %s" % components[v])

                # Set atom and cartesian component of perturbation
                self.perturbation.set_av(a, v)
                # Calculate linear response
                self.response_calc(self.perturbation)

                # Calculate row of the matrix of force constants
                self.dyn.calculate_row(self.perturbation, self.response_calc)

        self.comm.barrier()
        # np.save('bla', self.dyn.C_qaavv)
        self.dyn.assemble(dynmat=self.dyn.C_qaavv, acoustic=self.gamma)

        return self.dyn.D_k[0]

    @timer('PHONON diagonalise dynamicial matrix')
    def diagonalize_dynamicalmatrix(self, dynmat, modes=False):
        """Calculates phonon energies by diagonalising the dynamcial matrix.

        In case of negative eigenvalues (squared frequency), the corresponding
        negative frequency is returned.

        Parameters
        ----------
        dynmat: array
            Dynamical matrix
        modes: bool
            Returns both frequencies and modes (mass scaled) when True.
        """

        D_q = dynmat
        u_n = []

        if modes:
            m_inv_av = self.dyn.get_mass_array()
            N = len(self.dyn.get_indices())

            omega2_n, u_avn = la.eigh(D_q, UPLO='L')
            # Sort eigenmodes according to eigenvalues (see below) and
            # multiply with mass prefactor
            u_nav = u_avn[:, omega2_n.argsort()].T.copy() * m_inv_av
            # Multiply with mass prefactor
            u_n.append(u_nav.reshape((3*N, -1, 3)))
        else:
            omega2_n = la.eigvalsh(D_q, UPLO='L')
            # Sort eigenvalues in increasing order
        omega2_n.sort()

        # Use dtype=complex to handle negative eigenvalues
        omega_n = np.sqrt(omega2_n.astype(complex))

        # Take care of imaginary frequencies
        if not np.all(omega2_n >= 0.):
            indices = np.where(omega2_n < 0)[0]
            print(("WARNING, %i imaginary frequencies: (omega_i =% 5.3e*i)"
                   % (len(indices), omega_n[indices][0].imag)))
            omega_n[indices] = -1 * np.sqrt(np.abs(omega2_n[indices].real))

        # Conversion factor from sqrt(Ha / Bohr**2 / amu) -> eV
        s = units.Hartree**0.5 * units._hbar * 1.e10 / \
            (units._e * units._amu)**(0.5) / units.Bohr

        # Convert to eV and Ang
        omega_n = s**2 * np.asarray(omega_n.real)

        if modes:
            u_n = np.asarray(u_n) * units.Bohr
            return omega_n, u_n
        else:
            return omega_n
