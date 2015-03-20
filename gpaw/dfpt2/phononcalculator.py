import os
from math import pi

import numpy as np
import numpy.linalg as la

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
        wfs = WaveFunctions(nbands, kpt_u, setups, kd_gs, gd, dtype=dtype_gs)
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
                                   dtype=self.dtype)

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

    def __getstate__(self):
        """Method used when pickling.

        Bound method attributes cannot be pickled and must therefore be deleted
        before an instance is dumped to file.

        """

        # Get state of object and take care of troublesome attributes
        state = dict(self.__dict__)
        state['kd'].__dict__['comm'] = mpi.serial_comm
        state.pop('calc')
        state.pop('perturbation')
        state.pop('response_calc')

        return state

    def get_phonons(self, qpts_q=None, clean=False, name=None, path=None):
        """Run calculation for atomic displacements and update matrix.

        Parameters
        ----------
        qpts: List
            List of q-points indices for which the dynamical matrix will be
            calculated (only temporary).

        """

        if not self.initialized:
            self.initialize()

        assert isinstance(qpts_q, list)

        # Update name and path attributes
        self.set_name_and_path(name=name, path=path)
        # Get string template for filenames
        filename_str = self.get_filename_string()


        # XXX Make a single ground_state_contributions member function
        # Ground-state contributions to the force constants
        self.dyn.density_ground_state(self.calc)
        # self.dyn.wfs_ground_state(self.calc, self.response_calc)

        # Calculate linear response wrt q-vectors and displacements of atoms
        for q in qpts_q:

            if not np.allclose(q,[0,0,0]):
                self.perturbation.set_q(q)

            # First-order contributions to the force constants
            for a in self.dyn.indices:
                for v in [0, 1, 2]:

                    # Check if the calculation has already been done
                    filename = filename_str % (0, a, v)
                    # Wait for all sub-ranks to enter
                    self.comm.barrier()

                    if os.path.isfile(os.path.join(self.path, filename)):
                        continue

                    if self.comm.rank == 0:
                        fd = open(os.path.join(self.path, filename), 'w')

                    # Wait for all sub-ranks here
                    self.comm.barrier()

                    components = ['x', 'y', 'z']
                    symbols = self.atoms.get_chemical_symbols()
                    print("q-vector: %d %d %d" % (q[0],q[1],q[2]))
                    print("Atom index: %i" % a)
                    print("Atomic symbol: %s" % symbols[a])
                    print("Component: %s" % components[v])

                    # Set atom and cartesian component of perturbation
                    self.perturbation.set_av(a, v)
                    # Calculate linear response
                    self.response_calc(self.perturbation)

                    # Calculate row of the matrix of force constants
                    self.dyn.calculate_row(self.perturbation,
                                           self.response_calc)

                    # Write force constants to file
                    if self.comm.rank == 0:
                        self.dyn.write(fd, 0, a, v)
                        fd.close()


                    # Wait for the file-writing rank here
                    self.comm.barrier()

        # XXX
        # Check that all files are valid and collect in a single file
        # Remove the files
        if clean:
            self.clean()

    def get_atoms(self):
        """Return atoms."""

        return self.atoms

    def get_dynamical_matrix(self):
        """Return reference to ``dyn`` attribute."""

        return self.dyn

    def get_filename_string(self):
        """Return string template for force constant filenames."""

        name_str = (self.name + '.' + 'q_%%0%ii_' % len(str(self.kd.nibzkpts)) +
                    'a_%%0%ii_' % len(str(len(self.atoms))) + 'v_%i' + '.pckl')

        return name_str

    def set_atoms(self, atoms):
        """Set atoms to be included in the calculation.

        Parameters
        ----------
        atoms: list
            Can be either a list of strings, ints or ...
        """

        assert isinstance(atoms, list)

        if isinstance(atoms[0], str):
            assert np.all([isinstance(atom, str) for atom in atoms])
            sym_a = self.atoms.get_chemical_symbols()
            # List for atomic indices
            indices = []
            for type in atoms:
                indices.extend([a for a, atom in enumerate(sym_a)
                                if atom == type])
        else:
            assert np.all([isinstance(atom, int) for atom in atoms])
            indices = atoms

        self.dyn.set_indices(indices)

    def set_name_and_path(self, name=None, path=None):
        """Set name and path of the force constant files.

        name: str
            Base name for the files which the elements of the matrix of force
            constants will be written to.
        path: str
            Path specifying the directory where the files will be dumped.
        """

        if name is None:
            self.name = 'phonon.' + self.atoms.get_chemical_formula()
        else:
            self.name = name
        # self.name += '.nibzkpts_%i' % self.kd.nibzkpts

        if path is None:
            self.path = '.'
        else:
            self.path = path

        # Set corresponding attributes in the ``dyn`` attribute
        filename_str = self.get_filename_string()
        self.dyn.set_name_and_path(filename_str, self.path)

    def clean(self):
        """Delete generated files."""

        filename_str = self.get_filename_string()

        for q in range(self.kd.nibzkpts):
            for a in range(len(self.atoms)):
                for v in [0, 1, 2]:
                    filename = filename_str % (q, a, v)
                    if os.path.isfile(os.path.join(self.path, filename)):
                        os.remove(filename)

    def get_phonon_energies(self, k_c, modes=False, acoustic=True):
        """Calculate phonon dispersion along a path in the Brillouin zone.

        The dynamical matrix at arbitrary q-vectors is obtained by Fourier
        transforming the real-space matrix. In case of negative eigenvalues
        (squared frequency), the corresponding negative frequency is returned.

        Parameters
        ----------
        path_kc: ndarray
            List of k-point coordinates (in units of the reciprocal lattice
            vectors) specifying the path in the Brillouin zone for which the
            dynamical matrix will be calculated.
        modes: bool
            Returns both frequencies and modes (mass scaled) when True.
        acoustic: bool
            Restore the acoustic sum-rule in the calculated force constants.
        """

        assert np.all(np.asarray(k_c) <= 1.0), \
                "Scaled coordinates must be given"

        q_c = k_c

        # Assemble the dynanical matrix from calculated force constants
        self.dyn.assemble(acoustic=acoustic)
        print self.dyn.D_k
        # Get the dynamical matrix in real-space
        DR_lmn, R_clmn = self.dyn.real_space()

        # Reshape for the evaluation of the fourier sums
        shape = DR_lmn.shape
        DR_m = DR_lmn.reshape((-1,) + shape[-2:])
        R_cm = R_clmn.reshape((3, -1))

        # Lists for frequencies and modes along path
        omega_kn = []
        u_kn = []
        # Number of atoms included
        N = len(self.dyn.get_indices())

        # Mass prefactor for the normal modes
        m_inv_av = self.dyn.get_mass_array()

        # Evaluate fourier transform
        phase_m = np.exp(-2.j * pi * np.dot(q_c, R_cm))
        # Dynamical matrix in unit of Ha / Bohr**2 / amu
        D_q = np.sum(phase_m[:, np.newaxis, np.newaxis] * DR_m, axis=0)
        print D_q
       # D_q = self.dyn.D_k[0]
            #if modes:
                #omega2_n, u_avn = la.eigh(D_q, UPLO='L')
                ## Sort eigenmodes according to eigenvalues (see below) and
                ## multiply with mass prefactor
                #u_nav = u_avn[:, omega2_n.argsort()].T.copy() * m_inv_av
                ## Multiply with mass prefactor
                #u_kn.append(u_nav.reshape((3*N, -1, 3)))
            #else:
        omega2_n = la.eigvalsh(D_q, UPLO='L')
        print omega2_n
        # Sort eigenvalues in increasing order
        omega2_n.sort()
        # Use dtype=complex to handle negative eigenvalues
        omega_n = np.sqrt(omega2_n.astype(complex))

        # Take care of imaginary frequencies
        if not np.all(omega2_n >= 0.):
            indices = np.where(omega2_n < 0)[0]
            print(("WARNING, %i imaginary frequencies at "
                   "q = (% 5.2f, % 5.2f, % 5.2f) ; (omega_q =% 5.3e*i)"
                   % (len(indices), q_c[0], q_c[1], q_c[2],
                      omega_n[indices][0].imag)))

            omega_n[indices] = -1 * np.sqrt(np.abs(omega2_n[indices].real))

        omega_kn.append(omega_n.real)

        ## Conversion factor from sqrt(Ha / Bohr**2 / amu) -> eV
        s = units.Hartree**0.5 * units._hbar * 1.e10 / \
            (units._e * units._amu)**(0.5) / units.Bohr
        ## Convert to eV and Ang
        print s*omega2_n
        omega_kn = s * np.asarray(omega_kn)
        #if modes:
            #u_kn = np.asarray(u_kn) * units.Bohr
            #return omega_kn, u_kn

        return omega_kn
