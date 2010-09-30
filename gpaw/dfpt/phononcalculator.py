"""This module provides an interface class for phonon calculations."""

__all__ = ["PhononCalculator"]

import numpy as np

from gpaw import GPAW
from gpaw.mpi import world, rank, size, serial_comm
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.dfpt.poisson import PoissonSolver, FFTPoissonSolver
from gpaw.dfpt.responsecalculator import ResponseCalculator
from gpaw.dfpt.phononperturbation import PhononPerturbation
from gpaw.dfpt.wavefunctions import WaveFunctions
from gpaw.dfpt.dynamicalmatrix import DynamicalMatrix


class PhononCalculator:
    """This class defines the interface for phonon calculations."""
    
    def __init__(self, calc, gamma=True, e_ph=False, **kwargs):
        """Inititialize class with a list of atoms.

        The atoms object must contain a converged ground-state calculation.

        The irreducible set of q-vectors in which the dynamical matrix will be
        calculated will be the irreducible set of k-vector used in the
        ground-state calculations.
        
        For now the q-points are taken from the ground-state calculation.

        Parameters
        ----------
        calc: str or Calculator
            Calculator containing a ground-state calculation.
        gamma: bool
            Gamma-point calculation with respect to the q-vector of the
            dynamical matrix. When ``False``, the Monkhorst-Pack grid from the
            ground-state calculation is used.
        e_ph: bool
            Calculate electron-phonon coupling.

        """

        if isinstance(calc, str):
            self.calc = GPAW(calc, communicator=serial_comm, txt=None)
        else:
            self.calc = calc

        # Make sure localized functions are initialized
        self.calc.set_positions()
        # Note that this under some circumstances (e.g. when called twice)
        # allocates a new array for the P_ani coefficients !!
        
        # Store useful objects
        self.atoms = self.calc.get_atoms()

        # Boundary conditions
        pbc_c = self.calc.atoms.get_pbc()
        
        if np.all(pbc_c == False):
            self.gamma = True
            self.dtype = float
            bzq_kc = np.array(((0, 0, 0),), dtype=float)
            # Multigrid Poisson solver
            poisson_solver = PoissonSolver()
        else:
            if gamma:
                self.gamma = True
                self.dtype = float
                bzq_kc = np.array(((0, 0, 0),), dtype=float)
            else:
                self.gamma = False
                self.dtype = complex
                # Get k-points from ground-state calculation
                bzq_kc = self.calc.get_bz_k_points()
                
            # FFT Poisson solver
            poisson_solver = FFTPoissonSolver(dtype=self.dtype)
            
        # Include all atoms and cartesian coordinates by default
        self.atoms_a = dict([ (atom.index, [0, 1, 2]) for atom in self.atoms])
        
        # Ground-state k-point descriptor - used for the k-points in the
        # ResponseCalculator 
        kd_gs = self.calc.wfs.kd

        # K-point descriptor for the q-vectors of the dynamical matrix
        self.kd = KPointDescriptor(bzq_kc, 1)
        self.kd.set_symmetry(self.atoms, self.calc.wfs.setups, None)
        self.kd.set_communicator(world)
        
        # Number of occupied bands
        nvalence = self.calc.wfs.nvalence
        nbands = nvalence/2 + nvalence%2
        assert nbands <= self.calc.wfs.nbands

        #  WaveFunctions object
        wfs = WaveFunctions(nbands, self.calc.wfs.kpt_u, self.calc.wfs.setups,
                            kd_gs, self.calc.density.gd, dtype=self.calc.wfs.dtype)

        # Linear response calculator
        self.response_calc = ResponseCalculator(self.calc, wfs, dtype=self.dtype)
        
        # Phonon perturbation
        self.perturbation = PhononPerturbation(self.calc, self.kd,
                                               poisson_solver,
                                               dtype=self.dtype)

        # Dynamical matrix object - its dtype should be determined by the
        # presence of inversion symmetry - NO, only for monoatomic bases !
        inversion_symmetry = False
        if inversion_symmetry:
            D_dtype = float
        else:
            D_dtype = self.dtype
        
        self.D_matrix = DynamicalMatrix(self.atoms, self.kd,
                                        dtype=D_dtype)

        # Initialization flag
        self.initialized = False

        # Parallel stuff
        self.comm = world

    def initialize(self):
        """Initialize response calculator and perturbation."""

        # Get scaled atomic positions
        spos_ac = self.atoms.get_scaled_positions()

        self.perturbation.initialize(spos_ac)
        self.response_calc.initialize(spos_ac)

        self.initialized = True
        
    def set_atoms(self, atoms_a, exclude=True):
        """Set indices of atoms to include in the calculation.

        Parameters
        ----------
        exclude: bool
            If True, all other atoms are neglected in the calculation of the
            dynamical matrix.

        """

        assert isinstance(atoms_a, dict) or isinstance(atoms_a, list)
        
        if isinstance(atoms_a, dict):
            if exclude:
                self.atoms_a = atoms_a
            else:
                self.atoms_a.update(atoms_a)
        else:
            # List of atoms indices
            for a in atoms_a:
                self.atoms_a = [0, 1, 2]

    def __call__(self, qpts_q=None):
        """Run calculation for atomic displacements and update matrix.

        Parameters
        ----------
        qpts: List
            List of q-points indices for which the dynamical matrix will be
            calculated (only temporary).

        """

        if not self.initialized:
            self.initialize()

        if self.gamma:
            qpts_q = [0]
        elif qpts_q is None:
            qpts_q = range(self.kd.mynks)
        else:
            assert isinstance(qpts_q, list)

        # Calculate linear response wrt q-vectors and displacements of atoms
        for q in qpts_q:
            
            if not self.gamma:
                self.perturbation.set_q(q)

            # First-order contributions to the force constants
            for a in self.atoms_a:
    
                for v in self.atoms_a[a]:
    
                    components = ['x', 'y', 'z']
                    symbols = self.atoms.get_chemical_symbols()
                    print "q-vector index: %i" % (self.kd.ks0 + q)
                    print "Atom index: %i" % a
                    print "Atomic symbol: %s" % symbols[a]
                    print "Component: %s" % components[v]

                    # Set atom and cartesian component of perturbation
                    self.perturbation.set_av(a, v)
                    # Calculate linear response
                    self.response_calc(self.perturbation)

                    # Calculate corresponding row of dynamical matrix
                    self.D_matrix.update_row(self.perturbation,
                                             self.response_calc)

        # Ground-state contributions to the force constants
        self.D_matrix.density_ground_state(self.calc)
        # self.D_matrix.wfs_ground_state(self.calc, self.response_calc)

        self.kd.comm.barrier()
        
    def get_dynamical_matrix(self):
        """Return reference to ``DynamicalMatrix`` object."""
        
        return self.D_matrix
