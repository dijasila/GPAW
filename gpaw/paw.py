# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a PAW-class.

The central object that glues everything together!"""

import sys

import Numeric as num

import gpaw.io
import gpaw.mpi as mpi
import gpaw.occupations as occupations
from gpaw import output
from gpaw import debug
from gpaw import ConvergenceError
from gpaw.density import Density
from gpaw.eigensolvers import eigensolver
from gpaw.grid_descriptor import GridDescriptor
from gpaw.hamiltonian import Hamiltonian
from gpaw.kpoint import KPoint
from gpaw.localized_functions import LocFuncBroadcaster
from gpaw.utilities.timing import Timer
from gpaw.xc_functional import XCFunctional
from gpaw.mpi import run
import _gpaw

MASTER = 0


# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.


"""ASE-calculator interface."""


import os
import sys
import tempfile
import time
import weakref

import Numeric as num
from ASE.Units import units, Convert
from ASE.Utilities.MonkhorstPack import MonkhorstPack
from ASE.ChemicalElements.symbol import symbols
from ASE.ChemicalElements import numbers
import ASE

from gpaw.utilities import check_unit_cell
from gpaw.utilities.memory import maxrss
from gpaw.version import version
import gpaw.utilities.timing as timing
import gpaw
import gpaw.io
import gpaw.mpi as mpi
from gpaw.nucleus import Nucleus
from gpaw.rotation import rotation
from gpaw.domain import Domain
from gpaw.symmetry import Symmetry
from gpaw.xc_functional import XCFunctional
from gpaw.utilities import gcd
from gpaw.utilities.timing import Timer
from gpaw.utilities.memory import estimate_memory
from gpaw.setup import create_setup
from gpaw.pawextra import PAWExtra
from gpaw.output import Output
from gpaw import dry_run


MASTER = 0


class PAW(PAWExtra, Output):
    """This is the main calculation object for doing a PAW calculation.

    The ``Paw`` object is the central object for a calculation.  It is
    a container for **k**-points (there may only be one **k**-point).
    The attribute ``kpt_u`` is a list of ``KPoint`` objects (the
    **k**-point object stores the actual wave functions, occupation
    numbers and eigenvalues).  Each **k**-point object can be either
    spin up, spin down or no spin (spin-saturated calculation).
    Example: For a spin-polarized calculation on an isolated molecule,
    the **k**-point list will have length two (assuming the
    calculation is not parallelized over **k**-points/spin).

    These are the most important attributes of a ``Paw`` object:

    =============== =====================================================
    Name            Description
    =============== =====================================================
    ``domain``      Domain object.
    ``setups``      List of setup objects.
    ``symmetry``    Symmetry object.
    ``timer``       Timer object.
    ``nuclei``      List of ``Nucleus`` objects.
    ``out``         Output stream for text.
    ``gd``          Grid descriptor for coarse grids.
    ``finegd``      Grid descriptor for fine grids.
    ``kpt_u``       List of **k**-point objects.
    ``occupation``  Occupation-number object.
    ``nkpts``       Number of irreducible **k**-points.
    ``nmyu``        Number of irreducible spin/**k**-points pairs on
                    *this* CPU.
    ``nvalence``    Number of valence electrons.
    ``nbands``      Number of bands.
    ``nspins``      Number of spins.
    ``typecode``    Data type of wave functions (``Float`` or
                    ``Complex``).
    ``bzk_kc``      Scaled **k**-points used for sampling the whole
                    Brillouin zone - values scaled to [-0.5, 0.5).
    ``ibzk_kc``     Scaled **k**-points in the irreducible part of the
                    Brillouin zone.
    ``weights_k``   Weights of the **k**-points in the irreducible part
                    of the Brillouin zone (summing up to 1).
    ``myibzk_kc``   Scaled **k**-points in the irreducible part of the
                    Brillouin zone for this CPU.
    ``kpt_comm``    MPI-communicator for parallelization over
                    **k**-points.
    =============== =====================================================

    Energy contributions and forces:

    =========== ==========================================
                Description
    =========== ==========================================
    ``Ekin``    Kinetic energy.
    ``Epot``    Potential energy.
    ``Etot``    Total energy.
    ``Exc``     Exchange-Correlation energy.
    ``Eext``    Energy of external potential
    ``Eref``    Reference energy for all-electron atoms.
    ``S``       Entropy.
    ``Ebar``    Should be close to zero!
    ``F_ac``    Forces.
    =========== ==========================================

    The attribute ``usesymm`` has the same meaning as the
    corresponding ``Calculator`` keyword (see the Manual_).  Internal
    units are Hartree and Angstrom and ``Ha`` and ``a0`` are the
    conversion factors to external `ASE units`_.  ``error`` is the
    error in the Kohn-Sham wave functions - should be zero (or small)
    for a converged calculation.

    Booleans describing the current state:

    ============= ======================================
    Boolean       Description
    ============= ======================================
    ``forces_ok`` Have the forces bee calculated yet?
    ``converged`` Do we have a self-consistent solution?
    ============= ======================================

    Number of iterations for:

    ============ ===============================
                 Description
    ============ ===============================
    ``nfermi``   finding the Fermi-level
    ``niter``    solving the Kohn-Sham equations
    ``npoisson`` Solving the Poisson equation
    ============ ===============================

    Only attribute not mentioned now is ``nspins`` (number of spins) and
    those used for parallelization:

    ================== ===================================================
    ``my_nuclei``      List of nuclei that have their
                       center in this domain.
    ``pt_nuclei``      List of nuclei with projector functions
                       overlapping this domain.
    ``ghat_nuclei``    List of nuclei with compensation charges
                       overlapping this domain.
    ``locfuncbcaster`` ``LocFuncBroadcaster`` object for parallelizing
                       evaluation of localized functions (used when
                       parallelizing over **k**-points).
    ================== ===================================================

    .. _Manual: https://wiki.fysik.dtu.dk/gridcode/Manual
    .. _ASE units: https://wiki.fysik.dtu.dk/ase/Units

    Parameters:
    =============== ===================================================
    ``nvalence``    Number of valence electrons.
    ``nbands``      Number of bands.
    ``nspins``      Number of spins.
    ``random``      Initialize wave functions with random numbers
    ``typecode``    Data type of wave functions (``Float`` or
                    ``Complex``).
    ``kT``          Temperature for Fermi-distribution.
    ``bzk_kc``      Scaled **k**-points used for sampling the whole
                    Brillouin zone - values scaled to [-0.5, 0.5).
    ``ibzk_kc``     Scaled **k**-points in the irreducible part of the
                    Brillouin zone.
    ``myspins``     List of spin-indices for this CPU.
    ``weights_k``   Weights of the **k**-points in the irreducible part
                    of the Brillouin zone (summing up to 1).
    ``myibzk_kc``   Scaled **k**-points in the irreducible part of the
                    Brillouin zone for this CPU.
    ``myweights_k`` Weights of the **k**-points on this CPU.
    ``kpt_comm``    MPI-communicator for parallelization over
                    **k**-points.
    =============== ===================================================
    """

    def __init__(self, filename=None, **kwargs):
        """ASE-calculator interface.

        The following parameters can be used: `nbands`, `xc`, `kpts`,
        `spinpol`, `gpts`, `h`, `charge`, `usesymm`, `width`, `mix`,
        `hund`, `lmax`, `fixdensity`, `tolerance`, `txt`,
        `hosts`, `parsize`, `softgauss`, `stencils`, and
        `convergeall`.

        If you don't specify any parameters, you will get:

        Defaults: neutrally charged, LDA, gamma-point calculation, a
        reasonable grid-spacing, zero Kelvin electronic temperature,
        and the number of bands will be equal to the number of atomic
        orbitals present in the setups. Only occupied bands are used
        in the convergence decision. The calculation will be
        spin-polarized if and only if one or more of the atoms have
        non-zero magnetic moments. Text output will be written to
        standard output.

        For a non-gamma point calculation, the electronic temperature
        will be 0.1 eV (energies are extrapolated to zero Kelvin) and
        all symmetries will be used to reduce the number of
        **k**-points."""

        self.input_parameters = {
            'h':             None,
            'xc':            'LDA',
            'gpts':          None,
            'kpts':          None,
            'lmax':          2,
            'charge':        0,
            'fixmom':        False,
            'nbands':        None,
            'setups':        'paw',
            'width':         None,
            'spinpol':       None,
            'usesymm':       True,
            'stencils':      (2, 'M', 3),
            'tolerance':     1.0e-9,
            'fixdensity':    False,
            'convergeall':   False,
            'mix':           (0.25, 3, 1.0),
            'txt':           '-',
            'hund':          False,
            'random':        False,
            'external':      None,
            'decompose':     None,
            'verbosity':     0,
            'eigensolver':   RMM_DIIS,
            'poissonsolver': GaussSeidel}

        self.converged = False
        self.initialized = False
        self.wave_functions_initialized = False
        
        if filename is not None:
            reader = self.read_parameters(filename)

        if 'h' in kwargs and 'gpts' in kwargs:
            raise TypeError("""You can't use both "gpts" and "h"!""")
            
        for name, value in kwargs.items():
            if name in ['random', 'hund', 'mix', 'txt', 'verbosity',
                        'decompose', 'eigensolver', 'poissonsolver',
                        'external']:
                self.input_parameters[name] = value
            elif name in ['xc', 'nbands', 'spinpol', 'kpts', 'usesymm',
                          'gpts', 'h', 'width', 'lmax', 'setups', 'stencils',
                          'charge', 'fixmom', 'fixdensity', 'tolerance',
                          'convergeall']:
                self.converged = False
                self.input_parameters[name] = value
            else:
                raise RuntimeError('Unknown keyword: ' + name)

        Output.__init__(self)
        
        if filename is not None:
            self.initialize()
            gpaw.io.read(self, reader)
            self.plot_atoms()

    def initialize(self):
        """Inexpensive initialization."""
        self.timer = Timer()
        timer.start('Init')

        self.kpt_u = None
        
        atoms = self.atoms()
        self.natoms = len(atoms)
        magmom_a = atoms.GetMagneticMoments()
        pos_ac = atoms.GetCartesianPositions() / self.a0
        cell_cc = atoms.GetUnitCell() / self.a0
        pbc_c = atoms.GetPeriodicBoundaryConditions()
        Z_a = atoms.GetAtomicNumbers()
        
        # Check that the cell is orthorhombic:
        check_unit_cell(cell_cc)
        # Get the diagonal:
        cell_c = num.diagonal(cell_cc)
        
        p = self.input_parameters

        # Set the scaled k-points:
        kpts = p['kpts']
        if kpts is None:
            self.bzk_kc = num.zeros((1, 3), num.Float)
        elif isinstance(kpts[0], int):
            self.bzk_kc = MonkhorstPack(bzk_kc)
        else:
            self.bzk_kc = num.array(kpts)
        
        magnetic = num.sometrue(magmom_a)

        self.spinpol = p['spinpol']
        if self.spinpol is None:
            self.spinpol = magnetic
        elif magnetic and not self.spinpol:
            raise ValueError('Non-zero initial magnetic moment for a ' +
                             'spin-paired calculation!')

        self.nspins = 1 + int(self.spinpol)

        self.fixmom = p['fixmom']
        if p['hund']:
            self.fixmom = True
            assert self.spinpol and self.natoms == 1

        if self.fixmom:
            assert self.spinpol

        self.xcfunc = XCFunctional(p['xc'])
        
        if p['gpts'] is not None and p['h'] is None:
            N_c = num.array(p['gpts'])
        else:
            if p['h'] is None:
                self.text('Using default value for grid spacing.')
                h = Convert(0.2, 'Ang', 'Bohr')
            else:
                h = p['h']
            # N_c should be a multiplum of 4:
            N_c = num.array([max(4, int(L / h / 4 + 0.5) * 4) for L in cell_c])
        
        # Create a Domain object:
        self.domain = Domain(cell_c, pbc_c)

        type_a = self.create_setups(p['setups'], Z_a)
        
        # Is this a gamma-point calculation?
        if len(self.bzk_kc) == 1 and not num.sometrue(self.bzk_kc[0]):
            self.gamma = True
            self.typecode = num.Float
            self.symmetry = None
            self.weights_k = [1.0]
            self.ibzk_kc = num.zeros((1, 3), num.Float)
            self.nkpts = 1
        else:
            self.gamma = False
            self.typecode = num.Complex
            # Reduce the the k-points to those in the irreducible part of
            # the Brillouin zone:
            self.symmetry, self.weights_k, self.ibzk_kc = reduce_kpoints(
                self.bzk_kc, pos_ac, Z_a, type_a, magmom_a, self.domain,
                p['usesymm'])
            self.nkpts = len(self.ibzk_kc)
        
            if p['usesymm'] and self.symmetry is not None:
                # Find rotation matrices for spherical harmonics:
                R_slmm = [[rotation(l, symm) for l in range(3)]
                          for symm in self.symmetry.symmetries]
        
                for setup in self.setups.values():
                    setup.calculate_rotations(R_slmm)
        
        self.distribute_kpoints_and_spins(p['parsize'], N_c)
        
        if dry_run:
            # Estimate the amount of memory needed
            estimate_memory(N_c, nbands, nkpts, nspins, typecode, nuclei, h_c, out)
            out.flush()
            sys.exit()

        # Build list of nuclei:
        self.nuclei = [Nucleus(setups[(Z, type)], a, self.typecode)
                       for a, (Z, type) in enumerate(zip(Z_a, type_a))]
            
        # Sum up the number of valence electrons:
        self.nvalence = 0
        nao = 0
        for nucleus in self.nuclei:
            self.nvalence += nucleus.setup.Nv
            nao += nucleus.setup.niAO
        self.nvalence -= p['charge']
        
        if self.nvalence < 0:
            raise ValueError(
                'Charge %f is not possible - not enough valence electrons' %
                p['charge'])
        
        self.nbands = p['nbands']
        if self.nbands is None:
            self.nbands = nao
        elif self.nbands <= 0:
            self.nbands = (self.nvalence + 1) // 2 + (-self.nbands)
            
        if self.nvalence > 2 * self.nbands:
            raise ValueError('Too few bands!')

        self.kT = p['width']
        if self.kT is None:
            if self.gamma:
                self.kT = 0
            else:
                self.kT = Convert(0.1, 'eV', 'Hartree')
        
        self.stencils = p['stencils']
        self.maxiter = p['maxiter']

        self.random_wf = p['random']

        # Construct grid descriptors for coarse grids (wave functions) and
        # fine grids (densities and potentials):
        self.gd = GridDescriptor(self.domain, N_c)
        self.finegd = GridDescriptor(self.domain, 2 * N_c)

        self.F_ac = None

        # Total number of k-point/spin combinations:
        nu = self.nkpts * self.nspins

        # Number of k-point/spin combinations on this cpu:
        self.nmyu = nu // self.kpt_comm.size

        self.kpt_u = []
        for u in range(self.nmyu):
            s, k = divmod(self.kpt_comm.rank * self.nmyu + u, self.nkpts)
            weight = self.weights_k[k] * 2 / self.nspins
            k_c = self.ibzk_kc[k]
            self.kpt_u.append(KPoint(self.gd, weight, s, k, u, k_c,
                                     self.typecode))

        self.locfuncbcaster = LocFuncBroadcaster(self.kpt_comm)

        self.my_nuclei = []
        self.pt_nuclei = []
        self.ghat_nuclei = []

        self.density = Density(self)
        self.hamiltonian = Hamiltonian(self)

        # Create object for occupation numbers:
        if kT == 0 or 2 * nbands == nvalence:
            self.occupation = occupations.ZeroKelvin(nvalence, nspins)
        else:
            self.occupation = occupations.FermiDirac(nvalence, nspins, kT)

        xcfunc.set_non_local_things(self)

        if fixmom:
            M = sum(magmom_a)
            self.occupation.fix_moment(M)

        self.occupation.set_communicator(kpt_comm)

        self.Eref = 0.0
        for nucleus in self.nuclei:
            self.Eref += nucleus.setup.E

        output.print_info(self)

        self.eigensolver = eigensolver(p['eigensolver'], paw)

        self.initialized = True
        
        self.timer.stop('Init')

    def calculate(self):
        """Update PAW calculaton if needed."""

        if not self.initialized:
            self.initialize()
            self.find_ground_state()
            return
        
        if self.lastcount == atoms.GetCount():
            # Nothing to do:
            return

        atoms = self.atoms()
        pos_ac, Z_a, cell_c, pbc_c = self.last_atomic_configuration

        if (atoms.GetAtomicNumbers() != Z_a or
            atoms.GetUnitCell() / self.a0 != cell_cc or
            atoms.GetBoundaryConditions() != pbc_c):
            # Drastic changes:
            self.initialize()
            self.find_ground_state()
            return

        # Something else has changed:
        if atoms.GetCartesianPositions() / self.a0 != pos_ac:
            # It was the positions:
            self.find_ground_state()
        else:
            # It was something that we don't care about - like
            # velocities, masses, ...
            pass
        
    def get_atoms(self):
        assert not hasattr(self, 'atoms')
        pos_ac, Z_a, cell_c, pbc_c = self.last_atomic_configuration
        magmom_a, tag_a = self.extra_list_of_atoms_stuff
        atoms = ListOfAtoms([Atom(Z, pos_c, tag=tag, magmom=magmom)
                             for Z, pos_c, tag, magmom in
                             zip(Z_a, pos_ac, tag_a, magmom_a)],
                            cell=cell_c, perisodic=pbc_c)
        self.atoms = weakref.ref(atoms)
        atoms.calculator = self
        return atoms


    def find_ground_state(self):
        """Start iterating towards the ground state."""

        self.set_positions()

        if not self.wave_functions_initialized:
            self.initialize_wave_functions()

        self.hamiltonian.update()

        # Self-consistency loop:
        while not self.converged:
            if self.niter > 120:
                raise ConvergenceError('Did not converge!')
            self.step()
            self.add_up_energies()
            self.check_convergence()
            self.call()
            self.print_iteration()
            self.niter += 1

        self.call(final=True)
        output.print_converged(self)

        # Save the state of the atoms:
        atoms = self.atoms()
        self.count = atoms.GetCount()
        self.last_atomic_configuration = (
            atoms.GetCartesianPositions() / self.a0,
            atoms.GetAtomicNumbers(),
            atoms.GetUnitCell() / self.a0,
            atoms.GetBoundaryConditions())

    def step(self):
        if self.niter > 2:
            self.density.update()
            self.hamiltonian.update()

        self.eigensolver.iterate(self.hamiltonian, self.kpt_u)

        # Make corrections due to non-local xc:
        xcfunc = self.hamiltonian.xc.xcfunc
        self.Enlxc = xcfunc.get_non_local_energy()
        self.Enlkin = xcfunc.get_non_local_kinetic_corrections()

        # Calculate occupation numbers:
        self.occupation.calculate(self.kpt_u)

    def add_up_energies(self):
        H = self.hamiltonian
        self.Etot = (H.Ekin0 + self.occupation.Eband + self.Enlkin +
                     H.Epot + H.Ebar + self.Eext +
                     self.Exc + self.Enlxc -
                     self.S)

    def set_positions(self):
        """Update the positions of the atoms.

        Localized functions centered on atoms that have moved will
        have to be computed again.  Neighbor list is updated and the
        array holding all the pseudo core densities is updated."""

        pos_ac = self.atoms().GetCartesianPositions() / self.a0

        movement = False
        for nucleus, pos_c in zip(self.nuclei, pos_ac):
            spos_c = self.domain.scale_position(pos_c)
            if num.sometrue(spos_c != nucleus.spos_c) or not nucleus.ready:
                movement = True
                nucleus.set_position(spos_c, self.domain, self.my_nuclei,
                                     self.nspins, self.nmyu, self.nbands)
                nucleus.move(spos_c, self.gd, self.finegd,
                             self.ibzk_kc, self.locfuncbcaster,
                             self.domain,
                             self.pt_nuclei, self.ghat_nuclei)

        if movement:
            self.niter = 0
            self.converged = False
            self.F_ac = None

            self.locfuncbcaster.broadcast()

            for nucleus in self.nuclei:
                nucleus.normalize_shape_function_and_pseudo_core_density()

            if self.symmetry:
                self.symmetry.check(pos_ac)

            self.hamiltonian.pairpot.update(pos_ac, self.nuclei, self.domain)

            self.density.move()

    def initialize_wave_functions_from_atomic_orbitals(self):
        """Initialize wave function from atomic orbitals."""  # surprise!
        
        # count the total number of atomic orbitals (bands):
        nao = 0
        for nucleus in self.nuclei:
            nao += nucleus.get_number_of_atomic_orbitals()

        if self.random_wf:
            nao = 0

        nrandom = max(0, self.nbands - nao)

        if self.nbands == 1:
            string = 'Initializing one band from'
        else:
            string = 'Initializing %d bands from' % self.nbands
        if nao == 1:
            string += ' one atomic orbital'
        elif nao > 0:
            string += ' linear combination of %d atomic orbitals' % nao

        if nrandom > 0 :
            if nao > 0:
                string += ' and'
            string += ' %d random orbitals' % nrandom
        string += '.'

        print >> self.out, string


        xcfunc = self.hamiltonian.xc.xcfunc

        if xcfunc.hybrid > 0.0:
            # At this point, we can't use orbital dependent
            # functionals, because we don't have the right orbitals
            # yet.  So we use a simple density functional to set up the
            # initial hamiltonian:
            if xcfunc.xcname == 'EXX':
                localxcfunc = XCFunctional('LDAx', self.nspins)
            else:
                assert xcfunc.xcname == 'PBE0'
                localxcfunc = XCFunctional('PBE', self.nspins)
            self.hamiltonian.xc.set_functional(localxcfunc)
            for setup in self.setups:
                setup.xc_correction.xc.set_functional(localxcfunc)

        self.Ekin0, self.Epot, self.Ebar, self.Eext, self.Exc = \
            self.hamiltonian.update(self.density)

        if self.random_wf:
            for kpt in self.kpt_u:
                kpt.create_random_orbitals(self.nbands)
                # Calculate projections and orthogonalize wave functions:
                run([nucleus.calculate_projections(kpt)
                     for nucleus in self.pt_nuclei])
                kpt.orthonormalize(self.my_nuclei)
            # Improve the random guess with conjugate gradients
            eig = CG(self.timer,self.kpt_comm,
                     self.gd, self.hamiltonian.kin,
                     self.typecode, self.nbands)
            eig.set_convergence_criteria(True, 1e-2, self.nvalence)
            for nit in range(2):
                eig.iterate(self.hamiltonian, self.kpt_u)

        else:
            for nucleus in self.my_nuclei:
                # XXX already allocated once, but with wrong size!!!
                ni = nucleus.get_number_of_partial_waves()
                nucleus.P_uni = num.empty((self.nmyu, nao, ni), self.typecode)

            # Use the generic eigensolver for subspace diagonalization
            eig = Eigensolver(self.timer,self.kpt_comm,
                              self.gd, self.hamiltonian.kin,
                              self.typecode, nao)
            for kpt in self.kpt_u:
                kpt.create_atomic_orbitals(nao, self.nuclei)
                # Calculate projections and orthogonalize wave functions:
                run([nucleus.calculate_projections(kpt)
                     for nucleus in self.pt_nuclei])
                kpt.orthonormalize(self.my_nuclei)
                eig.diagonalize(self.hamiltonian, kpt)


        for nucleus in self.my_nuclei:
            nucleus.reallocate(self.nbands)

        for kpt in self.kpt_u:
            kpt.adjust_number_of_bands(self.nbands,
                                       self.pt_nuclei, self.my_nuclei)

        if xcfunc.hybrid > 0:
            # Switch back to the orbital dependent functional:
            self.hamiltonian.xc.set_functional(xcfunc)
            for setup in self.setups:
                setup.xc_correction.xc.set_functional(xcfunc)


        # Calculate occupation numbers:
        self.nfermi, self.magmom, self.S, Eband = \
            self.occupation.calculate(self.kpt_u)

        self.wave_functions_initialized = True


    def initialize_wave_functions(self):
        if not self.wave_functions_initialized:
            # Initialize wave functions from atomic orbitals:
            for nucleus in self.nuclei:
                nucleus.initialize_atomic_orbitals(self.gd, self.ibzk_kc,
                                                   self.locfuncbcaster)
            self.locfuncbcaster.broadcast()

            if not self.density.initialized:
                self.density.initialize()

            self.initialize_wave_functions_from_atomic_orbitals()

            self.converged = False

            # Free allocated space for radial grids:
            for setup in self.setups:
                del setup.phit_j
            for nucleus in self.nuclei:
                try:
                    del nucleus.phit_i
                except AttributeError:
                    pass

        elif not isinstance(self.kpt_u[0].psit_nG, num.ArrayType):
            # Calculation started from a restart file.  Copy data
            # from the file to memory:
            for kpt in self.kpt_u:
                kpt.psit_nG = kpt.psit_nG[:]

        for kpt in self.kpt_u:
            kpt.adjust_number_of_bands(self.nbands, self.pt_nuclei,
                                       self.my_nuclei)


    def calculate_forces(self,silent=False):
        """Return the atomic forces."""

        if self.F_ac is not None:
            retutrn

        nt_g = self.density.nt_g
        vt_sG = self.hamiltonian.vt_sG
        vHt_g = self.hamiltonian.vHt_g

        if self.nspins == 2:
            vt_G = 0.5 * (vt_sG[0] + vt_sG[1])
        else:
            vt_G = vt_sG[0]

        for nucleus in self.my_nuclei:
            nucleus.F_c[:] = 0.0

        # Calculate force-contribution from k-points:
        for kpt in self.kpt_u:
            for nucleus in self.pt_nuclei:
                nucleus.calculate_force_kpoint(kpt)
        for nucleus in self.my_nuclei:
            self.kpt_comm.sum(nucleus.F_c)

        for nucleus in self.nuclei:
            nucleus.calculate_force(vHt_g, nt_g, vt_G)

        # Global master collects forces from nuclei into self.F_ac:
        if mpi.rank == MASTER:
            for a, nucleus in enumerate(self.nuclei):
                if nucleus.in_this_domain:
                    self.F_ac[a] = nucleus.F_c
                else:
                    self.domain.comm.receive(self.F_ac[a], nucleus.rank, 7)
        else:
            if self.kpt_comm.rank == 0:
                for nucleus in self.my_nuclei:
                    self.domain.comm.send(nucleus.F_c, MASTER, 7)

        # Broadcast the forces to all processors
        mpi.world.broadcast(self.F_ac, MASTER)

        if self.symmetry is not None:
            # Symmetrize forces:
            F_ac = num.zeros((len(self.nuclei), 3), num.Float)
            for map_a, symmetry in zip(self.symmetry.maps,
                                       self.symmetry.symmetries):
                swap, mirror = symmetry
                for a1, a2 in enumerate(map_a):
                    F_ac[a2] += num.take(self.F_ac[a1] * mirror, swap)
            self.F_ac[:] = F_ac / len(self.symmetry.symmetries)

        if mpi.rank == MASTER and not silent:
            for a, nucleus in enumerate(self.nuclei):
                print >> self.out, 'forces ', \
                    a, nucleus.setup.symbol, self.F_ac[a] * c

    def attach(self, function, iters, *args, **kwargs):
        self.callback_functions.append((function, iters, args, kwargs))

    def call(self, niter, final=False):
        for function, n, args, kwargs in self.callback_functions:
            if ((niter % n) == 0) != final:
                function(*args, **kwargs)

    def create_setups(self, setup_types, Z_a):
        if isinstance(setup_types, str):
            setup_types = {None: setup_types}
        
        # setup_types is a dictionary mapping chemical symbols and atom
        # numbers to setup types.
        
        # If present, None will map to the default type:
        default = setup_types.get(None, 'paw')
        
        type_a = [default] * self.natoms
        
        # First symbols ...
        for symbol, type in setup_types.items():
            if isinstance(symbol, str):
                number = numbers[symbol]
                for a, Z in enumerate(Z_a):
                    if Z == number:
                        type_a[a] = type
        
        # and then atom numbers:
        for a, type in setup_types.items():
            if isinstance(a, int):
                type_a[a] = type
        
        # Construct necessary PAW-setup objects:
        self.setups = {}
        for a, (Z, type) in enumerate(zip(Z_a, type_a)):
            if (Z, type) not in self.setups:
                symbol = symbols[Z]
                setup = create_setup(symbol, xcfunc, lmax, nspins, softgauss,
                                     type)
                setup.print_info(out)
                self.setups[(Z, type)] = setup
        
        return type_a

    def read_parameters(self, filename):
        """Read state from file."""

        r = gpaw.io.open(filename, 'r')
        p = self.input_parameters

        p['setups'] = setup_types
        p['xc'] = r['XCFunctional']
        p['nbands'] = r.dimension('nbands')
        p['spinpol'] = (r.dimension('nspins') == 2)
        p['kpts'] = r.get('BZKPoints')
        p['usesymm'] = bool(r['UseSymmetry'])  # numpy!
        p['gpts'] = ((r.dimension('ngptsx') + 1) // 2 * 2
                     (r.dimension('ngptsy') + 1) // 2 * 2
                     (r.dimension('ngptsz') + 1) // 2 * 2)
        p['lmax'] = r['MaximumAngularMomentum']
        p['setups'] = eval(r['SetupTypes'])
        p['stencils'] = (r['KohnShamStencil'],
                         r['PoissonStencil'],
                         r['InterpolationStencil'])
        p['charge'] = r['Charge']
        p['fixmom'] = r['FixedMagneticMoment']
        p['fixdensity'] = bool(r['FixDensity'])  # numpy!
        p['tolerance'] = r['Tolerance']
        p['convergeall'] = bool(r['ConvergeEmptyStates'])
        p['width'] = r['FermiWidth'] 

        self.converged = bool(r['Converged'])

        return r
    
    def reset(self, restart_file=None):
        """Delete PAW-object."""
        self.stop_paw()
        self.restart_file = restart_file
        self.pos_ac = None
        self.cell_cc = None
        self.periodic_c = None
        self.Z_a = None

    def set_h(self, h):
        self.gpts = None
        self.h = h
        self.reset()
     
    def set_gpts(self, gpts):
        self.h = None
        self.gpts = gpts
        self.reset()
     
    
    def set_convergence_criteria(self, tol):
        """Set convergence criteria.

        Stop iterating when the size of the residuals are below
        ``tol``."""

        self.tolerance = tol
        #???self.maxiter ...

    def check_convergence(self):
        self.converged = (self.eigensolver.error < self.tolerance or
                          self.niter >= self.maxiter)
        return self.converged
    
    def __del__(self):
        """Destructor:  Write timing output before closing."""
        self.timer.write(self.out)

    def totype(self, typecode):
        """Converts all the typecode dependent quantities of Paw
        (Laplacian, wavefunctions etc.) to typecode"""

        from gpaw.operators import Laplace

        if typecode not in [num.Float, num.Complex]:
            raise RuntimeError('PAW can be converted only to Float or Complex')

        self.typecode = typecode

        # Hamiltonian
        nn = self.stencils[0]
        self.hamiltonian.kin = Laplace(self.gd, -0.5, nn, typecode)

        # Nuclei
        for nucleus in self.nuclei:
            nucleus.typecode = typecode
            nucleus.reallocate(self.nbands)
            nucleus.ready = False

        self.set_positions()

        # Wave functions
        for kpt in self.kpt_u:
            kpt.typecode = typecode
            kpt.psit_nG = num.array(kpt.psit_nG[:], typecode)

        # Eigensolver
        # !!! FIX ME !!!
        # not implemented yet...

    def distribute_kpoints_and_spins(self, parsize_c, N_c):
        """Distribute k-points/spins to processors.

        Construct communicators for parallelization over
        k-points/spins and for parallelization using domain
        decomposition."""
        
        ntot = self.nspins * self.nkpts
        size = mpi.size
        rank = mpi.rank

        if parsize_c is None:
            ndomains = size // gcd(ntot, size)
        else:
            ndomains = parsize_c[0] * parsize_c[1] * parsize_c[2]

        r0 = (rank // ndomains) * ndomains
        ranks = range(r0, r0 + ndomains)
        domain_comm = new_communicator(ranks)
        self.domain.set_decomposition(domain_comm, parsize_c, N_c)

        r0 = rank % ndomains
        ranks = range(r0, r0 + size, ndomains)
        self.kpt_comm = new_communicator(ranks)
