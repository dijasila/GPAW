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
import ASE

from gpaw.utilities import DownTheDrain, check_unit_cell
from gpaw.utilities.memory import maxrss
from gpaw.mpi.paw import MPIPaw
from gpaw.startup import create_paw_object
from gpaw.version import version
import gpaw.utilities.timing as timing
import gpaw
import gpaw.io
import gpaw.mpi as mpi
from gpaw import parallel

        import Numeric as num
from ASE.ChemicalElements.symbol import symbols
from ASE.ChemicalElements import numbers
from ASE.Units import Convert

import sys

from gpaw.nucleus import Nucleus
from gpaw.rotation import rotation
from gpaw.domain import Domain
from gpaw.symmetry import Symmetry
from gpaw.paw import Paw
from gpaw.xc_functional import XCFunctional
from gpaw.utilities import gcd
import gpaw.mpi as mpi
from gpaw.utilities.timing import Timer
from gpaw.utilities.memory import estimate_memory
from gpaw.setup import create_setup
            

from gpaw import dry_run
MASTER = 0


class Calculator:
    """This is the ASE-calculator frontend for doing a PAW calculation.

    The calculator object controls a paw object that does the actual
    work.  The paw object can run in serial or in parallel, the
    calculator interface will allways be the same."""

#    write = ('gpaw-restart.gpw', 0)
    def attach(self, iters, function, *args, **kwargs):
        self.callbacks.append((iters, function, args, kwargs))
        for iters, function, args, kwargs in self.callbacks:
            if 0:
                function(*args, **kwargs)

    def __init__(self, filename=None, **kwargs):
        """ASE-calculator interface.

        The following parameters can be used: `nbands`, `xc`, `kpts`,
        `spinpol`, `gpts`, `h`, `charge`, `usesymm`, `width`, `mix`,
        `hund`, `lmax`, `fixdensity`, `tolerance`, `out`,
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

        self.a0 = Convert(1, 'Bohr', units.GetLengthUnit())
        self.Ha = Convert(1, 'Hartree', units.GetEnergyUnit())

        self.xc = 'LDA'
        self.nbands = None
        self.spinpol = None
        self.kpts = None
        self.usesymm = True
        self.gpts = None
        self.h = None
        self.lmax = 2
        self.setups = 'paw'
        self.stencils = (2, 3)
        self.charge = 0
        self.fixmom = False
        self.width = None
        self.fixdensity = False
        self.tolerance = 1.0e-9
        self.convergeall = False

        self.random = False 
        self.hund = False
        self.mix = (0.25, 3, 1.0)
        self.out = '-'
        self.verbosity = 0
        self.decompose = None
        self.eigensolver = RMM_DIIS
        self.poissonsolver = GaussSeidel
        self.external = None

        self.converged = False
        
        if filename is not None:
            self.read_parameters(filename)

        if 'h' in kwargs and 'gpts' in kwargs:
            raise TypeError("""You can't use both "gpts" and "h"!""")
            
        # Convert from ASE units:
        if 'h' in kwargs:
            kwargs['h'] /= self.a0
        if 'width' in kwargs:
            kwargs['width'] /= self.Ha
        if 'external' in kwargs:
            kwargs['external'] = kwargs['external'] / self.Ha
            
        for name, value in kwargs.items():
            if name in ['random', 'hund', 'mix', 'out', 'verbosity',
                        'decompose', 'eigensolver', 'poissonsolver',
                        'external']:
                setattr(self, name, value)
            elif name in ['xc', 'nbands', 'spinpol', 'kpts', 'usesymm',
                          'gpts', 'h', 'width', 'lmax', 'setups', 'stencils',
                          'charge', 'fixmom', 'fixdensity', 'tolerance',
                          'convergeall']:
                self.converged = False
                setattr(self, name, value)
            else:
                raise RuntimeError('Unknown keyword: ' + name)

        if filename is not None:
            self.start()

    def start(self):

        out, verbosity, write, a0, Ha,
                      pos_ac, Z_a, magmom_a, cell_c, bc_c,
                      h, N_c, xcname,
                      nbands, spinpol, kT,
                      charge, random,
                      bzk_kc,
                      softgauss, stencils, usesymm, mix, fixdensity,
                      hund, fixmom, lmax, tolerance, maxiter,
                      convergeall, eigensolver, relax, setup_types,
                      parsize_c,
                      restart_file, vext_g):

        self.timer = Timer()
        timer.start('Init')
        
        magmom_a = num.array(magmom_a)
        magnetic = num.sometrue(magmom_a)
        
        # Is this a gamma-point calculation?
        gamma = (len(bzk_kc) == 1 and not num.sometrue(bzk_kc[0]))
        
        # Default values:
        if spinpol is None:
            spinpol = magnetic
        if hund and (not spinpol or len(Z_a) != 1):
            hund = False
        
        fixmom = fixmom and spinpol
        fixmom = hund or fixmom
        
        if kT is None:
            if gamma:
                kT = 0
            else:
                kT = Convert(0.1, 'eV', 'Hartree') * Ha
        
        if magnetic and not spinpol:
            raise ValueError('Non-zero initial magnetic moment for a ' +
                             'spin-paired calculation!')
        
        xcfunc = XCFunctional(xcname)
        
        if spinpol:
            nspins = 2
        else:
            nspins = 1
        
        # Default value for grid spacing:
        if N_c is None:
            if h is None:
                print >> out, 'Using default value for grid spacing.'
                h = Convert(0.2, 'Ang', 'Bohr') * a0
            # N_c should be a multiplum of 4:
            N_c = [max(4, int(L / h / 4 + 0.5) * 4) for L in cell_c]
        N_c = num.array(N_c)
        
        
        # Create a Domain object:
        domain = Domain(cell_c / a0, bc_c)
        h_c = domain.cell_c / N_c
        
        if isinstance(setup_types, str):
            setup_types = {None: setup_types}
        
        # setup_types is a dictionary mapping chemical symbols and atom
        # numbers to setup types.
        
        # If present, None will map to the default type:
        default = setup_types.get(None, 'paw')
        
        type_a = [default] * len(Z_a)
        
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
        setups = {}
        for a, (Z, type) in enumerate(zip(Z_a, type_a)):
            if (Z, type) not in setups:
                symbol = symbols[Z]
                setup = create_setup(symbol, xcfunc, lmax, nspins, softgauss, type)
                setup.print_info(out)
                setups[(Z, type)] = setup
        
        # Brillouin-zone stuff:
        if gamma:
            typecode = num.Float
            symmetry = None
            weights_k = [1.0]
            ibzk_kc = num.zeros((1, 3), num.Float)
            nkpts = 1
            print >> out, 'Gamma-point calculation'
        else:
            typecode = num.Complex
            # Reduce the the k-points to those in the irreducible part of
            # the Brillouin zone:
            symmetry, weights_k, ibzk_kc = reduce_kpoints(
                bzk_kc, pos_ac / a0, Z_a, type_a, magmom_a, domain, usesymm)
        
            if symmetry is not None:
                symmetry.print_symmetries(out)
        
            nkpts = len(ibzk_kc)
            print >> out
            print >> out, (('%d k-point%s in the irreducible part of the ' +
                           'Brillouin zone (total: %d)') %
                           (nkpts, ' s'[1:nkpts], len(bzk_kc)))
            print >> out
        
        # Build list of nuclei:
        nuclei = []
        for a, (Z, type) in enumerate(zip(Z_a, type_a)):
            nuclei.append(Nucleus(setups[(Z, type)], a, typecode))
            
        setups = setups.values()
        
        if usesymm and symmetry is not None:
            # Find rotation matrices for spherical harmonics:
            R_slmm = [[rotation(l, symm) for l in range(3)]
                        for symm in symmetry.symmetries]
        
            for setup in setups:
                setup.calculate_rotations(R_slmm)
        
        # Sum up the number of valence electrons:
        nvalence = 0
        nao = 0
        for nucleus in nuclei:
            nvalence += nucleus.setup.Nv
            nao += nucleus.setup.niAO
        nvalence -= charge
        
        if nvalence < 0:
            raise ValueError(
                'Charge %f is not possible - not enough valence electrons' %
                charge)
        
        if nbands is None:
            # Default value for number of bands:
            #nbands = (nvalence + 7) // 2 + int(num.sum(magmom_a) / 2)
            nbands = nao
        elif nbands <= 0:
            nbands = (nvalence + 1) // 2 + (-nbands)
            
        if nvalence > 2 * nbands:
            raise ValueError('Too few bands!')
        
        # Get the local number of spins and k-points, and return a
        # domain_comm and kpt_comm for this processor:
        domain_comm, kpt_comm = distribute_kpoints_and_spins(nspins, nkpts,
                                                             parsize_c)
        
        domain.set_decomposition(domain_comm, parsize_c, N_c)
        
        timer.stop()
        # We now have all the parameters needed to construct a PAW object:
        paw = Paw(a0, Ha,
                  setups, nuclei, domain, N_c, symmetry, xcfunc,
                  nvalence, charge, nbands, nspins, random, 
                  typecode, bzk_kc, ibzk_kc, weights_k,
                  stencils, usesymm, mix, fixdensity, maxiter,
                  convergeall, eigensolver, relax, pos_ac / a0, timer, kT / Ha,
                  tolerance, kpt_comm, restart_file, hund, fixmom, magmom_a,
                  out, verbosity, write, vext_g)
        
        if dry_run:
            # Estimate the amount of memory needed
            estimate_memory(N_c, nbands, nkpts, nspins, typecode, nuclei, h_c, out)
            out.flush()
            sys.exit()
        
        return paw

    
def reduce_kpoints(bzk_kc, pos_ac, Z_a, type_a, magmom_a, domain, usesymm):
    """Reduce the number of k-points using symmetry.

    Returns symmetry object, weights and k-points in the irreducible
    part of the BZ."""

    for c in range(3):
        if not domain.periodic_c[c] and num.sometrue(bzk_kc[:, c]):
            raise ValueError('K-points can only be used with PBCs!')

    # Construct a Symmetry instance containing the identity
    # operation only:
    symmetry = Symmetry(Z_a, type_a, magmom_a, domain)

    if usesymm:
        # Find symmetry operations of atoms:
        symmetry.analyze(pos_ac)

    # Reduce the set of k-points:
    ibzk_kc, weights_k = symmetry.reduce(bzk_kc)

    if usesymm:
        symmetry = symmetry
    else:
        symmetry = None

    return symmetry, weights_k, ibzk_kc

def new_communicator(ranks):
    if len(ranks) == 1:
        return mpi.serial_comm
    elif len(ranks) == mpi.size:
        return mpi.world
    else:
        return mpi.world.new_communicator(num.array(ranks))


def distribute_kpoints_and_spins(nspins, nkpts, parsize_c):
    ntot = nspins * nkpts
    size = mpi.size
    rank = mpi.rank

    if parsize_c is None:
        ndomains = size // gcd(ntot, size)
    else:
        ndomains = parsize_c[0] * parsize_c[1] * parsize_c[2]

    r0 = (rank // ndomains) * ndomains
    ranks = range(r0, r0 + ndomains)
    domain_comm = new_communicator(ranks)

    r0 = rank % ndomains
    ranks = range(r0, r0 + size, ndomains)
    kpt_comm = new_communicator(ranks)

    return domain_comm, kpt_comm

        self.t0 = time.time()
    
        self.paw = None
        self.restart_file = None

        # Set default parameters and adjust with user parameters:
        self.Set(**Calculator.parameters)
        self.Set(**kwargs)
        
        out = self.out
        print >> out
        print >> out, '  ___ ___ ___ _ _ _  '
        print >> out, ' |   |   |_  | | | | '
        print >> out, ' | | | | | . | | | | '
        print >> out, ' |__ |  _|___|_____| ', version
        print >> out, ' |___|_|             '
        print >> out

        uname = os.uname()
        print >> out, 'User:', os.getenv('USER') + '@' + uname[1]
        print >> out, 'Date:', time.asctime()
        print >> out, 'Arch:', uname[4]
        print >> out, 'Pid: ', os.getpid()
        print >> out, 'Dir: ', os.path.dirname(gpaw.__file__)
        print >> out, 'ASE: ', os.path.dirname(ASE.__file__)
        print >> out

        lengthunit = units.GetLengthUnit()
        energyunit = units.GetEnergyUnit()
        print >> out, 'units:', lengthunit, 'and', energyunit
        self.a0 = Convert(1, 'Bohr', lengthunit)
        self.Ha = Convert(1, 'Hartree', energyunit)

        self.reset()

        self.tempfile = None

        self.parallel_cputime = 0.0

    def read_parameters(self, filename):
        """Read state from file."""

        r = gpaw.io.open(filename, 'r')

        self.setups = setup_types
        self.xc = r['XCFunctional']
        self.nbands = r.dimension('nbands')
        self.spinpol = (r.dimension('nspins') == 2)
        self.kpts = r.get('BZKPoints')
        self.usesymm = bool(r['UseSymmetry'])  # numpy!
        self.gpts = ((r.dimension('ngptsx') + 1) // 2 * 2
                     (r.dimension('ngptsy') + 1) // 2 * 2
                     (r.dimension('ngptsz') + 1) // 2 * 2)
        self.lmax = r['MaximumAngularMomentum']
        self.setups = eval(r['SetupTypes'])
        self.stencils = ...
        self.charge = ...
        self.fixmom = ...
        self.fixdensity = bool(r['FixDensity'])  # numpy!
        self.tolerance = r['Tolerance']
        self.convergeall = bool(r['ConvergeEmptyStates'])
        self.width = r['FermiWidth'] 
        self.converged = bool(r['Converged'])

        Z_a = num.asarray(r.get('AtomicNumbers'), num.Int)
        pos_ac = r.get('CartesianPositions') * a0
        periodic_c = r.get('BoundaryConditions')
        cell_cc = r.get('UnitCell') * a0
        atoms = ASE.ListOfAtoms([ASE.Atom(Z=Z,
                                          position=pos,
                                          magmom=magmom,
                                          tag=tag)
                                 for Z, pos, magmom, tag in
                                 zip(Z_a,
                                     pos_ac,
                                     r.get('MagneticMoments'),
                                     r.get('Tags'))],
                                periodic=periodic_c,
                                cell=cell_cc)

        atoms.SetCalculator(calc)

        # Wave functions and other stuff will be read from 'filename'
        # later, when requiered:
        calc.restart_file = filename
        calc.initialize_paw_object()

        # Get the forces from the old calculation:
        calc.paw.set_forces(r.get('CartesianForces'))

        r.close()

        calc.lastcount = atoms.GetCount()
        calc.Z_a = Z_a
        calc.pos_ac = pos_ac
        calc.periodic_c = periodic_c
        calc.cell_cc = cell_cc
        
        return atoms

    # Make ReadAtoms a static method:
    ReadAtoms = staticmethod(ReadAtoms)

    def reset(self, restart_file=None):
        """Delete PAW-object."""
        self.stop_paw()
        self.restart_file = restart_file
        self.pos_ac = None
        self.cell_cc = None
        self.periodic_c = None
        self.Z_a = None

    def set_out(self, out):
        """Set the stream for text output.

        If `out` is not a stream-object, then it must be one of:

        ``None``:
          Throw output away.
        ``'-'``:
          Use standard-output (``sys.stdout``).
        A filename:
          open a new file.
        """
        
        if out is None or mpi.rank != MASTER:
            out = DownTheDrain()
        elif out == '-':
            out = sys.stdout
        elif isinstance(out, str):
            out = open(out, 'w')
        self.out = out

    def set_kpts(self, bzk_kc):
        """Set the scaled k-points. 
        
        ``kpts`` should be an array of scaled k-points."""
        
        if bzk_kc is None:
            bzk_kc = (1, 1, 1)
        if isinstance(bzk_kc[0], int):
            bzk_kc = MonkhorstPack(bzk_kc)
        self.bzk_kc = num.array(bzk_kc)
        self.reset(self.restart_file)

    def set_h(self, h):
        self.gpts = None
        self.h = h
        self.reset()
     
    def set_gpts(self, gpts):
        self.h = None
        self.gpts = gpts
        self.reset()
     
    def update(self):
        """Update PAW calculaton if needed."""
        atoms = self.atoms()

        if self.paw is not None and self.lastcount == atoms.GetCount():
            # Nothing to do:
            return

        if (self.paw is None or
            atoms.GetAtomicNumbers() != self.Z_a or
            atoms.GetUnitCell() != self.cell_cc or
            atoms.GetBoundaryConditions() != self.periodic_c):
            # Drastic changes:
            self.reset(self.restart_file)
            self.initialize_paw_object()
            self.find_ground_state()
        else:
            # Something else has changed:
            if (atoms.GetCartesianPositions() != self.pos_ac):
                # It was the positions:
                self.find_ground_state()
            else:
                # It was something that we don't care about - like
                # velocities, masses, ...
                pass
    
    def initialize_paw_object(self):
        """Initialize PAW-object."""
        atoms = self.atoms()

        pos_ac = atoms.GetCartesianPositions()
        Z_a = atoms.GetAtomicNumbers()
        cell_cc = num.array(atoms.GetUnitCell())
        periodic_c = atoms.GetBoundaryConditions()
        
        # Check that the cell is orthorhombic:
        check_unit_cell(cell_cc)
        # Get the diagonal:
        cell_c = num.diagonal(cell_cc)

        magmoms = [atom.GetMagneticMoment() for atom in atoms]

        # Get rid of the old calculator before the new one is created:
        self.stop_paw()

        # Maybe parsize has been set by command line argument
        # --gpaw-domain-decomposition? (see __init__.py)
        if gpaw.parsize is not None:
            # Yes, it was:
            self.parsize = gpaw.parsize

        if self.external is not None:
            self.external /= self.Ha
        args = [self.out,
                self.verbosity,
                self.write,
                self.a0, self.Ha,
                pos_ac, Z_a, magmoms, cell_c, periodic_c,
                self.h, self.gpts, self.xc,
                self.nbands, self.spinpol, self.width,
                self.charge,
                self.random,
                self.bzk_kc,
                self.softgauss,
                self.stencils,
                self.usesymm,
                self.mix,
                self.fixdensity,
                self.hund,
                self.fixmom,
                self.lmax,
                self.tolerance,
                self.maxiter,
                self.convergeall,
                self.eigensolver,
                self.relax,
                self.setups,
                self.parsize,
                self.restart_file,
                self.external,
                ]

        if gpaw.hosts is not None:
            # The hosts have been set by one of the command line arguments
            # --gpaw-hosts or --gpaw-hostfile (see __init__.py):
            self.hosts = gpaw.hosts

        if self.hosts is None:
            if os.environ.has_key('PBS_NODEFILE'):
                # This job was submitted to the PBS queing system.  Get
                # the hosts from the PBS_NODEFILE environment variable:
                self.hosts = os.environ['PBS_NODEFILE']
                
                try:
                    nodes = len(open(self.hosts).readlines())
                    if nodes == 1:
                        self.hosts = None
                except:
                    pass
            elif os.environ.has_key('NSLOTS'):
                # This job was submitted to the Grid Engine queing system:
                self.hosts = int(os.environ['NSLOTS'])
            elif os.environ.has_key('LOADL_PROCESSOR_LIST'):
                self.hosts = 'dummy file-name'
            #elif os.environ.has_key('GPAW_MPI_COMMAND'):
            #    self.hosts = 'dummy file-name'

        if isinstance(self.hosts, int):
            if self.hosts == 1:
                # Only one node - don't do a parallel calculation:
                self.hosts = None
            else:
                self.hosts = [os.uname()[1]] * self.hosts
            
        if isinstance(self.hosts, list):
            # We need the hosts in a file:
            fd, self.tempfile = tempfile.mkstemp('.hosts')
            for host in self.hosts:
                os.write(fd, host + '\n')
            os.close(fd)
            self.hosts = self.tempfile
            # (self.tempfile is removed in Calculator.__del__)

        if self.hosts is not None and mpi.size == 1:
            parallel_with_sockets = True
        else:
            parallel_with_sockets = False

        # What kind of calculation should we do?
        if parallel_with_sockets:
            self.paw = MPIPaw(self.hosts, *args)
        else:
            self.paw = create_paw_object(*args)
            
    def find_ground_state(self):
        """Tell PAW-object to start iterating ..."""
        atoms = self.atoms()
        pos_ac = atoms.GetCartesianPositions()
        Z_a = atoms.GetAtomicNumbers()
        cell_cc = atoms.GetUnitCell()
        periodic_c = atoms.GetBoundaryConditions()
        
        # Check that the cell is orthorhombic:
        check_unit_cell(cell_cc)

        self.paw.find_ground_state(pos_ac, num.diagonal(cell_cc))
        
        # Save the state of the atoms:
        self.lastcount = atoms.GetCount()
        self.pos_ac = pos_ac
        self.cell_cc = cell_cc
        self.periodic_c = periodic_c
        self.Z_a = Z_a

        timing.update()

    def stop_paw(self):
        """Delete PAW-object."""
        if isinstance(self.paw, MPIPaw):
            # Stop old MPI calculation and get total CPU time for all CPUs:
            self.parallel_cputime += self.paw.stop()
        self.paw = None
        
    def __del__(self):
        """Destructor:  Write timing output before closing."""
        if self.tempfile is not None:
            # Delete hosts file:
            os.remove(self.tempfile)

        self.stop_paw()
        
        # Get CPU time:
        c = self.parallel_cputime + timing.clock()
                
        if c > 1.0e99:
            print >> self.out, 'cputime : unknown!'
        else:
            print >> self.out, 'cputime : %f' % c

        print >> self.out, 'walltime: %f' % (time.time() - self.t0)
        mr = maxrss()
        if mr > 0:
            def round(x): return int(100*x/1024.**2+.5)/100.
            print >> self.out, 'memory  : '+str(round(maxrss()))+' MB'
        print >> self.out, 'date    :', time.asctime()

    #####################
    ## User interface: ##
    #####################
    def Set(self, **kwargs):
        """Set keyword parameters.

        Works like this:

        >>> calc.Set(out='stuff.txt')
        >>> calc.Set(nbands=24, spinpol=True)

        """

            
    def GetReferenceEnergy(self):
        """Get reference energy for all-electron atoms."""
        return self.paw.Eref * self.Ha

    def GetEnsembleCoefficients(self):
        """Get BEE ensemble coefficients.

        See The ASE manual_ for details.

        .. _manual: https://wiki.fysik.dtu.dk/ase/Utilities
                    #bayesian-error-estimate-bee
        """

        E = self.GetPotentialEnergy()
        E0 = self.GetXCDifference('XC-9-1.0')
        coefs = (E + E0,
                 self.GetXCDifference('XC-0-1.0') - E0,
                 self.GetXCDifference('XC-1-1.0') - E0,
                 self.GetXCDifference('XC-2-1.0') - E0)
        print >> self.out, 'BEE: (%.9f, %.9f, %.9f, %.9f)' % coefs
        return num.array(coefs)

    def GetXCDifference(self, xcname):
        """Calculate non-seflconsistent XC-energy difference."""
        self.update()
        return self.paw.get_xc_difference(xcname)

    def Write(self, filename, mode='all'):
        """Write current state to file."""
        pos_ac = self.atoms().GetCartesianPositions()
        magmom_a = self.atoms().GetMagneticMoments()
        tag_a = self.atoms().GetTags()
        self.paw.write_state_to_file(filename, pos_ac, magmom_a, tag_a, mode,
                                     self.setups)
        
    def GetNumberOfIterations(self):
        """Return the number of SCF iterations."""
        return self.paw.niter

    ####################
    ## ASE interface: ##
    ####################
    def GetPotentialEnergy(self, force_consistent=False):
        """Return the energy for the current state of the ListOfAtoms."""
        self.update()
        return self.paw.get_total_energy(force_consistent)

    def GetCartesianForces(self):
        """Return the forces for the current state of the ListOfAtoms."""
        self.update()
        return self.paw.get_cartesian_forces()
      
    def GetStress(self):
        """Return the stress for the current state of the ListOfAtoms."""
        raise NotImplementedError

    def _SetListOfAtoms(self, atoms):
        """Make a weak reference to the ListOfAtoms."""
        self.lastcount = -1
        self.atoms = weakref.ref(atoms)
        self.stop_paw()

    def GetNumberOfBands(self):
        """Return the number of bands."""
        return self.nbands 
  
    def SetNumberOfBands(self, nbands):
        """Set the number of bands."""
        self.Set(nbands=nbands)
  
    def GetXCFunctional(self):
        """Return the XC-functional identifier.
        
        'LDA', 'PBE', ..."""
        
        return self.xc 
 
    def GetBZKPoints(self):
        """Return the k-points."""
        return self.bzk_kc
 
    def GetSpinPolarized(self):
        """Is it a spin-polarized calculation?"""
        return self.paw.nspins == 2
    
    def GetIBZKPoints(self):
        """Return k-points in the irreducible part of the Brillouin zone."""
        return self.paw.get_ibz_kpoints()

    # Alternative name:
    GetKPoints = GetIBZKPoints
 
    def GetExactExchange(self):
        """Return non-selfconsistent value of exact exchange"""
        return self.Ha * self.paw.get_exact_exchange()
    
    def GetXCEnergy(self):
        return self.paw.Exc * self.Ha

    def GetIBZKPointWeights(self):
        """Weights of the k-points. 
        
        The sum of all weights is one."""
        
        return self.paw.get_weights()

    def GetDensityArray(self):
        """Return pseudo-density array."""
        return self.paw.density.get_density_array() / self.a0**3

    def GetAllElectronDensity(self, gridrefinement=2):
        """Return reconstructed all-electron density array."""
        return self.paw.density.get_all_electron_density(gridrefinement)\
               / self.a0**3

    def GetWaveFunctionArray(self, band=0, kpt=0, spin=0):
        """Return pseudo-wave-function array."""
        c =  1.0 / self.a0**1.5
        return self.paw.get_wave_function_array(band, kpt, spin) * c

    def GetEigenvalues(self, kpt=0, spin=0):
        """Return eigenvalue array."""
        return self.paw.get_eigenvalues(kpt, spin) * self.Ha

    def GetWannierLocalizationMatrix(self, G_I, kpoint, nextkpoint, spin,
                                     dirG, **args):
        """Calculate integrals for maximally localized Wannier functions."""

        c = dirG.index(1)
        return self.paw.get_wannier_integrals(c, spin, kpoint, nextkpoint, G_I)

    def GetMagneticMoment(self):
        """Return the magnetic moment."""
        return self.paw.magmom

    def GetFermiLevel(self):
        """Return the Fermi-level."""
        return self.paw.get_fermi_level()

    def GetElectronicTemperature(self):
        """Return the electronic temperature in energy units."""
        return self.paw.occupation.kT * self.Ha
        
    def GetElectronicStates(self):
        """Return electronic-state object."""
        from ASE.Utilities.ElectronicStates import ElectronicStates
        self.Write('tmp27.nc')
        return ElectronicStates('tmp27.nc')
    
    def GetListOfAtoms(self):
        """Return attached 'list of atoms' object."""
        return self.atoms()

    def GetGridSpacings(self):
        return self.paw.get_grid_spacings()

    def GetNumberOfGridPoints(self):
        return self.paw.gd.N_c
