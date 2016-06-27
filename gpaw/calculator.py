"""ASE-calculator interface."""
import numpy as np
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator

import warnings

import numpy as np
from ase.units import Bohr, Hartree
from ase.dft.kpoints import monkhorst_pack
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.utils.timing import Timer
from ase.io.trajectory import read_atoms, write_atoms

import gpaw.io
import gpaw.mpi as mpi
from gpaw.xc import XC
from gpaw.xc.sic import SIC
from gpaw.scf import SCFLoop
from gpaw.setup import Setups
from gpaw.symmetry import Symmetry
import gpaw.wavefunctions.pw as pw
from gpaw.output import PAWTextOutput
import gpaw.occupations as occupations
from gpaw.wavefunctions.lcao import LCAO
from gpaw.wavefunctions.fd import FD
from gpaw.density import RealSpaceDensity
from gpaw.eigensolvers import get_eigensolver
from gpaw.band_descriptor import BandDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.hamiltonian import RealSpaceHamiltonian
from gpaw.utilities.memory import MemNode, maxrss
from gpaw.kohnsham_layouts import get_KohnSham_layouts
from gpaw.wavefunctions.base import EmptyWaveFunctions
from gpaw.utilities.gpts import get_number_of_grid_points
from gpaw.utilities.grid import GridRedistributor
from gpaw.utilities.partition import AtomPartition
from gpaw import dry_run, memory_estimate_depth, KohnShamConvergenceError

from gpaw.paw import PAW
from gpaw.io import Reader, Writer
from gpaw.forces import calculate_forces
from gpaw.stress import calculate_stress


class GPAW(Calculator, PAW, PAWTextOutput):
    """This is the ASE-calculator frontend for doing a PAW calculation."""

    implemented_properties = ['energy', 'forces', 'stress', 'dipole',
                              'magmom', 'magmoms']

    default_parameters = {
        'h': None,  # Angstrom
        'xc': 'LDA',
        'gpts': None,
        'kpts': [(0.0, 0.0, 0.0)],
        'lmax': 2,
        'charge': 0,
        'background_charge': None,
        'nbands': None,
        'setups': 'paw',
        'basis': {},
        'occupations': None,
        'spinpol': None,
        'stencils': (3, 3),
        'fixdensity': False,
        'mixer': None,
        'txt': '-',
        'hund': False,
        'random': False,
        'dtype': None,
        'filter': None,
        'maxiter': 333,  # google its spiritual meaning!
        'parallel': {'kpt': None,
                     'domain': gpaw.parsize_domain,
                     'band': gpaw.parsize_bands,
                     'order': 'kdb',
                     'stridebands': False,
                     # Distribute density/potential on world.
                     # What is a better name for this parameter?
                     # It should probably accept a parsize tuple
                     'augment_grids': gpaw.augment_grids,
                     'sl_auto': False,
                     'sl_default': gpaw.sl_default,
                     'sl_diagonalize': gpaw.sl_diagonalize,
                     'sl_inverse_cholesky': gpaw.sl_inverse_cholesky,
                     'sl_lcao': gpaw.sl_lcao,
                     'sl_lrtddft': gpaw.sl_lrtddft,
                     'buffer_size': gpaw.buffer_size},
        'external': None,  # eV
        'verbose': 0,
        'eigensolver': None,
        'poissonsolver': None,
        'communicator': None,
        'idiotproof': True,
        'mode': 'fd',
        'convergence': {'energy': 0.0005,  # eV / electron
                        'density': 1.0e-4,
                        'eigenstates': 4.0e-8,  # eV^2
                        'bands': 'occupied',
                        'forces': None},  # eV / Ang Max
        'realspace': None,
        'symmetry': {'point_group': True,
                     'time_reversal': True,
                     'symmorphic': True,
                     'tolerance': 1e-7}}
    
    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, timer=None, **kwargs):
    
        if timer is None:
            self.timer = Timer()
        else:
            self.timer = timer

        self.scf = None
        self.wfs = None
        self.occupations = None
        self.density = None
        self.hamiltonian = None

        self.iter = 0
        self.observers = []
        self.initialized = False

        PAWTextOutput.__init__(self)

        Calculator.__init__(self, restart, ignore_bad_restart_file, label,
                            atoms)

    def read(self, filename):
        reader = Reader(filename)
        self.atoms = read_atoms(reader.atoms)
        self.results = reader.results
        self.parameters = reader.parameters
        self.reader.close()
        
    def write(self, filename, mode=''):
        writer = Writer(filename)
        write_atoms(writer.subwriter('atoms'), self.atoms)
        writer.write('results', self.results)
        writer.write('parameters', self.paramesters)
        writer.close()
        
    def check_state(self, atoms, tol=1e-15):
        system_changes = Calculator.check_state(self, atoms, tol)
        if 'positions' not in system_changes:
            if self.hamiltonian.vext:
                if self.hamiltonian.vext.vext_g is None:
                    # QMMM atoms have moved:
                    system_changes.appen('positions')
        return system_changes
        
    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['cell']):
        Calculator.calculate(self, atoms)
        atoms = self.atoms
        
        if system_changes:
            if system_changes == ['positions']:
                # Only positions have changed:
                self.density.reset()
            else:
                # Drastic changes:
                self.wfs = None
                self.occupations = None
                self.density = None
                self.hamiltonian = None
                self.scf = None
                self.initialize(atoms)
            
            self.set_positions(atoms)

        assert not self.scf.converged
        
        self.print_cell_and_parameters()

        self.timer.start('SCF-cycle')
        for iter in self.scf.run(self.wfs, self.hamiltonian, self.density,
                                 self.occupations):
            self.iter = iter
            self.call_observers(iter)
            self.print_iteration(iter)
        self.timer.stop('SCF-cycle')

        if self.scf.converged:
            self.call_observers(iter, final=True)
            self.print_converged(iter)
        else:
            self.txt.write(oops)
            raise KohnShamConvergenceError(
                'Did not converge!  See text output for help.')

        self.results['energy'] = self.hamiltonian.Etot * Hartree
        efree = self.occupations.extrapolate_energy_to_zero_width(
            self.hamiltonian.Etot) * Hartree
        self.results['free_energy'] = efree

        dipole_v = self.density.calculate_dipole_moment()
        self.results['dipole'] = dipole_v * Bohr

        # Magnetic moments within augmentation spheres:
        magmom_a = self.density.estimate_magnetic_moments()
        momsum = self.magmom_a.sum()
        magmom = self.occupations.magmom
        if abs(magmom) > 1e-7 and abs(momsum) > 1e-7:
            magmom_a *= magmom / momsum
        self.results['magmom'] = self.occupations.magmom
        self.results['magmoms'] = magmom_a

        if 'forces' in properties:
            with self.timer('Forces'):
                F_av = calculate_forces(self.wfs, self.density,
                                        self.hamiltonian)
                self.print_forces()
                self.results['forces'] = F_av * (Hartree / Bohr)

        if 'stress' in properties:
            stress = calculate_stress(self).flat[[0, 4, 8, 5, 2, 1]]
            self.results['stress'] = stress * (Hartree / Bohr**3)
            
    def set(self, **kwargs):
        """Change parameters for calculator.

        Examples::

            calc.set(xc='PBE')
            calc.set(nbands=20, kpts=(4, 1, 1))
        """

        if (kwargs.get('h') is not None) and (kwargs.get('gpts') is not None):
            raise ValueError("""You can't use both "gpts" and "h"!""")

        changed_parameters = Calculator.set(self, **kwargs)
        if not changed_parameters:
            return {}
            
        self.initialized = False

        for key in kwargs:
            if key == 'basis' and str(p['mode']) == 'fd':  # umm what about PW?
                # The second criterion seems buggy, will not touch it.  -Ask
                continue

            if key == 'eigensolver':
                self.wfs.set_eigensolver(None)

            if key in ['fixmom', 'mixer',
                       'verbose', 'txt', 'hund', 'random',
                       'eigensolver', 'idiotproof']:
                continue

            if key in ['convergence', 'fixdensity', 'maxiter']:
                self.scf = None
                continue

            # More drastic changes:
            self.scf = None
            self.wfs.set_orthonormalized(False)
            if key in ['lmax', 'stencils', 'external', 'xc',
                       'poissonsolver']:
                self.hamiltonian = None
                self.occupations = None
            elif key in ['occupations']:
                self.occupations = None
            elif key in ['charge', 'background_charge']:
                self.hamiltonian = None
                self.density = None
                self.wfs = None
                self.occupations = None
            elif key in ['kpts', 'nbands', 'usesymm', 'symmetry']:
                self.wfs = None
                self.occupations = None
            elif key in ['h', 'gpts', 'setups', 'spinpol', 'realspace',
                         'parallel', 'communicator', 'dtype', 'mode']:
                self.density = None
                self.occupations = None
                self.hamiltonian = None
                self.wfs = None
            elif key in ['basis']:
                self.wfs = None
            elif key in ['parsize', 'parsize_bands', 'parstride_bands']:
                name = {'parsize': 'domain',
                        'parsize_bands': 'band',
                        'parstride_bands': 'stridebands'}[key]
                raise DeprecationWarning(
                    'Keyword argument has been moved ' +
                    "to the 'parallel' dictionary keyword under '%s'." % name)
            else:
                raise TypeError("Unknown keyword argument: '%s'" % key)

        p.update(kwargs)

    def initialize_positions(self, atoms=None):
        """Update the positions of the atoms."""
        if atoms is None:
            atoms = self.atoms
        else:
            # Save the state of the atoms:
            self.atoms = atoms.copy()

        self.synchronize_atoms()

        spos_ac = atoms.get_scaled_positions() % 1.0

        rank_a = self.wfs.gd.get_ranks_from_positions(spos_ac)
        atom_partition = AtomPartition(self.wfs.gd.comm, rank_a, name='gd')
        self.wfs.set_positions(spos_ac, atom_partition)
        self.density.set_positions(spos_ac, atom_partition)
        self.hamiltonian.set_positions(spos_ac, atom_partition)

        return spos_ac

    def set_positions(self, atoms=None):
        """Update the positions of the atoms and initialize wave functions."""
        spos_ac = self.initialize_positions(atoms)
        self.wfs.initialize(self.density, self.hamiltonian, spos_ac)
        self.wfs.eigensolver.reset()
        self.scf.reset()
        self.print_positions()

    def initialize(self, atoms=None):
        """Inexpensive initialization."""

        if atoms is None:
            atoms = self.atoms
        else:
            # Save the state of the atoms:
            self.atoms = atoms.copy()

        par = self.parameters

        world = par.communicator
        if world is None:
            world = mpi.world
        elif hasattr(world, 'new_communicator'):
            # Check for whether object has correct type already
            #
            # Using isinstance() is complicated because of all the
            # combinations, serial/parallel/debug...
            pass
        else:
            # world should be a list of ranks:
            world = mpi.world.new_communicator(np.asarray(world))
        self.wfs.world = world

        if 'txt' in self._changed_keywords:
            self.set_txt(par.txt)
        self.verbose = par.verbose

        natoms = len(atoms)

        cell_cv = atoms.get_cell() / Bohr
        pbc_c = atoms.get_pbc()
        Z_a = atoms.get_atomic_numbers()
        magmom_av = atoms.get_initial_magnetic_moments()

        self.synchronize_atoms()

        # Generate new xc functional only when it is reset by set
        # XXX sounds like this should use the _changed_keywords dictionary.
        if self.hamiltonian is None or self.hamiltonian.xc is None:
            if isinstance(par.xc, str):
                xc = XC(par.xc)
            else:
                xc = par.xc
        else:
            xc = self.hamiltonian.xc

        mode = par.mode

        if mode == 'fd':
            mode = FD()
        elif mode == 'pw':
            mode = pw.PW()
        elif mode == 'lcao':
            mode = LCAO()
        else:
            assert hasattr(mode, 'name'), str(mode)

        if xc.orbital_dependent and mode.name == 'lcao':
            raise NotImplementedError('LCAO mode does not support '
                                      'orbital-dependent XC functionals.')

        if par.realspace is None:
            realspace = (mode.name != 'pw')
        else:
            realspace = par.realspace
            if mode.name == 'pw':
                assert not realspace

        if par.filter is None and mode.name != 'pw':
            gamma = 1.6
            if par.gpts is not None:
                h = ((np.linalg.inv(cell_cv)**2).sum(0)**-0.5 / par.gpts).max()
            else:
                h = (par.h or 0.2) / Bohr

            def filter(rgd, rcut, f_r, l=0):
                gcut = np.pi / h - 2 / rcut / gamma
                f_r[:] = rgd.filter(f_r, rcut * gamma, gcut, l)
        else:
            filter = par.filter

        setups = Setups(Z_a, par.setups, par.basis, par.lmax, xc,
                        filter, world)

        if magmom_av.ndim == 1:
            collinear = True
            magmom_av, magmom_a = np.zeros((natoms, 3)), magmom_av
            magmom_av[:, 2] = magmom_a
        else:
            collinear = False

        magnetic = magmom_av.any()

        spinpol = par.spinpol
        if par.hund:
            if natoms != 1:
                raise ValueError('hund=True arg only valid for single atoms!')
            spinpol = True
            magmom_av[0] = (0, 0, setups[0].get_hunds_rule_moment(par.charge))

        if spinpol is None:
            spinpol = magnetic
        elif magnetic and not spinpol:
            raise ValueError('Non-zero initial magnetic moment for a ' +
                             'spin-paired calculation!')

        if collinear:
            nspins = 1 + int(spinpol)
            ncomp = 1
        else:
            nspins = 1
            ncomp = 2

        if par.usesymm != 'default':
            warnings.warn('Use "symmetry" keyword instead of ' +
                          '"usesymm" keyword')
            par.symmetry = usesymm2symmetry(par.usesymm)

        symm = par.symmetry
        if symm == 'off':
            symm = {'point_group': False, 'time_reversal': False}

        bzkpts_kc = kpts2ndarray(par.kpts, self.atoms)
        kd = KPointDescriptor(bzkpts_kc, nspins, collinear)
        m_av = magmom_av.round(decimals=3)  # round off
        id_a = list(zip(setups.id_a, *m_av.T))
        symmetry = Symmetry(id_a, cell_cv, atoms.pbc, **symm)
        self.timer.start('Set symmetry')
        kd.set_symmetry(atoms, symmetry, comm=world)
        self.timer.stop('Set symmetry')
        setups.set_symmetry(symmetry)

        if par.gpts is not None:
            N_c = np.array(par.gpts)
        else:
            h = par.h
            if h is not None:
                h /= Bohr
            N_c = get_number_of_grid_points(cell_cv, h, mode, realspace,
                                            kd.symmetry)

        symmetry.check_grid(N_c)

        width = par.width
        if width is None:
            if pbc_c.any():
                width = 0.1  # eV
            else:
                width = 0.0
        else:
            assert par.occupations is None

        if hasattr(self, 'time') or par.dtype == complex:
            dtype = complex
        else:
            if kd.gamma:
                dtype = float
            else:
                dtype = complex

        nao = setups.nao
        nvalence = setups.nvalence - par.charge
        if par.background_charge is not None:
            nvalence += par.background_charge.charge
        M_v = magmom_av.sum(0)
        M = np.dot(M_v, M_v) ** 0.5

        nbands = par.nbands

        orbital_free = any(setup.orbital_free for setup in setups)
        if orbital_free:
            nbands = 1

        if isinstance(nbands, str):
            if nbands[-1] == '%':
                basebands = int(nvalence + M + 0.5) // 2
                nbands = int((float(nbands[:-1]) / 100) * basebands)
            else:
                raise ValueError('Integer Expected: Only use a string '
                                 'if giving a percentage of occupied bands')

        if nbands is None:
            nbands = 0
            for setup in setups:
                nbands_from_atom = setup.get_default_nbands()

                # Any obscure setup errors?
                if nbands_from_atom < -(-setup.Nv // 2):
                    raise ValueError('Bad setup: This setup requests %d'
                                     ' bands but has %d electrons.'
                                     % (nbands_from_atom, setup.Nv))
                nbands += nbands_from_atom
            nbands = min(nao, nbands)
        elif nbands > nao and mode.name == 'lcao':
            raise ValueError('Too many bands for LCAO calculation: '
                             '%d bands and only %d atomic orbitals!' %
                             (nbands, nao))

        if nvalence < 0:
            raise ValueError(
                'Charge %f is not possible - not enough valence electrons' %
                par.charge)

        if nbands <= 0:
            nbands = int(nvalence + M + 0.5) // 2 + (-nbands)

        if nvalence > 2 * nbands and not orbital_free:
            raise ValueError('Too few bands!  Electrons: %f, bands: %d'
                             % (nvalence, nbands))

        nbands *= ncomp

        if par.width is not None:
            self.text('**NOTE**: please start using '
                      'occupations=FermiDirac(width).')
        if par.fixmom:
            self.text('**NOTE**: please start using '
                      'occupations=FermiDirac(width, fixmagmom=True).')

        if self.occupations is None:
            if par.occupations is None:
                # Create object for occupation numbers:
                if orbital_free:
                    width = 0.0  # even for PBC
                    self.occupations = occupations.TFOccupations(width,
                                                                 par.fixmom)
                else:
                    self.occupations = occupations.FermiDirac(width,
                                                              par.fixmom)
            else:
                self.occupations = par.occupations

            # If occupation numbers are changed, and we have wave functions,
            # recalculate the occupation numbers
            if self.wfs is not None and not isinstance(
                    self.wfs,
                    EmptyWaveFunctions):
                self.occupations.calculate(self.wfs)

        self.occupations.magmom = M_v[2]
        
        cc = par.convergence

        if mode.name == 'lcao':
            niter_fixdensity = 0
        else:
            niter_fixdensity = None

        if self.scf is None:
            force_crit = cc['forces']
            if force_crit is not None:
                force_crit /= Hartree / Bohr
            self.scf = SCFLoop(
                cc['eigenstates'] / Hartree**2 * nvalence,
                cc['energy'] / Hartree * max(nvalence, 1),
                cc['density'] * nvalence,
                par.maxiter, par.fixdensity,
                niter_fixdensity,
                force_crit)

        parsize_kpt = par.parallel['kpt']
        parsize_domain = par.parallel['domain']
        parsize_bands = par.parallel['band']

        if not realspace:
            pbc_c = np.ones(3, bool)

        if not self.wfs:
            parallelization = mpi.Parallelization(world,
                                                  nspins * kd.nibzkpts)
            ndomains = None
            if parsize_domain is not None:
                ndomains = np.prod(parsize_domain)
            if mode.name == 'pw':
                if ndomains is not None and ndomains > 1:
                    raise ValueError('Planewave mode does not support '
                                     'domain decomposition.')
                ndomains = 1
            parallelization.set(kpt=parsize_kpt,
                                domain=ndomains,
                                band=parsize_bands)
            comms = parallelization.build_communicators()
            domain_comm = comms['d']
            kpt_comm = comms['k']
            band_comm = comms['b']
            kptband_comm = comms['D']
            domainband_comm = comms['K']

            self.comms = comms
            kd.set_communicator(kpt_comm)

            parstride_bands = par.parallel['stridebands']

            # Unfortunately we need to remember that we adjusted the
            # number of bands so we can print a warning if it differs
            # from the number specified by the user.  (The number can
            # be inferred from the input parameters, but it's tricky
            # because we allow negative numbers)
            self.nbands_parallelization_adjustment = -nbands % band_comm.size
            nbands += self.nbands_parallelization_adjustment

            # I would like to give the following error message, but apparently
            # there are cases, e.g. gpaw/test/gw_ppa.py, which involve
            # nbands > nao and are supposed to work that way.
            # if nbands > nao:
            #    raise ValueError('Number of bands %d adjusted for band '
            #                    'parallelization %d exceeds number of atomic '
            #                     'orbitals %d.  This problem can be fixed '
            #                     'by reducing the number of bands a bit.'
            #                     % (nbands, band_comm.size, nao))
            bd = BandDescriptor(nbands, band_comm, parstride_bands)

            if (self.density is not None and
                    self.density.gd.comm.size != domain_comm.size):
                # Domain decomposition has changed, so we need to
                # reinitialize density and hamiltonian:
                if par.fixdensity:
                    raise RuntimeError(
                        'Density reinitialization conflict ' +
                        'with "fixdensity" - specify domain decomposition.')
                self.density = None
                self.hamiltonian = None

            # Construct grid descriptor for coarse grids for wave functions:
            gd = self.grid_descriptor_class(N_c, cell_cv, pbc_c,
                                            domain_comm, parsize_domain)

            # do k-point analysis here? XXX
            wfs_kwargs = dict(gd=gd, nvalence=nvalence, setups=setups,
                              bd=bd, dtype=dtype, world=world, kd=kd,
                              kptband_comm=kptband_comm, timer=self.timer)

            if par.parallel['sl_auto']:
                # Choose scalapack parallelization automatically

                for key, val in par.parallel.items():
                    if (key.startswith('sl_') and key != 'sl_auto' and
                        val is not None):
                        raise ValueError("Cannot use 'sl_auto' together "
                                         "with '%s'" % key)
                max_scalapack_cpus = bd.comm.size * gd.comm.size
                nprow = max_scalapack_cpus
                npcol = 1

                # Get a sort of reasonable number of columns/rows
                while npcol < nprow and nprow % 2 == 0:
                    npcol *= 2
                    nprow //= 2
                assert npcol * nprow == max_scalapack_cpus

                # ScaLAPACK creates trouble if there aren't at least a few
                # whole blocks; choose block size so there will always be
                # several blocks.  This will crash for small test systems,
                # but so will ScaLAPACK in any case
                blocksize = min(-(-nbands // 4), 64)
                sl_default = (nprow, npcol, blocksize)
            else:
                sl_default = par.parallel['sl_default']

            if mode.name == 'lcao':
                # Layouts used for general diagonalizer
                sl_lcao = par.parallel['sl_lcao']
                if sl_lcao is None:
                    sl_lcao = sl_default
                lcaoksl = get_KohnSham_layouts(sl_lcao, 'lcao',
                                               gd, bd, domainband_comm, dtype,
                                               nao=nao, timer=self.timer)

                self.wfs = mode(collinear, lcaoksl, **wfs_kwargs)

            elif mode.name == 'fd' or mode.name == 'pw':
                # buffer_size keyword only relevant for fdpw
                buffer_size = par.parallel['buffer_size']
                # Layouts used for diagonalizer
                sl_diagonalize = par.parallel['sl_diagonalize']
                if sl_diagonalize is None:
                    sl_diagonalize = sl_default
                diagksl = get_KohnSham_layouts(sl_diagonalize, 'fd',  # XXX
                                               # choice of key 'fd' not so nice
                                               gd, bd, domainband_comm, dtype,
                                               buffer_size=buffer_size,
                                               timer=self.timer)

                # Layouts used for orthonormalizer
                sl_inverse_cholesky = par.parallel['sl_inverse_cholesky']
                if sl_inverse_cholesky is None:
                    sl_inverse_cholesky = sl_default
                if sl_inverse_cholesky != sl_diagonalize:
                    message = 'sl_inverse_cholesky != sl_diagonalize ' \
                        'is not implemented.'
                    raise NotImplementedError(message)
                orthoksl = get_KohnSham_layouts(sl_inverse_cholesky, 'fd',
                                                gd, bd, domainband_comm, dtype,
                                                buffer_size=buffer_size,
                                                timer=self.timer)

                # Use (at most) all available LCAO for initialization
                lcaonbands = min(nbands, nao)

                try:
                    lcaobd = BandDescriptor(lcaonbands, band_comm,
                                            parstride_bands)
                except RuntimeError:
                    initksl = None
                else:
                    # Layouts used for general diagonalizer
                    # (LCAO initialization)
                    sl_lcao = par.parallel['sl_lcao']
                    if sl_lcao is None:
                        sl_lcao = sl_default
                    initksl = get_KohnSham_layouts(sl_lcao, 'lcao',
                                                   gd, lcaobd, domainband_comm,
                                                   dtype, nao=nao,
                                                   timer=self.timer)

                if hasattr(self, 'time'):
                    assert mode.name == 'fd'
                    from gpaw.tddft import TimeDependentWaveFunctions
                    self.wfs = TimeDependentWaveFunctions(
                        stencil=par.stencils[0],
                        diagksl=diagksl,
                        orthoksl=orthoksl,
                        initksl=initksl,
                        gd=gd,
                        nvalence=nvalence,
                        setups=setups,
                        bd=bd,
                        world=world,
                        kd=kd,
                        kptband_comm=kptband_comm,
                        timer=self.timer)
                elif mode.name == 'fd':
                    self.wfs = mode(par.stencils[0], diagksl,
                                    orthoksl, initksl, **wfs_kwargs)
                else:
                    assert mode.name == 'pw'
                    self.wfs = mode(diagksl, orthoksl, initksl,
                                    **wfs_kwargs)
            else:
                self.wfs = mode(self, **wfs_kwargs)
        else:
            self.wfs.set_setups(setups)

        if not self.wfs.eigensolver:
            # Number of bands to converge:
            nbands_converge = cc['bands']
            if nbands_converge == 'all':
                nbands_converge = nbands
            elif nbands_converge != 'occupied':
                assert isinstance(nbands_converge, int)
                if nbands_converge < 0:
                    nbands_converge += nbands
            eigensolver = get_eigensolver(par.eigensolver, mode,
                                          par.convergence)
            eigensolver.nbands_converge = nbands_converge
            # XXX Eigensolver class doesn't define an nbands_converge property

            if isinstance(xc, SIC):
                eigensolver.blocksize = 1
            self.wfs.set_eigensolver(eigensolver)

        if self.density is None:
            gd = self.wfs.gd

            big_gd = gd.new_descriptor(comm=world)
            # Check whether grid is too small.  8 is smallest admissible.
            # (we decide this by how difficult it is to make the tests pass)
            # (Actually it depends on stencils!  But let the user deal with it)
            N_c = big_gd.get_size_of_global_array(pad=True)
            too_small = np.any(N_c / big_gd.parsize_c < 8)
            if par.parallel['augment_grids'] and not too_small:
                aux_gd = big_gd
            else:
                aux_gd = gd

            redistributor = GridRedistributor(world, kptband_comm, gd, aux_gd)

            if par.stencils[1] != 9:
                # Construct grid descriptor for fine grids for densities
                # and potentials:
                finegd = aux_gd.refine()
            else:
                # Special case (use only coarse grid):
                finegd = aux_gd

            if realspace:
                self.density = RealSpaceDensity(
                    gd, finegd, nspins, par.charge + setups.core_charge,
                    redistributor, collinear=collinear,
                    stencil=par.stencils[1],
                    background_charge=par.background_charge)
            else:
                self.density = pw.ReciprocalSpaceDensity(
                    gd, finegd, nspins, par.charge + setups.core_charge,
                    redistributor, collinear=collinear,
                    background_charge=par.background_charge)

        # XXXXXXXXXX if setups change, then setups.core_charge may change.
        # But that parameter was supplied in Density constructor!
        # This surely is a bug!
        self.density.initialize(setups, self.timer, magmom_av, par.hund)
        self.density.set_mixer(par.mixer)

        if self.hamiltonian is None:
            # XXX we hope and pray that hamiltonian becomes None
            # whenever the density does; else the finegds of the two
            # objects may have different communicators depending on
            # whether someone called set()!
            gd, finegd = self.density.gd, self.density.finegd

            if realspace:
                self.hamiltonian = self.real_space_hamiltonian_class(
                    gd=gd, finegd=finegd, nspins=nspins,
                    setups=setups, timer=self.timer, xc=xc,
                    world=world, kptband_comm=self.wfs.kptband_comm,
                    redistributor=self.density.redistributor,
                    vext=par.external, collinear=collinear,
                    psolver=par.poissonsolver,
                    stencil=par.stencils[1])
            else:
                assert par.poissonsolver is None, str(par.poissonsolver) +\
                        ' is not supported in PW mode'
                self.hamiltonian = self.reciprocal_space_hamiltonian_class(
                    gd, finegd, self.density.pd2, self.density.pd3,
                    nspins=nspins, setups=setups, timer=self.timer,
                    xc=xc, world=world,
                    kptband_comm=self.wfs.kptband_comm,
                    redistributor=self.density.redistributor,
                    vext=par.external, collinear=collinear,
                    psolver=par.poissonsolver)

        xc.initialize(self.density, self.hamiltonian, self.wfs,
                      self.occupations)

        self.text()
        self.print_memory_estimate(self.txt, maxdepth=memory_estimate_depth)
        self.txt.flush()

        self.timer.print_info(self)

        if dry_run:
            self.dry_run()

        if realspace and \
                self.hamiltonian.poisson.get_description() == 'FDTD+TDDFT':
            self.hamiltonian.poisson.set_density(self.density)
            self.hamiltonian.poisson.print_messages(self.text)
            self.txt.flush()

        self.initialized = True

    def dry_run(self):
        # Can be overridden like in gpaw.atom.atompaw
        self.print_cell_and_parameters()
        self.print_positions()
        self.txt.flush()
        raise SystemExit

        
oops = """
Did not converge!

Here are some tips:

1) Make sure the geometry and spin-state is physically sound.
2) Use less aggressive density mixing.
3) Solve the eigenvalue problem more accurately at each scf-step.
4) Use a smoother distribution function for the occupation numbers.
5) Try adding more empty states.
6) Use enough k-points.
7) Don't let your structure optimization algorithm take too large steps.
8) Solve the Poisson equation more accurately.
9) Better initial guess for the wave functions.

See details here:

    https://wiki.fysik.dtu.dk/gpaw/documentation/convergence.html

"""
