"""ASE-calculator interface."""
import warnings

import numpy as np
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator
from ase.utils import basestring
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
from gpaw.io.logger import GPAWLogger
from gpaw.output import (print_cell, print_positions,
                         print_parallelization_details)
from gpaw.occupations import create_occupation_number_object
from gpaw.wavefunctions.mode import create_wave_function_mode
from gpaw.density import RealSpaceDensity
from gpaw.eigensolvers import get_eigensolver
from gpaw.band_descriptor import BandDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.kpt_descriptor import KPointDescriptor, kpts2ndarray
from gpaw.hamiltonian import RealSpaceHamiltonian
from gpaw.kohnsham_layouts import get_KohnSham_layouts
from gpaw.utilities.gpts import get_number_of_grid_points
from gpaw.utilities.grid import GridRedistributor
from gpaw.utilities.partition import AtomPartition
from gpaw import dry_run, memory_estimate_depth
from gpaw.paw import PAW
from gpaw.io import Reader, Writer
from gpaw.forces import calculate_forces
from gpaw.stress import calculate_stress


class GPAW(Calculator, PAW):
    """This is the ASE-calculator frontend for doing a PAW calculation."""

    implemented_properties = ['energy', 'forces', 'stress', 'dipole',
                              'magmom', 'magmoms']

    default_parameters = {
        'mode': 'fd',
        'xc': 'LDA',
        'occupations': None,
        'poissonsolver': None,
        'h': None,  # Angstrom
        'gpts': None,
        'kpts': [(0.0, 0.0, 0.0)],
        'nbands': None,
        'charge': 0,
        'setups': {},
        'basis': {},
        'spinpol': None,
        'fixdensity': False,
        'filter': None,
        'mixer': None,
        'eigensolver': None,
        'background_charge': None,
        'external': None,
        'random': False,
        'txt': '-',
        'hund': False,
        'maxiter': 333,
        'idiotproof': True,
        'symmetry': {'point_group': True,
                     'time_reversal': True,
                     'symmorphic': True,
                     'tolerance': 1e-7},
        'convergence': {'energy': 0.0005,  # eV / electron
                        'density': 1.0e-4,
                        'eigenstates': 4.0e-8,  # eV^2
                        'bands': 'occupied',
                        'forces': np.inf},  # eV / Ang Max
        'parallel': {'kpt': None,
                     'domain': gpaw.parsize_domain,
                     'band': gpaw.parsize_bands,
                     'order': 'kdb',
                     'stridebands': False,
                     'augment_grids': gpaw.augment_grids,
                     'sl_auto': False,
                     'sl_default': gpaw.sl_default,
                     'sl_diagonalize': gpaw.sl_diagonalize,
                     'sl_inverse_cholesky': gpaw.sl_inverse_cholesky,
                     'sl_lcao': gpaw.sl_lcao,
                     'sl_lrtddft': gpaw.sl_lrtddft,
                     'buffer_size': gpaw.buffer_size},
        'dtype': None,
        'verbose': 0}

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, timer=None, communicator=None, txt='-', **kwargs):
    
        if timer is None:
            self.timer = Timer()
        else:
            self.timer = timer

        self.scf = None
        self.wfs = None
        self.occupations = None
        self.density = None
        self.hamiltonian = None

        self.observers = []  # XXX move to self.scf
        self.initialized = False

        self.world = communicator
        if self.world is None:
            self.world = mpi.world
        elif not hasattr(self.world, 'new_communicator'):
            self.world = mpi.world.new_communicator(np.asarray(self.world))

        self.log = GPAWLogger(world=self.world)
        self.log.fd = txt
        
        self.reader = None
        
        Calculator.__init__(self, restart, ignore_bad_restart_file, label,
                            atoms, **kwargs)
    
    def __del__(self):
        self.timer.write(self.log.fd)
        if self.reader is not None:
            self.reader.close()

    def write(self, filename, mode=''):
        self.log('Writing to {0} (mode={1})'.format(filename, mode))
        writer = Writer(filename)
        writer.write(version=1, ha=Hartree, bohr=Bohr)
        write_atoms(writer.child('atoms'), self.atoms)
        writer.child('results').write(**self.results)
        writer.child('parameters').write(**self.todict())
        writer.write('density', self.density)
        writer.write('hamiltonian', self.hamiltonian)
        writer.write('occupations', self.occupations)
        writer.write('scf', self.scf)
        self.wfs.write(writer.child('wave_functions'), mode == 'all')
        writer.close()
        
    def read(self, filename):
        self.log('Reading from {0}'.format(filename))
        
        self.reader = reader = Reader(filename)
        
        self.atoms = read_atoms(reader.atoms)

        res = reader.results
        self.results = dict((key, res.get(key)) for key in res.keys())
        if self.results:
            self.log('Read {0}'.format(', '.join(self.results)))
                     
        self.log('Reading input parameters:')
        self.parameters = self.get_default_parameters()
        for key in reader.parameters.keys():
            value = reader.parameters.get(key)
            if isinstance(value, (int, float, basestring, np.ndarray)):
                self.log('  {0}: {1}'.format(key, value))
                self.parameters[key] = value
            else:
                value = dict(value.items())
                if len(value) == 1:
                    self.log('  {0}: {1}'.format(key, value))
                else:
                    s = ',\n'.join('{0}: {1}'.format(*item)
                                   for item in sorted(value.items()))
                    self.log('  {0}: {{{1}}}'.format(key, s))
                if isinstance(self.parameters[key], dict):
                    self.parameters[key].update(value)
                else:
                    self.parameters[key] = value
                
        self.initialize()

        self.density.read(reader)
        self.hamiltonian.read(reader)
        self.scf.read(reader)
        self.wfs.read(reader)
        
        # We need to do this in a better way:  XXX
        from gpaw.utilities.partition import AtomPartition
        atom_partition = AtomPartition(self.wfs.gd.comm,
                                       np.zeros(len(self.atoms), dtype=int))
        self.wfs.atom_partition = atom_partition
        self.density.atom_partition = atom_partition
        self.hamiltonian.atom_partition = atom_partition
        
    def check_state(self, atoms, tol=1e-15):
        system_changes = Calculator.check_state(self, atoms, tol)
        if 'positions' not in system_changes:
            if self.hamiltonian:
                if self.hamiltonian.vext:
                    if self.hamiltonian.vext.vext_g is None:
                        # QMMM atoms have moved:
                        system_changes.append('positions')
        return system_changes
        
    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['cell']):
        Calculator.calculate(self, atoms)
        atoms = self.atoms
        
        if system_changes:
            self.log('System changes:', ', '.join(system_changes))
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

        if not self.initialized:
            self.initialize(atoms)
            self.set_positions(atoms)
            
        if not self.scf.converged:
            print_cell(self.wfs.gd, self.atoms.pbc, self.log)
            
            with self.timer('SCF-cycle'):
                self.scf.run(self.wfs, self.hamiltonian,
                             self.density, self.occupations,
                             self.log, self.call_observers)
    
            self.log('------------------------------------')
            self.log('Converged After %d Iterations.' % self.scf.niter)

            e_free = self.hamiltonian.e_total_free
            e_extrapolated = self.hamiltonian.e_total_extrapolated
            self.results['energy'] = e_extrapolated * Hartree
            self.results['free_energy'] = e_free * Hartree

            if not self.atoms.pbc.all() and self.density.charge == 0:
                dipole_v = self.density.calculate_dipole_moment() * Bohr
                self.log('Dipole Moment: ({0:.6f}, {1:.6f}, {2:.6f}) |e|*Ang'
                         .format(*dipole_v))
                self.results['dipole'] = dipole_v
                
            if self.wfs.nspins == 2:
                magmom = self.occupations.magmom
                magmom_a = self.density.estimate_magnetic_moments(
                    total=magmom)
                self.log('Total magnetic moment: %f' % magmom)
                self.log('Local magnetic moments:')
                for a, mom in enumerate(magmom_a):
                    self.log('{0:4} {1:.6f}'.format(a, mom))
                self.results['magmom'] = self.occupations.magmom
                self.results['magmoms'] = magmom_a
    
            self.summary()
        
            self.call_observers(self.scf.niter, final=True)
        
        if 'forces' in properties:
            with self.timer('Forces'):
                F_av = calculate_forces(self.wfs, self.density,
                                        self.hamiltonian, self.log)
                self.results['forces'] = F_av * (Hartree / Bohr)

        if 'stress' in properties:
            with self.timer('Stress'):
                stress = calculate_stress(self).flat[[0, 4, 8, 5, 2, 1]]
                self.results['stress'] = stress * (Hartree / Bohr**3)
                
    def summary(self):
        self.hamiltonian.summary(self.occupations.fermilevel, self.log)
        self.density.summary(self.atoms, self.occupations.magmom, self.log)
        self.occupations.summary(self.log)
        self.wfs.summary(self.log)
        self.log.fd.flush()
            
    def set(self, **kwargs):
        """Change parameters for calculator.

        Examples::

            calc.set(xc='PBE')
            calc.set(nbands=20, kpts=(4, 1, 1))
        """

        changed_parameters = Calculator.set(self, **kwargs)

        # We need to handle txt early in order to get logging up and running:
        if 'txt' in changed_parameters:
            self.log.fd = changed_parameters.pop('txt')
            
        if not changed_parameters:
            return {}
            
        self.initialized = False

        self.log('Input parameters:')
        for key in sorted(changed_parameters):
            self.log('  {0}: {1}'.format(key, changed_parameters[key]))

            if key == 'eigensolver' and self.wfs:
                self.wfs.set_eigensolver(None)

            if key in ['mixer',
                       'verbose', 'txt', 'hund', 'random',
                       'eigensolver', 'idiotproof']:
                continue

            self.scf = None
            self.results = {}
            
            if key in ['convergence', 'fixdensity', 'maxiter']:
                continue

            # More drastic changes:
            if self.wfs:
                self.wfs.set_orthonormalized(False)
            if key in ['external', 'xc', 'poissonsolver']:
                self.hamiltonian = None
                self.occupations = None
            elif key in ['occupations']:
                self.occupations = None
            elif key in ['charge', 'background_charge']:
                self.hamiltonian = None
                self.density = None
                self.wfs = None
                self.occupations = None
            elif key in ['kpts', 'nbands', 'symmetry']:
                self.wfs = None
                self.occupations = None
            elif key in ['h', 'gpts', 'setups', 'spinpol',
                         'parallel', 'dtype', 'mode']:
                self.density = None
                self.occupations = None
                self.hamiltonian = None
                self.wfs = None
            elif key in ['basis']:
                self.wfs = None
            else:
                raise TypeError('Unknown keyword argument: "%s"' % key)
        self.log()

    def initialize_positions(self, atoms=None):
        """Update the positions of the atoms."""
        if atoms is None:
            atoms = self.atoms
        else:
            # Save the state of the atoms:
            self.atoms = atoms.copy()

        mpi.synchronize_atoms(atoms, self.world)

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
        print_positions(self.atoms, self.log)

    def initialize(self, atoms=None):
        """Inexpensive initialization."""

        if atoms is None:
            atoms = self.atoms
        else:
            # Save the state of the atoms:
            self.atoms = atoms.copy()

        par = self.parameters

        natoms = len(atoms)

        cell_cv = atoms.get_cell() / Bohr
        pbc_c = atoms.get_pbc()
        magmom_a = atoms.get_initial_magnetic_moments()

        mpi.synchronize_atoms(atoms, self.world)

        # Generate new xc functional only when it is reset by set
        # XXX sounds like this should use the _changed_keywords dictionary.
        if self.hamiltonian is None or self.hamiltonian.xc is None:
            if isinstance(par.xc, basestring):
                xc = XC(par.xc)
            else:
                xc = par.xc
        else:
            xc = self.hamiltonian.xc

        mode = par.mode
        if isinstance(mode, basestring):
            mode = {'name': mode}
        if isinstance(mode, dict):
            mode = create_wave_function_mode(**mode)

        if par.dtype == complex:
            warnings.warn('Use mode={0}(..., force_complex_dtype=True) '
                          'instead of dtype=complex'.format(mode.name.upper()),
                          stacklevel=3)
            mode.force_complex_dtype = True
            del par['dtype']
            par.mode = mode
            
        if xc.orbital_dependent and mode.name == 'lcao':
            raise NotImplementedError('LCAO mode does not support '
                                      'orbital-dependent XC functionals.')

        realspace = (mode.name != 'pw' and mode.interpolation != 'fft')

        if not realspace:
            pbc_c = np.ones(3, bool)

        self.create_setups(mode, xc)
                
        magnetic = magmom_a.any()

        spinpol = par.spinpol
        if par.hund:
            if natoms != 1:
                raise ValueError('hund=True arg only valid for single atoms!')
            spinpol = True
            magmom_a[0] = self.setups[0].get_hunds_rule_moment(par.charge)

        if spinpol is None:
            spinpol = magnetic
        elif magnetic and not spinpol:
            raise ValueError('Non-zero initial magnetic moment for a ' +
                             'spin-paired calculation!')

        nspins = 1 + int(spinpol)

        if spinpol:
            self.log('Spin-Polarized Calculation.')
            self.log('Magnetic Moment:  {0:.6f}'.format(magmom_a.sum()))
        else:
            self.log('Spin-Paired Calculation')
            
        nao = self.setups.nao
        nvalence = self.setups.nvalence - par.charge
        if par.background_charge is not None:
            nvalence += par.background_charge.charge
        M = abs(magmom_a.sum())

        nbands = par.nbands

        orbital_free = any(setup.orbital_free for setup in self.setups)
        if orbital_free:
            nbands = 1

        if isinstance(nbands, basestring):
            if nbands[-1] == '%':
                basebands = int(nvalence + M + 0.5) // 2
                nbands = int((float(nbands[:-1]) / 100) * basebands)
            else:
                raise ValueError('Integer Expected: Only use a string '
                                 'if giving a percentage of occupied bands')

        if nbands is None:
            nbands = 0
            for setup in self.setups:
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

        if self.occupations is None:
            self.create_occupations(orbital_free)
        self.occupations.magmom = magmom_a.sum()
        
        if self.scf is None:
            self.create_scf(nvalence, mode)
    
        self.create_symmetry(magmom_a, cell_cv)
        
        if not self.wfs:
            self.create_wave_functions(mode, realspace,
                                       nspins, nbands, nao, nvalence,
                                       self.setups,
                                       magmom_a, cell_cv, pbc_c)
        else:
            self.wfs.set_setups(self.setups)

        if not self.wfs.eigensolver:
            self.create_eigensolver(xc, nbands, mode)
            
        if self.density is None:
            self.create_density(realspace, mode)
    
        # XXXXXXXXXX if setups change, then setups.core_charge may change.
        # But that parameter was supplied in Density constructor!
        # This surely is a bug!
        self.density.initialize(self.setups, self.timer, magmom_a, par.hund)
        self.density.set_mixer(par.mixer)
        self.log(self.density.mixer)
        self.density.fixed = par.fixdensity
        self.density.log = self.log

        if self.hamiltonian is None:
            self.create_hamiltonian(realspace, mode, xc)

        xc.initialize(self.density, self.hamiltonian, self.wfs,
                      self.occupations)

        self.print_memory_estimate(maxdepth=memory_estimate_depth)
        
        print_parallelization_details(self.wfs, self.density, self.log)
        
        self.log('Number of Atoms:', natoms)
        self.log('Number of Atomic Orbitals:', self.wfs.setups.nao)
        if self.nbands_parallelization_adjustment != 0:
            self.log(
                'Adjusting Number of Bands by %+d to Match Parallelization' %
                self.nbands_parallelization_adjustment)
        self.log('Number of Bands in Calculation:', self.wfs.bd.nbands)
        self.log('Bands to Converge: ', end='')
        n = par.convergence['bands']
        if n == 'occupied':
            self.log('Occupied States Only')
        elif n == 'all':
            self.log('All')
        else:
            self.log('%d Lowest Bands' % n)
        self.log('Number of Valence Electrons:', self.wfs.nvalence)

        self.log(flush=True)

        self.timer.print_info(self)

        if dry_run:
            self.dry_run()

        if (realspace and
            self.hamiltonian.poisson.get_description() == 'FDTD+TDDFT'):
            self.hamiltonian.poisson.set_density(self.density)
            self.hamiltonian.poisson.print_messages(self.log)
            self.log.fd.flush()

        self.initialized = True

    def create_setups(self, mode, xc):
        if self.parameters.filter is None and mode.name != 'pw':
            gamma = 1.6
            N_c = self.parameters.get('gpts')
            if N_c is None:
                h = (self.parameters.h or 0.2) / Bohr
            else:
                icell_vc = np.linalg.inv(self.atoms.cell)
                h = ((icell_vc**2).sum(0)**-0.5 / N_c).max() / Bohr

            def filter(rgd, rcut, f_r, l=0):
                gcut = np.pi / h - 2 / rcut / gamma
                f_r[:] = rgd.filter(f_r, rcut * gamma, gcut, l)
        else:
            filter = self.parameters.filter

        Z_a = self.atoms.get_atomic_numbers()
        self.setups = Setups(Z_a,
                             self.parameters.setups, self.parameters.basis,
                             xc, filter, self.world)
        self.log(self.setups)

    def create_grid_descriptor(self, N_c, cell_cv, pbc_c,
                               domain_comm, parsize_domain):
        return GridDescriptor(N_c, cell_cv, pbc_c, domain_comm, parsize_domain)
            
    def create_occupations(self, orbital_free):
        occ = self.parameters.occupations
        
        if occ is None:
            # Create object for occupation numbers:
            if orbital_free:
                occ = {'name': 'orbital-free'}
            else:
                if self.atoms.pbc.any():
                    width = 0.1  # eV
                else:
                    width = 0.0
                occ = {'name': 'fermi-dirac', 'width': width}
                
        if isinstance(occ, dict):
            occ = create_occupation_number_object(**occ)
            
        self.occupations = occ

        # If occupation numbers are changed, and we have wave functions,
        # recalculate the occupation numbers
        if self.wfs is not None:
            self.occupations.calculate(self.wfs)
            
        self.log(self.occupations)

    def create_scf(self, nvalence, mode):
        if mode.name == 'lcao':
            niter_fixdensity = 0
        else:
            niter_fixdensity = 2
            
        nv = max(nvalence, 1)
        cc = self.parameters.convergence
        self.scf = SCFLoop(
            cc['eigenstates'] / Hartree**2 * nv,
            cc['energy'] / Hartree * nv,
            cc['density'] * nv,
            cc['forces'] / (Hartree / Bohr),
            self.parameters.maxiter,
            niter_fixdensity, nv)
        self.log(self.scf)
            
    def create_symmetry(self, magmom_a, cell_cv):
        symm = self.parameters.symmetry
        if symm == 'off':
            symm = {'point_group': False, 'time_reversal': False}
        m_a = magmom_a.round(decimals=3)  # round off
        id_a = list(zip(self.setups.id_a, m_a))
        self.symmetry = Symmetry(id_a, cell_cv, self.atoms.pbc, **symm)
        self.symmetry.analyze(self.atoms.get_scaled_positions())
        self.setups.set_symmetry(self.symmetry)

    def create_eigensolver(self, xc, nbands, mode):
        # Number of bands to converge:
        nbands_converge = self.parameters.convergence['bands']
        if nbands_converge == 'all':
            nbands_converge = nbands
        elif nbands_converge != 'occupied':
            assert isinstance(nbands_converge, int)
            if nbands_converge < 0:
                nbands_converge += nbands
        eigensolver = get_eigensolver(self.parameters.eigensolver, mode,
                                      self.parameters.convergence)
        eigensolver.nbands_converge = nbands_converge
        # XXX Eigensolver class doesn't define an nbands_converge property

        if isinstance(xc, SIC):
            eigensolver.blocksize = 1
        self.wfs.set_eigensolver(eigensolver)

        self.log(self.wfs.eigensolver)
        
    def create_density(self, realspace, mode):
        gd = self.wfs.gd

        big_gd = gd.new_descriptor(comm=self.world)
        # Check whether grid is too small.  8 is smallest admissible.
        # (we decide this by how difficult it is to make the tests pass)
        # (Actually it depends on stencils!  But let the user deal with it)
        N_c = big_gd.get_size_of_global_array(pad=True)
        too_small = np.any(N_c / big_gd.parsize_c < 8)
        if self.parameters.parallel['augment_grids'] and not too_small:
            aux_gd = big_gd
        else:
            aux_gd = gd

        redistributor = GridRedistributor(self.world,
                                          self.wfs.kptband_comm,
                                          gd, aux_gd)

        # Construct grid descriptor for fine grids for densities
        # and potentials:
        finegd = aux_gd.refine()

        kwargs = dict(
            gd=gd, finegd=finegd,
            nspins=self.wfs.nspins,
            charge=self.parameters.charge + self.wfs.setups.core_charge,
            redistributor=redistributor,
            background_charge=self.parameters.background_charge)
        
        if realspace:
            self.density = RealSpaceDensity(stencil=mode.interpolation,
                                            **kwargs)
        else:
            self.density = pw.ReciprocalSpaceDensity(**kwargs)
            
        self.log(self.density)
            
    def create_hamiltonian(self, realspace, mode, xc):
        dens = self.density
        kwargs = dict(
            gd=dens.gd, finegd=dens.finegd,
            nspins=dens.nspins,
            setups=dens.setups,
            timer=self.timer,
            xc=xc,
            world=self.world,
            redistributor=dens.redistributor,
            vext=self.parameters.external,
            psolver=self.parameters.poissonsolver)
        if realspace:
            self.hamiltonian = RealSpaceHamiltonian(stencil=mode.interpolation,
                                                    **kwargs)
        else:
            self.hamiltonian = pw.ReciprocalSpaceHamiltonian(
                pd2=dens.pd2, pd3=dens.pd3, **kwargs)
            
        self.log(self.hamiltonian)
        
    def create_wave_functions(self, mode, realspace,
                              nspins, nbands, nao, nvalence, setups,
                              magmom_a, cell_cv, pbc_c):
        par = self.parameters
        
        bzkpts_kc = kpts2ndarray(par.kpts, self.atoms)
        kd = KPointDescriptor(bzkpts_kc, nspins)

        parallelization = mpi.Parallelization(self.world,
                                              nspins * kd.nibzkpts)

        parsize_kpt = par.parallel['kpt']
        parsize_domain = par.parallel['domain']
        parsize_bands = par.parallel['band']

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
        
        self.timer.start('Set symmetry')
        kd.set_symmetry(self.atoms, self.symmetry, comm=self.world)
        self.timer.stop('Set symmetry')

        if par.gpts is not None:
            if par.h is not None:
                raise ValueError("""You can't use both "gpts" and "h"!""")
            N_c = np.array(par.gpts)
        else:
            h = par.h
            if h is not None:
                h /= Bohr
            N_c = get_number_of_grid_points(cell_cv, h, mode, realspace,
                                            kd.symmetry)

        self.symmetry.check_grid(N_c)
        
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
            1 / 0
            
        # Construct grid descriptor for coarse grids for wave functions:
        gd = self.create_grid_descriptor(N_c, cell_cv, pbc_c,
                                         domain_comm, parsize_domain)

        if hasattr(self, 'time') or mode.force_complex_dtype:
            dtype = complex
        else:
            if kd.gamma:
                dtype = float
            else:
                dtype = complex

        wfs_kwargs = dict(gd=gd, nvalence=nvalence, setups=setups,
                          bd=bd, dtype=dtype, world=self.world, kd=kd,
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

            self.wfs = mode(lcaoksl, **wfs_kwargs)

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
                    stencil=mode.nn,
                    diagksl=diagksl,
                    orthoksl=orthoksl,
                    initksl=initksl,
                    gd=gd,
                    nvalence=nvalence,
                    setups=setups,
                    bd=bd,
                    world=self.world,
                    kd=kd,
                    kptband_comm=kptband_comm,
                    timer=self.timer)
            elif mode.name == 'fd':
                self.wfs = mode(diagksl, orthoksl, initksl, **wfs_kwargs)
            else:
                self.wfs = mode(diagksl, orthoksl, initksl, **wfs_kwargs)
        else:
            self.wfs = mode(self, **wfs_kwargs)
        
        self.log(self.wfs)

    def dry_run(self):
        # Can be overridden like in gpaw.atom.atompaw
        print_cell(self.wfs.gd, self.atoms.pbc, self.log)
        print_positions(self.atoms, self.log)
        self.log.fd.flush()
        raise SystemExit
