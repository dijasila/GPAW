from __future__ import print_function
import os
import sys
import time
import platform

import numpy as np
import ase
from ase.units import Bohr, Hartree
from ase.data import chemical_symbols
from ase import __version__ as ase_version
from ase.utils import convert_string_to_fd

import gpaw
import _gpaw
from gpaw.utilities.memory import maxrss
from gpaw import dry_run, extra_parameters


class GPAWLogger:
    """Class for handling all text output."""

    def __init__(self, verbose=False, world):
        self.verbose = verbose
        self.world = world
        
        self._fd = 42
        self.oldfd = -42

    @property
    def get_fd(self):
        return self._fd
        
    @get_fd.setter
    def set_fd(self, fd):
        """Set the stream for text output.

        If `txt` is not a stream-object, then it must be one of:

        * None:  Throw output away.
        * '-':  Use stdout (``sys.stdout``) on master, elsewhere throw away.
        * A filename:  Open a new file on master, elsewhere throw away.
        """

        if fd == self.oldfd:
            return
        self.oldfd = fd
        self._fd = convert_string_to_fd(fd, self.world)
        self.header()
        
    def __call__(self, *args, **kwargs):
        flush = kwargs.pop('flush', False)
        print(*args, file=self._fd, **kwargs)
        if flush:
            self._fd.flush()

    def header(self):
        self()
        self('  ___ ___ ___ _ _ _  ')
        self(' |   |   |_  | | | | ')
        self(' | | | | | . | | | | ')
        self(' |__ |  _|___|_____| ', gpaw.__version__)
        self(' |___|_|             ')
        self()

        uname = platform.uname()
        self('User:  ', os.getenv('USER', '???') + '@' + uname[1])
        self('Date:  ', time.asctime())
        self('Arch:  ', uname[4])
        self('Pid:   ', os.getpid())
        self('Python: {0}.{1}.{2}'.format(*sys.version_info[:3]))
        self('gpaw:  ', os.path.dirname(gpaw.__file__))
        
        # Find C-code:
        c = getattr(_gpaw, '__file__', None)
        if not c:
            c = sys.executable
        self('_gpaw: ', os.path.normpath(c))
                  
        self('ase:    %s (version %s)' %
             (os.path.dirname(ase.__file__), ase_version))
        self('numpy:  %s (version %s)' %
             (os.path.dirname(np.__file__), np.version.version))
        try:
            import scipy as sp
            self('scipy:  %s (version %s)' %
                 (os.path.dirname(sp.__file__), sp.version.version))
            # Explicitly deleting SciPy seems to remove garbage collection
            # problem of unknown cause
            del sp
        except ImportError:
            self('scipy:  Not available')
        self('units:  Angstrom and eV')
        self('cores: ', self.world.size)

        if gpaw.debug:
            self('DEBUG MODE')

        if extra_parameters:
            self('Extra parameters:', extra_parameters)

    def __del__(self):
        """Destructor:  Write timing output before closing."""
        if dry_run:
            return
            
        try:
            mr = maxrss()
        except (LookupError, TypeError, NameError):
            # Thing can get weird during interpreter shutdown ...
            mr = 0

        if mr > 0:
            if mr < 1024.0**3:
                log('Memory usage: %.2f MiB' % (mr / 1024.0**2))
            else:
                log('Memory usage: %.2f GiB' % (mr / 1024.0**3))

        self.timer.write(self._fd)

        
def print_cell(gd, pbc_c, log):
    log("""Unit Cell:
           Periodic     X           Y           Z      Points  Spacing
  --------------------------------------------------------------------""")
    h_c = gd.get_grid_spacings()
    for c in range(3):
        log('  %d. axis:    %s  %10.6f  %10.6f  %10.6f   %3d   %8.4f'
            % ((c + 1, ['no ', 'yes'][int(pbc_c[c])]) +
               tuple(Bohr * gd.cell_cv[c]) +
               (gd.N_c[c], Bohr * h_c[c])))
    log()

        
def print_positions(atoms):
    log()
    log('Positions:')
    symbols = atoms.get_chemical_symbols()
    for a, pos_v in enumerate(atoms.get_positions()):
        symbol = symbols[a]
        log('%3d %-2s %9.4f %9.4f %9.4f' % ((a, symbol) + tuple(pos_v)))
    log()

    
def print_parallelization_details(wfs, dens, log):
    nibzkpts = wfs.kd.nibzkpts

    # Print parallelization details
    log('Total number of cores used: %d' % wfs.world.size)
    if wfs.kd.comm.size > 1:  # kpt/spin parallization
        if wfs.nspins == 2 and nibzkpts == 1:
            log('Parallelization over spin')
        elif wfs.nspins == 2:
            log('Parallelization over k-points and spin: %d' %
                wfs.kd.comm.size)
        else:
            log('Parallelization over k-points: %d' %
                wfs.kd.comm.size)

    # Domain decomposition settings:
    coarsesize = tuple(wfs.gd.parsize_c)
    finesize = tuple(dens.finegd.parsize_c)
    try:  # Only planewave density
        xc_redist = dens.xc_redistributor
    except AttributeError:
        xcsize = finesize
    else:
        xcsize = tuple(xc_redist.aux_gd.parsize_c)

    if any(np.prod(size) != 1 for size in [coarsesize, finesize, xcsize]):
        title = 'Domain Decomposition:'
        template = '%d x %d x %d'
        log(title, template % coarsesize)
        if coarsesize != finesize:
            log(' ' * len(title), template % finesize, '(fine grid)')
        if xcsize != finesize:
            log(' ' * len(title), template % xcsize, '(xc only)')

    if wfs.bd.comm.size > 1:  # band parallelization
        log('Parallelization over states: %d' % wfs.bd.comm.size)
        

        if self.scf.fixdensity > self.scf.maxiter:
            t('Fixing the initial density')
        else:
            mixer = self.density.mixer
            t('Mixer Type:', mixer.__class__.__name__)
            t('Linear Mixing Parameter: %g' % mixer.beta)
            t('Mixing with %d Old Densities' % mixer.nmaxold)
            if mixer.weight == 1:
                t('No Damping of Long Wave Oscillations')
            else:
                t('Damping of Long Wave Oscillations: %g' % mixer.weight)

        t('Number of Atoms: %d' % len(self.wfs.setups))
        t('Number of Atomic Orbitals: %d' % self.wfs.setups.nao)
        if self.nbands_parallelization_adjustment != 0:
            t('Adjusting Number of Bands by %+d to Match Parallelization'
              % self.nbands_parallelization_adjustment)
        t('Number of Bands in Calculation: %d' % self.wfs.bd.nbands)
        t('Bands to Converge: ', end='')
        if cc['bands'] == 'occupied':
            t('Occupied States Only')
        elif cc['bands'] == 'all':
            t('All')
        else:
            t('%d Lowest Bands' % cc['bands'])
        t('Number of Valence Electrons: %g' % self.wfs.nvalence)

    def print_converged(self, iter):
        t = self.text
        t('------------------------------------')
        t('Converged After %d Iterations.' % iter)

        t()
        self.print_all_information()

    def print_all_information(self):
        t = self.text
        if len(self.atoms) == 1:
            t('Energy Contributions Relative to Reference Atom:', end='')
        else:
            t('Energy Contributions Relative to Reference Atoms:', end='')
        t('(reference = %.6f)' % (self.wfs.setups.Eref * Hartree))

        t('-------------------------')

        energies = [('Kinetic:      ', self.hamiltonian.e_kinetic),
                    ('Potential:    ', self.hamiltonian.e_coulomb),
                    ('External:     ', self.hamiltonian.e_external),
                    ('XC:           ', self.hamiltonian.e_xc),
                    ('Entropy (-ST):', self.hamiltonian.e_entropy),
                    ('Local:        ', self.hamiltonian.e_zero)]

        for name, e in energies:
            t('%-14s %+11.6f' % (name, Hartree * e))

        e_free = self.hamiltonian.e_total_free
        e_extrapolated = self.hamiltonian.e_total_extrapolated
        t('-------------------------')
        t('Free Energy:   %+11.6f' % (Hartree * e_free))
        t('Zero Kelvin:   %+11.6f' % (Hartree * e_extrapolated))
        t()
        self.occupations.print_fermi_level(self.txt)

        self.print_eigenvalues()

        self.hamiltonian.xc.summary(self.txt)

        t()

        dipole_v = self.results.get('dipole')
        if dipole_v is None:
            dipole_v = self.density.calculate_dipole_moment() * Bohr
        if self.density.charge == 0:
            t('Dipole Moment: %s' % dipole_v)
        else:
            t('Center of Charge: %s' % (dipole_v / self.density.charge))

        try:
            correction = self.hamiltonian.poisson.correction
            epsF = self.occupations.fermilevel
        except AttributeError:
            pass
        else:
            wf1 = (-epsF + correction) * Hartree
            wf2 = (-epsF - correction) * Hartree
            t('Dipole-corrected work function: %f, %f' % (wf1, wf2))

        if self.wfs.nspins == 2:
            t()
            magmom = self.occupations.magmom
            t('Total Magnetic Moment: %f' % magmom)
            try:
                # XXX This doesn't always work, HGH, SIC, ...
                sc = self.density.get_spin_contamination(self.atoms,
                                                         int(magmom < 0))
                t('Spin contamination: %f electrons' % sc)
            except (TypeError, AttributeError):
                pass
            t('Local Magnetic Moments:')
            magmom_a = self.results.get('magmoms')
            if magmom_a is None:
                magmom_a = self.density.estimate_magnetic_moments(
                    total=magmom)
            for a, mom in enumerate(magmom_a):
                t(a, mom)
        t(flush=True)



def eigenvalue_string(paw, comment=' '):
    """
    Write eigenvalues and occupation numbers into a string.
    The parameter comment can be used to comment out non-numers,
    for example to escape it for gnuplot.
    """

    tokens = []
    
    def add(*line):
        for token in line:
            tokens.append(token)
        tokens.append('\n')

    if len(paw.wfs.kd.ibzk_kc) == 1:
        if paw.wfs.nspins == 1:
            add(comment, 'Band  Eigenvalues  Occupancy')
            eps_n = paw.get_eigenvalues(kpt=0, spin=0)
            f_n = paw.get_occupation_numbers(kpt=0, spin=0)
            if paw.wfs.world.rank == 0:
                for n in range(paw.wfs.bd.nbands):
                    add('%5d  %11.5f  %9.5f' % (n, eps_n[n], f_n[n]))
        else:
            add(comment, '                  Up                     Down')
            add(comment, 'Band  Eigenvalues  Occupancy  Eigenvalues  '
                'Occupancy')
            epsa_n = paw.get_eigenvalues(kpt=0, spin=0, broadcast=False)
            epsb_n = paw.get_eigenvalues(kpt=0, spin=1, broadcast=False)
            fa_n = paw.get_occupation_numbers(kpt=0, spin=0, broadcast=False)
            fb_n = paw.get_occupation_numbers(kpt=0, spin=1, broadcast=False)
            if paw.wfs.world.rank == 0:
                for n in range(paw.wfs.bd.nbands):
                    add('%5d  %11.5f  %9.5f  %11.5f  %9.5f' %
                        (n, epsa_n[n], fa_n[n], epsb_n[n], fb_n[n]))
        return ''.join(tokens)

    if len(paw.wfs.kd.ibzk_kc) > 10:
        add('Warning: Showing only first 10 kpts')
        print_range = 10
    else:
        add('Showing all kpts')
        print_range = len(paw.wfs.kd.ibzk_kc)

    if paw.wfs.nvalence / 2. > 10:
        m = int(paw.wfs.nvalence / 2. - 10)
    else:
        m = 0
    if paw.wfs.bd.nbands - paw.wfs.nvalence / 2. > 10:
        j = int(paw.wfs.nvalence / 2. + 10)
    else:
        j = int(paw.wfs.bd.nbands)

    if paw.wfs.nspins == 1:
        add(comment, 'Kpt  Band  Eigenvalues  Occupancy')
        for i in range(print_range):
            eps_n = paw.get_eigenvalues(kpt=i, spin=0)
            f_n = paw.get_occupation_numbers(kpt=i, spin=0)
            if paw.wfs.world.rank == 0:
                for n in range(m, j):
                    add('%3i %5d  %11.5f  %9.5f' % (i, n, eps_n[n], f_n[n]))
                add()
    else:
        add(comment, '                     Up                     Down')
        add(comment, 'Kpt  Band  Eigenvalues  Occupancy  Eigenvalues  '
            'Occupancy')

        for i in range(print_range):
            epsa_n = paw.get_eigenvalues(kpt=i, spin=0, broadcast=False)
            epsb_n = paw.get_eigenvalues(kpt=i, spin=1, broadcast=False)
            fa_n = paw.get_occupation_numbers(kpt=i, spin=0, broadcast=False)
            fb_n = paw.get_occupation_numbers(kpt=i, spin=1, broadcast=False)
            if paw.wfs.world.rank == 0:
                for n in range(m, j):
                    add('%3i %5d  %11.5f  %9.5f  %11.5f  %9.5f' %
                        (i, n, epsa_n[n], fa_n[n], epsb_n[n], fb_n[n]))
                add()
    return ''.join(tokens)


def plot(atoms):
    """Ascii-art plot of the atoms."""

#   y
#   |
#   .-- x
#  /
# z

    cell_cv = atoms.get_cell()
    if (cell_cv - np.diag(cell_cv.diagonal())).any():
        atoms = atoms.copy()
        atoms.cell = [1, 1, 1]
        atoms.center(vacuum=2.0)
        cell_cv = atoms.get_cell()
        plot_box = False
    else:
        plot_box = True

    cell = np.diagonal(cell_cv) / Bohr
    positions = atoms.get_positions() / Bohr
    numbers = atoms.get_atomic_numbers()

    s = 1.3
    nx, ny, nz = n = (s * cell * (1.0, 0.25, 0.5) + 0.5).astype(int)
    sx, sy, sz = n / cell
    grid = Grid(nx + ny + 4, nz + ny + 1)
    positions = (positions % cell + cell) % cell
    ij = np.dot(positions, [(sx, 0), (sy, sy), (0, sz)])
    ij = np.around(ij).astype(int)
    for a, Z in enumerate(numbers):
        symbol = chemical_symbols[Z]
        i, j = ij[a]
        depth = positions[a, 1]
        for n, c in enumerate(symbol):
            grid.put(c, i + n + 1, j, depth)
    if plot_box:
        k = 0
        for i, j in [(1, 0), (1 + nx, 0)]:
            grid.put('*', i, j)
            grid.put('.', i + ny, j + ny)
            if k == 0:
                grid.put('*', i, j + nz)
            grid.put('.', i + ny, j + nz + ny)
            for y in range(1, ny):
                grid.put('/', i + y, j + y, y / sy)
                if k == 0:
                    grid.put('/', i + y, j + y + nz, y / sy)
            for z in range(1, nz):
                if k == 0:
                    grid.put('|', i, j + z)
                grid.put('|', i + ny, j + z + ny)
            k = 1
        for i, j in [(1, 0), (1, nz)]:
            for x in range(1, nx):
                if k == 1:
                    grid.put('-', i + x, j)
                grid.put('-', i + x + ny, j + ny)
            k = 0
    return '\n'.join([''.join([chr(x) for x in line])
                      for line in np.transpose(grid.grid)[::-1]])


class Grid:
    def __init__(self, i, j):
        self.grid = np.zeros((i, j), np.int8)
        self.grid[:] = ord(' ')
        self.depth = np.zeros((i, j))
        self.depth[:] = 1e10

    def put(self, c, i, j, depth=1e9):
        if depth < self.depth[i, j]:
            self.grid[i, j] = ord(c)
            self.depth[i, j] = depth
