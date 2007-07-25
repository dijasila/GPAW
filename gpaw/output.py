import time
from math import log

import Numeric as num
from ASE.ChemicalElements.symbol import symbols

from gpaw.utilities import devnull

class Output:
    def __init__(self):
        """Set the stream for text output.

        If `txt` is not a stream-object, then it must be one of:

        ``None``:
          Throw output away.
        ``'-'``:
          Use standard-output (``sys.stdout``).
        A filename:
          open a new file.
        """

        txt = self.txt
        if txt is None or rank != MASTER:
            txt = devnull
        elif txt == '-':
            txt = sys.stdout
        elif isinstance(txt, str):
            txt = open(txt, 'w')
        self.txt = txt

    def text(self, *args, sep=' ', end='\n'):
        self.txt.write(sep.join(args) + end)
        
    def print_logo(self):
        self.text()
        self.text('  ___ ___ ___ _ _ _  ')
        self.text(' |   |   |_  | | | | ')
        self.text(' | | | | | . | | | | ')
        self.text(' |__ |  _|___|_____| ', version)
        self.text(' |___|_|             ')
        self.text()

        uname = os.uname()
        self.text('User:', os.getenv('USER') + '@' + uname[1]
        self.text('Date:', time.asctime()
        self.text('Arch:', uname[4]
        self.text('Pid: ', os.getpid()
        self.text('Dir: ', os.path.dirname(gpaw.__file__)
                  
    def print_init(self):
        t = self.text
        if self.gamma:
            self.text('Gamma-point calculation')
        self.text('Reference energy:', self.Eref * self.Ha)

        if self.kpt_comm.size > 1:
            if self.nspins == 2:
                self.text(
            'Parallelization over k-points and spin with %d processors' %
            self.kpt_comm.size)
            else:
                self.text('Parallelization over k-points with %d processors'
                           % self.kpt_comm.size)

        domain = self.domain
        if domain.comm.size > 1:
            self.text(('Using domain decomposition: %d x %d x %d' %
                           tuple(domain.parsize_c)))

        if symmetry is not None:
                          symmetry.print_symmetries(out)
        
            nkpts = len(ibzk_kc)
            self.text((('%d k-point%s in the irreducible part of the ' +
                           'Brillouin zone (total: %d)') %
                           (nkpts, ' s'[1:nkpts], len(bzk_kc)))

            print >> self.out, 'Positions:'
            for a, pos_c in enumerate(pos_ac):
                symbol = self.nuclei[a].setup.symbol
                print >> self.out, '%3d %2s %8.4f%8.4f%8.4f' % \
                    ((a, symbol) + tuple(self.a0 * pos_c))

        self.timer.stop()

    def print_converged(self):
        out = self.out
        self.text('------------------------------------'
        self.text('Converged after %d iterations.' % self.niter

        self.text()
        print_all_information(self)

    def print_all_information(self):

        out = self.out    
        if len(self.nuclei) == 1:
            self.text('energy contributions relative to reference atom:',
        else:
            self.text('energy contributions relative to reference atoms:',
        self.text('(reference = %.5f)' % (self.Eref * self.Ha)

        self.text('-------------------------'

        energies = [('kinetic:', self.Ekin),
                    ('potential:', self.Epot),
                    ('external:', self.Eext),
                    ('XC:', self.Exc),
                    ('entropy (-ST):', -self.S),
                    ('local:', self.Ebar)]

        for name, e in energies:
            self.text('%-14s %+10.5f' % (name, self.Ha * e)

        self.text('-------------------------'
        self.text('free energy:   %+10.5f' % (self.Ha * self.Etot)
        self.text('zero Kelvin:   %+10.5f' % \
              (self.Ha * (self.Etot + 0.5 * self.S))
        self.text()
        epsF = self.occupation.get_fermi_level()
        if epsF is not None:
            self.text('Fermi level:', self.Ha * epsF

        print_eigenvalues(self)

        self.text()
        charge = self.finegd.integrate(self.density.rhot_g)
        self.text('total charge: %f electrons' % charge

        dipole = self.finegd.calculate_dipole_moment(self.density.rhot_g)
        if self.density.charge == 0:
            self.text('dipole moment: %s' % (dipole * self.a0)
        else:
            self.text('center of charge: %s' % (dipole * self.a0)

        if self.nspins == 2:
            self.density.calculate_local_magnetic_moments()

            self.text()
            self.text('total magnetic moment: %f' % self.magmom
            self.text('local magnetic moments:'
            for nucleus in self.nuclei:
                self.text(nucleus.a, nucleus.mom
            self.text()


    def iteration(self):
        # Output from each iteration:
        write = self.write

        if self.verbosity != 0:
            t = time.localtime()
            self.text()
            self.text('------------------------------------')
            self.text('iter: %d %d:%02d:%02d' % (self.niter, t[3], t[4], t[5]))
            self.text()
            self.text('Poisson solver converged in %d iterations' %
                      self.hamiltonian.npoisson)
            self.text('Fermi level found  in %d iterations' % self.nfermi)
            self.text('Log10 error in wave functions: %4.1f' %
                      (log(self.error) / log(10)))
            self.text()
            print_all_information(self)

        else:        
            if self.niter == 0:
                self.text("""\
                          log10     total     iterations:
                 time     error     energy    fermi  poisson  magmom"""

            t = time.localtime()

            write('iter: %4d %3d:%02d:%02d %6.1f %13.7f %4d %7d' %
                  (self.niter,
                   t[3], t[4], t[5],
                   log(self.error) / log(10),
                   self.Ha * (self.Etot + 0.5 * self.S),
                   self.nfermi,
                   self.hamiltonian.npoisson))

            if self.nspins == 2:
                self.text('%11.4f' % self.magmom)
            else:
                self.text('       --')

        out.flush()

    def print_eigenvalues(self):
        """Print eigenvalues and occupation numbers."""

        Ha = self.Ha

        if self.nkpts > 1 or self.kpt_comm.size > 1:
            # not implemented yet:
            return ''

        s = ''
        if self.nspins == 1:
            s += comment + ' band     eps        occ\n'
            kpt = self.kpt_u[0]
            for n in range(self.nbands):
                s += ('%4d %10.5f %10.5f\n' %
                      (n, Ha * kpt.eps_n[n], kpt.f_n[n]))
        else:
            s += comment + '                up                   down\n'
            s += comment + ' band     eps        occ        eps        occ\n'
            epsa_n = self.kpt_u[0].eps_n
            epsb_n = self.kpt_u[1].eps_n
            fa_n = self.kpt_u[0].f_n
            fb_n = self.kpt_u[1].f_n
            for n in range(self.nbands):
                s += ('%4d %10.5f %10.5f %10.5f %10.5f\n' %
                      (n,
                       Ha * epsa_n[n], fa_n[n],
                       Ha * epsb_n[n], fb_n[n]))
        return s

    def plot_atoms(self):
        domain = self.domain
        nuclei = self.nuclei
        out = self.out
        cell_c = domain.cell_c
        pos_ac = cell_c * [nucleus.spos_c for nucleus in nuclei]
        Z_a = [nucleus.setup.Z for nucleus in nuclei]
        self.text(plot(pos_ac, Z_a, cell_c))
        self.text()
        self.text('unitcell:')
        self.text('         periodic  length  points   spacing')
        self.text('  -----------------------------------------')
        for c in range(3):
            self.text('  %s-axis   %s   %8.4f   %3d    %8.4f' % 
                      ('xyz'[c],
                       ['no ', 'yes'][domain.periodic_c[c]],
                       self.a0 * domain.cell_c[c],
                       self.gd.N_c[c],
                       self.a0 * self.gd.h_c[c]))
        self.text()

def plot(positions, numbers, cell):
    """Ascii-art plot of the atoms.

    Example::

      from ASE import ListOfAtoms, Atom
      a = 4.0
      n = 20
      d = 1.0
      x = d / 3**0.5
      atoms = ListOfAtoms([Atom('C', (0.0, 0.0, 0.0)),
                           Atom('H', (x, x, x)),
                           Atom('H', (-x, -x, x)),
                           Atom('H', (x, -x, -x)),
                           Atom('H', (-x, x, -x))],
                          cell=(a, a, a), periodic=True)
      for line in plot(2*atoms.GetCartesianPositions() + (a,a,a),
                       atoms.GetAtomicNumbers(),
                       2*num.array(atoms.GetUnitCell().flat[::4])):
          print line

          .-----------.
         /|           |
        / |           |
       *  |      H    |
       |  | H  C      |
       |  |  H        |
       |  .-----H-----.
       | /           /
       |/           /
       *-----------*
    """

    s = 1.3
    nx, ny, nz = n = (s * cell * (1.0, 0.25, 0.5) + 0.5).astype(num.Int)
    sx, sy, sz = n / cell
    grid = Grid(nx + ny + 4, nz + ny + 1)
    positions = (positions % cell + cell) % cell
    ij = num.dot(positions, [(sx, 0), (sy, sy), (0, sz)])
    ij = num.around(ij).astype(num.Int)
    for a, Z in enumerate(numbers):
        symbol = symbols[Z]
        i, j = ij[a]
        depth = positions[a, 1]
        for n, c in enumerate(symbol):
            grid.put(c, i + n + 1, j, depth)
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
                      for line in num.transpose(grid.grid)[::-1]])

class Grid:
    def __init__(self, i, j):
        self.grid = num.zeros((i, j), num.Int8)
        self.grid[:] = ord(' ')
        self.depth = num.zeros((i, j), num.Float)
        self.depth[:] = 1e10

    def put(self, c, i, j, depth=1e9):
        if depth < self.depth[i, j]:
            self.grid[i, j] = ord(c)
            self.depth[i, j] = depth

