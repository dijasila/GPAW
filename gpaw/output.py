import numpy as np
from _gpaw import get_num_threads
from ase import Atoms
from ase.data import chemical_symbols
from ase.geometry import cell_to_cellpar
from ase.units import Bohr


def print_cell(gd, pbc_c, log):
    h_c = gd.get_grid_spacings()
    par = cell_to_cellpar(gd.cell_cv * Bohr)
    log(cell=Bohr * gd.cell_cv,
        periodic=pbc_c,
        points=gd.N_c,
        spacing=Bohr * h_c,
        lengths=par[:3],
        angles=par[3:])
    h_eff = gd.dv**(1.0 / 3.0) * Bohr
    log.comment(f'Effective grid spacing dv^(1/3) = {h_eff:.4f}')
    log()


def print_positions(atoms, log, magmom_av):
    log.comment(plot(atoms))
    symbols = atoms.get_chemical_symbols()
    log(positions=[[a, symbols[a], pos_v, magmom_av[a]]
                   for a, pos_v
                   in enumerate(atoms.get_positions())])
    log()


def print_parallelization_details(wfs, ham, log):
    log('Total number of cores used:', wfs.world.size)
    if wfs.kd.comm.size > 1:
        log('Parallelization over k-points:', wfs.kd.comm.size)

    # Domain decomposition settings:
    coarsesize = tuple(wfs.gd.parsize_c)
    finesize = tuple(ham.finegd.parsize_c)

    try:  # Only planewave density
        xc_gd = ham.xc_gd
    except AttributeError:
        xc_gd = ham.finegd
    xcsize = tuple(xc_gd.parsize_c)

    if any(np.prod(size) != 1 for size in [coarsesize, finesize, xcsize]):
        title = 'Domain decomposition:'
        template = '%d x %d x %d'
        log(title, template % coarsesize)
        if coarsesize != finesize:
            log(' ' * len(title), template % finesize, '(fine grid)')
        if xcsize != finesize:
            log(' ' * len(title), template % xcsize, '(xc only)')

    if wfs.bd.comm.size > 1:  # band parallelization
        log('Parallelization over states: %d' % wfs.bd.comm.size)

    if get_num_threads() > 1:  # OpenMP threading
        log('OpenMP threads: {}'.format(get_num_threads()))
    log()


def plot(atoms: Atoms) -> str:
    """Ascii-art plot of the atoms."""

    #   y
    #   |
    #   .-- x
    #  /
    # z

    if atoms.cell.handedness != 1:
        # See example we can't handle in test_ascii_art()
        return ''

    cell_cv = atoms.get_cell()
    if atoms.cell.orthorhombic:
        plot_box = True
    else:
        atoms = atoms.copy()
        atoms.cell = [1, 1, 1]
        atoms.center(vacuum=2.0)
        cell_cv = atoms.get_cell()
        plot_box = False

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
