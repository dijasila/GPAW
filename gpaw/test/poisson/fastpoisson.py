import itertools
import numpy as np
from ase.build import bulk
from gpaw.poisson import FastPoissonSolver, BadAxesError
from gpaw.grid_descriptor import GridDescriptor
from gpaw.fd_operators import Laplace
from gpaw.mpi import world
from gpaw.utilities import h2gpts

# Test: different pbcs
# For pbc=000, test charged system
# Different cells (orthorhombic/general)
# use_cholesky keyword


cell_cv = np.array(bulk('Au').cell)
rng = np.random.RandomState(42)

tf = range(2)
comm = world


def icells():
    # cells: orthorhombic fcc bcc hcp
    yield 'diag', np.diag([3., 4., 5.])

    from ase.build import fcc111
    atoms = fcc111('Au', size=(1, 1, 1))
    atoms.center(vacuum=1, axis=2)
    yield 'fcc111@z', atoms.cell.copy()
    yield 'fcc111@x', atoms.cell[[2, 0, 1]]
    yield 'fcc111@y', atoms.cell[[1, 2, 0]]

    for sym in ['Au', 'Fe', 'Sc']:
        cell = bulk(sym).cell
        yield sym, cell.copy()



#import matplotlib.pyplot as plt

tolerance = 1e-12

nn = 1

def test(cellno, cellname, cell_cv, idiv, pbc):
    N_c = h2gpts(0.12, cell_cv, idiv=idiv)
    if idiv == 1:
        N_c += 1 - N_c % 2  # We want especially to test uneven grids
    gd = GridDescriptor(N_c, cell_cv, pbc_c=pbc)
    rho_g = gd.zeros()
    phi_g = gd.zeros()
    rho_g[:] = -0.3 + rng.rand(*rho_g.shape)

    # Neutralize charge:
    charge = gd.integrate(rho_g)
    magic = gd.get_size_of_global_array().prod()
    rho_g -= charge / gd.dv / magic
    charge = gd.integrate(rho_g)
    assert abs(charge) < 1e-12

    # Check use_cholesky=True/False ?
    from gpaw.poisson import FDPoissonSolver
    ps = FastPoissonSolver(nn=nn)
    #print('setgrid')

    # Will raise BadAxesError for some pbc/cell combinations
    ps.set_grid_descriptor(gd)

    ps.solve(phi_g, rho_g)

    laplace = Laplace(gd, scale=-1.0 / (4.0 * np.pi), n=nn)

    def get_residual_err(phi_g):
        rhotest_g = gd.zeros()
        laplace.apply(phi_g, rhotest_g)
        return np.abs(rhotest_g - rho_g).max()

    maxerr = get_residual_err(phi_g)
    pbcstring = '{}{}{}'.format(*pbc)

    if 0:
        ps2 = FDPoissonSolver(relax='J', nn=nn, eps=1e-18)
        ps2.set_grid_descriptor(gd)
        phi2_g = gd.zeros()
        ps2.solve(phi2_g, rho_g)

        phimaxerr = np.abs(phi2_g - phi_g).max()
        maxerr2 = get_residual_err(phi2_g)
        msg = ('{:2d} {:8s} pbc={} err={:8.5e} err[J]={:8.5e} '
               'err[phi]={:8.5e}'.format(cellno, cellname, pbcstring,
                                         maxerr, maxerr2, phimaxerr))

    state = 'ok' if maxerr < tolerance else 'FAIL'

    msg = ('{:2d} {:8s} grid={} pbc={} err[fast]={:8.5e} {}'
           .format(cellno, cellname, N_c, pbcstring, maxerr, state))
    if world.rank == 0:
        print(msg)

    return maxerr


errs = []
for idiv in [4, 1]:
    for cellno, (cellname, cell_cv) in enumerate(icells()):
        for pbc in itertools.product(tf, tf, tf):
            args = (cellno, cellname, cell_cv, idiv, pbc)

            try:
                err = test(*args)
            except BadAxesError:  # Ignore incompatible pbc/cell combinations
                continue

            errs.append(err)


for i, err in enumerate(errs):
    assert err < tolerance, err
