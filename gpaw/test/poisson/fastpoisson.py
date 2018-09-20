import itertools
import numpy as np
from ase.build import bulk
from gpaw.poisson import FastPoissonSolver
from gpaw.grid_descriptor import GridDescriptor
from gpaw.fd_operators import Laplace
from gpaw.mpi import world
from gpaw.utilities.gpts import get_number_of_grid_points

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
    atoms = fcc111('Au', size=(1,1,1))
    atoms.center(vacuum=1, axis=2)
    cell = atoms.cell[[2,0,1]].copy()
    yield 'fcc111', cell

    for sym in ['Au', 'Fe', 'Sc']:
        cell = bulk(sym).cell.round(1)  # Round for nice printing
        yield sym, cell.copy()


        return  # XXXX must enable more cells
        cell += 2.0 * rng.rand(3, 3)
        cell = cell.round(1)
        yield sym + 'disp', cell


#import matplotlib.pyplot as plt


nn = 1

for cellno, (cellname, cell_cv) in enumerate(icells()):
    N_c = get_number_of_grid_points(cell_cv, 0.12, 'fd', True, None)
    for pbc in itertools.product(tf, tf, tf):
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
        ps.set_grid_descriptor(gd)
        ps.solve(phi_g, rho_g)

        laplace = Laplace(gd, scale=-1.0 / (4.0 * np.pi), n=nn)

        def get_residual_err(phi_g):
            rhotest_g = gd.zeros()
            laplace.apply(phi_g, rhotest_g)
            return np.abs(rhotest_g - rho_g).max()

        maxerr = get_residual_err(phi_g)
        ps2 = FDPoissonSolver(relax='J', nn=nn, eps=1e-18)
        ps2.set_grid_descriptor(gd)
        phi2_g = gd.zeros()
        ps2.solve(phi2_g, rho_g)

        phimaxerr = np.abs(phi2_g - phi_g).max()
        maxerr2 = get_residual_err(phi2_g)

        pbcstring = '{}{}{}'.format(*pbc)
        print('{:2d} {:8s} pbc={} err[fast]={:8.5e} err[J]={:8.5e} '
              'err[phi]={:8.5e}'
              .format(cellno, cellname, pbcstring, maxerr, maxerr2, phimaxerr))
        #assert maxerr < 1e-13
        #assert maxerr2 < 1e-8
        #assert phimaxerr < 1e-8
