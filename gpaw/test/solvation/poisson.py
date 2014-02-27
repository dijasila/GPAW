from gpaw.solvation.poisson import (
    WeightedFDPoissonSolver, ADM12PoissonSolver, PolarizationPoissonSolver
    )
from gpaw.solvation.dielectric import FDGradientDielectric
from gpaw.grid_descriptor import GridDescriptor
from gpaw.utilities.gauss import Gaussian
from ase.units import Bohr
from gpaw.test import equal
import numpy as np


def make_gd(h, box, pbc):
    diag = np.array([box] * 3)
    cell = np.diag(diag)
    grid_shape = tuple((diag / h * 2).astype(int))
    return GridDescriptor(grid_shape, cell / Bohr, pbc)


class MockDielectric(FDGradientDielectric):
    def update(self, cavity):
        self.update_gradient()

box = 12.
gd = make_gd(h=.4, box=box, pbc=False)


def solve(ps, eps, rho):
    phi = gd.zeros()
    dielectric = MockDielectric(epsinf=eps.max(), nn=3)
    dielectric.set_grid_descriptor(gd)
    dielectric.allocate()
    dielectric.eps_gradeps[0][...] = eps
    dielectric.update(None)
    solver = ps(nn=3, relax='J', eps=2e-10)
    solver.set_dielectric(dielectric)
    solver.set_grid_descriptor(gd)
    solver.initialize()
    solver.solve(phi, rho)
    return phi


psolvers = (
    WeightedFDPoissonSolver,
    ADM12PoissonSolver,
    PolarizationPoissonSolver
    )


# test neutral system with constant permittivity
epsinf = 80.
eps = gd.zeros()
eps.fill(epsinf)
qs = (-1., 1.)
shifts = (-1., 1.)
rho = gd.zeros()
phi_expected = gd.zeros()
for q, shift in zip(qs, shifts):
    gauss_norm = q / np.sqrt(4 * np.pi)
    gauss = Gaussian(gd, center=(box / 2. + shift) * np.ones(3.) / Bohr)
    rho += gauss_norm * gauss.get_gauss(0)
    phi_expected += gauss_norm * gauss.get_gauss_pot(0) / epsinf

for ps in psolvers:
    phi = solve(ps, eps, rho)
    print ps, np.abs(phi - phi_expected).max()
    equal(phi, phi_expected, 1e-3)


# test charged system with constant permittivity
epsinf = 80.
eps = gd.zeros()
eps.fill(epsinf)
q = -2.
gauss_norm = q / np.sqrt(4 * np.pi)
gauss = Gaussian(gd, center=(box / 2. + 1.) * np.ones(3.) / Bohr)
rho_gauss = gauss_norm * gauss.get_gauss(0)
phi_gauss = gauss_norm * gauss.get_gauss_pot(0)
phi_expected = phi_gauss / epsinf

for ps in psolvers:
    phi = solve(ps, eps, rho_gauss)
    print ps, np.abs(phi - phi_expected).max()
    equal(phi, phi_expected, 1e-3)
