from gpaw.utilities.gauss import Gaussian
from gpaw.grid_descriptor import GridDescriptor
from gpaw.poisson import PoissonSolver
from gpaw.poisson_with_gaussian_charges import PoissonSolverWithGaussianCharges


def test_PoissonSolverWithGaussianCharges():
    a = 20.0  # Size of cell
    Nc = (100, 100, 100)  # Number of grid points along each axis
    center_of_charge_1, center_of_charge_2 = [
        [a / 2, a / 2, a / 4], [a / 2, a / 2, a * 3 / 4]]
    gd = GridDescriptor(Nc, (a, a, a), 0)    # Grid-descriptor object
    gaussian_1 = Gaussian(gd, a=19.0, center=center_of_charge_1)
    gaussian_2 = Gaussian(gd, a=19.0, center=center_of_charge_2)
    rho_1 = gaussian_1.get_gauss(0)
    rho_2 = -1 * gaussian_2.get_gauss(0)
    # Total density of two gaussian densities of different sign
    total_density = rho_1 + rho_2
    analytical_potential = gaussian_1.get_gauss_pot(
        0) + (-1 * gaussian_2.get_gauss_pot(0))  # Get analytic potential
    # Test agreement with standard PoissonSolver
    phi_num = gd.zeros()  # Array for storing the potential
    solver = PoissonSolver(nn=3, use_charge_center=True)
    solver.set_grid_descriptor(gd)
    solver.solve(phi_num, total_density, zero_initial_phi=True)
    # Check the agreement between the analytical and numerical potentials
    # at the boundaries
    assert abs(analytical_potential[50, 50, :]
               [0] - phi_num[50, 50, :][0]) > 0.44
    assert abs(analytical_potential[50, 50, :]
               [-1] - phi_num[50, 50, :][-1]) > 0.44
    # There is a difference in values on borders
    # Test agreement with Gaussian charges
    poisson = PoissonSolverWithGaussianCharges()
    poisson.set_grid_descriptor(gd)
    poisson.charges = [int(-gd.integrate(rho_1)), int(-gd.integrate(rho_2))]
    poisson.positions = [center_of_charge_1, center_of_charge_2]
    phi_gaussian = gd.zeros()  # Array for storing the potential
    poisson.solve(phi_gaussian, total_density)
    # Check the agreement at the boundaries
    assert abs(analytical_potential[50, 50, :]
               [0] - phi_gaussian[50, 50, :][0]) < 0.07
    assert abs(analytical_potential[50, 50, :]
               [-1] - phi_gaussian[50, 50, :][-1]) < 0.07
    # The difference is quite small
