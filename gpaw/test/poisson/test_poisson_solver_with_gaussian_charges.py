from gpaw.utilities.gauss import Gaussian
from gpaw.grid_descriptor import GridDescriptor
from gpaw.poisson import PoissonSolver
import matplotlib.pyplot as plt
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
    do_plot = False
    phi_num = gd.zeros()  # Array for storing the potential
    solver = PoissonSolver(nn=3, use_charge_center=True)
    solver.set_grid_descriptor(gd)
    solver.solve(phi_num, total_density, zero_initial_phi=True)
    if not do_plot:
        plt.plot(analytical_potential[50, 50, :], label="Analytical_Potential")
        plt.plot(phi_num[50, 50, :], '--', label="Numerical_Potential")
        plt.legend()
        plt.grid()
    plt.show()  # There is a difference in values on borders
    # Test agreement with Gaussian charges
    poisson = PoissonSolverWithGaussianCharges()
    poisson.set_grid_descriptor(gd)
    poisson.charges = [int(-gd.integrate(rho_1)), int(-gd.integrate(rho_2))]
    poisson.positions = [center_of_charge_1, center_of_charge_2]
    phi_gaussian = gd.zeros()  # Array for storing the potential
    poisson.solve(phi_gaussian, total_density)

    if not do_plot:
        plt.plot(analytical_potential[50, 50, :], label="Analytical_Potential")
        plt.plot(phi_gaussian[50, 50, :], '--',
                 color="green", label="Corrected_Potential")
        plt.plot(phi_num[50, 50, :], '--', color="orange",
                 label="Previous_Numerical_Potential")
        plt.legend()
        plt.grid()
        # The potential we get form PoissionSolverWithGaussianCharges gives the
        # correct expected potential.
    plt.show()
