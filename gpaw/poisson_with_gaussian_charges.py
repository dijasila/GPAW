import numpy as np
from gpaw.poisson import FDPoissonSolver
from gpaw.utilities.blas import axpy


class PoissonSolverWithGaussianCharges(FDPoissonSolver):

    @property
    def charges(self):
        return self._charges

    @charges.setter
    def charges(self, values):
        """poisson.charge = [1, 1, -2]"""
        values = np.array(values)
        assert values.sum() == 0, 'The charges must sum up to zero'
        self._charges = values

    def solve(self, phi, rho, **kwargs):

        zero_initial_phi = kwargs.get('zero_initial_phi', False)

        rho_mod = rho.copy()
        for q, center in zip(self.charges, self.positions):
            self.load_gauss(center=center)
            q /= np.sqrt(4 * np.pi)
            rho_mod += q * self.rho_gauss
            if not zero_initial_phi:
                axpy(q, self.phi_gauss, phi)  # phi += q * self.phi_gauss

        niter = super().solve(phi, rho_mod, **kwargs)

        # correct error introduced by the Gaussian charges
        for q, center in zip(self.charges, self.positions):
            self.load_gauss(center=center)
            q /= np.sqrt(4 * np.pi)
            axpy(-q, self.phi_gauss, phi)  # phi -= q * self.phi_gauss

        return niter
