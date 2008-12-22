import numpy as np

from gpaw import KohnShamConvergenceError


class SCFLoop:
    """Energy contributions and forces:

    =========== ==========================================
                Description
    =========== ==========================================
    ``Ekin``    Kinetic energy.
    ``Epot``    Potential energy.
    ``Etot``    Total energy.
    ``Exc``     Exchange-Correlation energy.
    ``Eext``    Energy of external potential
    ``Eref``    Reference energy for all-electron atoms.
    ``S``       Entropy.
    ``Ebar``    Should be close to zero!
    ``converged`` Do we have a self-consistent solution?
    """
    def __init__(self, eigenstates=0.1, energy=0.1, density=0.1, maxiter=100,
                 fixdensity=False, niter_fixdensity=2):
        self.max_eigenstates_error = max(eigenstates, 1e-20)
        self.max_energy_error = energy
        self.max_density_error = max(density, 1e-20)
        self.maxiter = maxiter
        self.fixdensity = fixdensity
        self.niter_fixdensity = niter_fixdensity
        if fixdensity:
            self.niter_fixdensity = 10000000
            self.max_density_error = np.inf
            
        self.reset()

    def reset(self):
        self.energies = []
        self.eigenstates_error = None
        self.energy_error = None
        self.density_error = None
        self.converged = False

    def run(self, wfs, hamiltonian, density, occupations):
        if self.converged:
            return

        for iter in range(1, self.maxiter + 1):
            wfs.eigensolver.iterate(hamiltonian, wfs)
            occupations.calculate(wfs.kpt_u)
            # XXX ortho, dens, wfs?

            energy = hamiltonian.get_energy(occupations)
            self.energies.append(energy)
            self.check_convergence(density, wfs.eigensolver)
            yield iter
            
            if self.converged:
                break

            if iter > self.niter_fixdensity:
                density.update(wfs)
                hamiltonian.update(density)

        # Don't fix the density in the next step:
        self.niter_fixdensity = 0

    def check_convergence(self, density, eigensolver):
        """Check convergence of eigenstates, energy and density."""

        self.eigenstates_error = eigensolver.error

        if len(self.energies) < 3:
            self.energy_error = self.max_energy_error
        else:
            self.energy_error = np.ptp(self.energies[-3:])

        self.density_error = density.mixer.get_charge_sloshing()
        if self.density_error is None:
            self.density_error = self.max_density_error

        self.converged = (
            self.eigenstates_error < self.max_eigenstates_error and
            self.energy_error < self.max_energy_error and
            self.density_error <= self.max_density_error)
        return self.converged
    
