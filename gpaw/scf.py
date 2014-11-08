import numpy as np
from gpaw.mixer import Mixer
from copy import deepcopy as copy

class SCFLoop:
    """Self-consistent field loop.
    
    converged: Do we have a self-consistent solution?
    """
    
    def __init__(self, eigenstates=0.1, energy=0.1, density=0.1, maxiter=100,
                 fixdensity=False, niter_fixdensity=None,
                 store_iter_D_asp = False,
                 HubAlphaIO = False, force = None):
        self.max_eigenstates_error = max(eigenstates, 1e-20)
        self.max_energy_error = energy
        self.max_force_error = force
        self.max_density_error = max(density, 1e-20)
        self.maxiter = maxiter
        self.fixdensity = fixdensity
        
        self.store_iter_D_asp = store_iter_D_asp
        self.HubAlphaIO = HubAlphaIO
        
        if niter_fixdensity is None:
            niter_fixdensity = 2
        self.niter_fixdensity = niter_fixdensity

        if fixdensity:
            self.fix_density()
            
        self.reset()

    def fix_density(self):
        self.fixdensity = True
        self.niter_fixdensity = 10000000
        self.max_density_error = np.inf
        
    def reset(self):
        self.energies = []
        self.eigenstates_error = None
        self.energy_error = None
        self.density_error = None
        self.force_error = None
        self.force_last = None
        self.converged = False

    def run(self, wfs, hamiltonian, density, occupations, forces):
        if self.converged:
            return
        flag = 0
        for iter in range(1, self.maxiter + 1):
            wfs.eigensolver.iterate(hamiltonian, wfs)
            occupations.calculate(wfs)
            energy = hamiltonian.get_energy(occupations)
            self.energies.append(energy)
            self.check_convergence(density, wfs.eigensolver,
                                   wfs, hamiltonian, forces)
            #print 'convenged?',self.converged 
            yield iter
            if self.HubAlphaIO:
                if iter <= 6 or flag == 0:
                    density.update(wfs)
                    iter_0 = iter
                    self.D_asp_0 = dict((k,v.copy()) for k,v in (density.D_asp).items())
                    if self.converged == True:
                        flag = 1
                else:
                    if iter > iter_0+5 and self.converged:
                        break
                    density.update(wfs)
                    hamiltonian.update(density)
            else:
                if self.converged:
                    break
                if iter > self.niter_fixdensity:
                    density.update(wfs)
                    hamiltonian.update(density)
                else:
                    hamiltonian.npoisson = 0
        # Don't fix the density in the next step:
        self.niter_fixdensity = 0
        
    def check_convergence(self, density, eigensolver,
                          wfs=None, hamiltonian=None, forces=None):
        """Check convergence of eigenstates, energy and density."""
        if self.converged:
            return True

        self.eigenstates_error = eigensolver.error

        if len(self.energies) < 3:
            self.energy_error = self.max_energy_error
        else:
            self.energy_error = np.ptp(self.energies[-3:])

        self.density_error = density.mixer.get_charge_sloshing()
        if self.density_error is None:
            self.density_error = 1000000.0

        if self.max_force_error is not None:
            forces.reset()
            F_av = forces.calculate(wfs, density, hamiltonian)
            
            if self.force_last is None:
                self.force_last = F_av
            else:
                F_av_diff = ((F_av - self.force_last)**2).sum(axis=1)
                self.force_error = abs(F_av_diff).max()
                self.force_last = F_av

        self.converged = (
            self.eigenstates_error < self.max_eigenstates_error and
            self.energy_error < self.max_energy_error and
            self.density_error < self.max_density_error and
            (self.force_error or 0) < ((self.max_force_error)
                                       or float('Inf')))
        return self.converged
