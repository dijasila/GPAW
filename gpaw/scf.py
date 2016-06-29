import time
from math import log as ln

import numpy as np
from ase.units import Hartree

from gpaw import KohnShamConvergenceError


class SCFLoop:
    """Self-consistent field loop.
    
    converged: Do we have a self-consistent solution?
    """
    
    def __init__(self, eigenstates=0.1, energy=0.1, density=0.1, maxiter=100,
                 fixdensity=False, niter_fixdensity=None, force=None):
        self.max_eigenstates_error = max(eigenstates, 1e-20)
        self.max_energy_error = energy
        self.max_force_error = force
        self.max_density_error = max(density, 1e-20)
        self.maxiter = maxiter
        self.fixdensity = fixdensity

        if niter_fixdensity is None:
            niter_fixdensity = 2
        self.niter_fixdensity = niter_fixdensity

        if fixdensity:
            self.fix_density()
            
        self.iter = None
        
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

    def run(self, wfs, ham, dens, occ, log, callback):
        for self.iter in range(1, self.maxiter + 1):
            wfs.eigensolver.iterate(ham, wfs)
            occ.calculate(wfs)

            energy = ham.get_energy(occ)
            self.energies.append(energy)
            if self.max_force_error is not None:
                forces.reset()
            self.check_convergence(dens, wfs.eigensolver, wfs, ham)
            
            callback(self.iter)
            self.log(log, self.iter, wfs, ham, dens, occ)
            
            if self.converged:
                break

            if self.iter > self.niter_fixdensity:
                dens.update(wfs)
                ham.update(dens)
            else:
                ham.npoisson = 0

        # Don't fix the density in the next step:
        self.niter_fixdensity = 0

        if not self.converged:
            log(oops)
            raise KohnShamConvergenceError(
                'Did not converge!  See text output for help.')
        
    def check_convergence(self, dens, eigensolver,
                          wfs=None, ham=None):
        """Check convergence of eigenstates, energy and density."""
        if self.converged:
            return True

        self.eigenstates_error = eigensolver.error

        if len(self.energies) < 3:
            self.energy_error = self.max_energy_error
        else:
            self.energy_error = np.ptp(self.energies[-3:])

        self.density_error = dens.mixer.get_charge_sloshing()
        if self.density_error is None:
            self.density_error = 1000000.0

        if self.max_force_error is not None:
            F_av = forces.calculate(wfs, dens, ham)
            
            if self.force_last is None:
                self.force_last = F_av
            else:
                F_av_diff = ((F_av - self.force_last)**2).sum(axis=1)
                self.force_error = abs(F_av_diff).max()
                self.force_last = F_av

        self.converged = (
            (self.eigenstates_error or 0.0) < self.max_eigenstates_error and
            self.energy_error < self.max_energy_error and
            self.density_error < self.max_density_error and
            (self.force_error or 0) < ((self.max_force_error) or float('inf')))
        return self.converged

    def log(self, log, iter, wfs, ham, dens, occ):
        """Output from each iteration."""

        nvalence = wfs.nvalence
        if nvalence > 0:
            eigerr = self.eigenstates_error * Hartree**2 / nvalence
        else:
            eigerr = 0.0

        T = time.localtime()
        
        if log.verbose != 0:
            log()
            log('------------------------------------')
            log('iter: %d %d:%02d:%02d' % (iter, T[3], T[4], T[5]))
            log()
            log('Poisson Solver Converged in %d Iterations' % ham.npoisson)
            log('Fermi Level Found  in %d Iterations' % occ.niter)
            log('Error in Wave Functions: %.13f' % eigerr)
            log()
            log.print_all_information()
        else:
            if iter == 1:
                header = """\
                     log10-error:    Total        Iterations:
           Time      WFS    Density  Energy       Fermi  Poisson"""
                if wfs.nspins == 2:
                    header += '  MagMom'
                if self.max_force_error is not None:
                    l1 = header.find('Total')
                    header = header[:l1] + '       ' + header[l1:]
                    l2 = header.find('Energy')
                    header = header[:l2] + 'Force  ' + header[l2:]
                log(header)

            if eigerr == 0.0:
                eigerr = ''
            else:
                eigerr = '%+.2f' % (ln(eigerr) / ln(10))

            denserr = dens.mixer.get_charge_sloshing()
            if denserr is None or denserr == 0 or nvalence == 0:
                denserr = ''
            else:
                denserr = '%+.2f' % (ln(denserr / nvalence) / ln(10))

            niterocc = occ.niter
            if niterocc == -1:
                niterocc = ''
            else:
                niterocc = '%d' % niterocc

            if ham.npoisson == 0:
                niterpoisson = ''
            else:
                niterpoisson = str(ham.npoisson)

            log('iter: %3d  %02d:%02d:%02d %6s %6s  ' %
                (iter,
                 T[3], T[4], T[5],
                 eigerr,
                 denserr), end='')

            if self.max_force_error is not None:
                if self.force_error is not None:
                    log('  %+.2f' %
                        (ln(self.scf.force_error) / ln(10)), end='')
                else:
                    log('       ', end='')

            log('%11.6f    %-5s  %-7s' %
                (Hartree * ham.e_total_extrapolated,
                 niterocc,
                 niterpoisson), end='')

            if wfs.nspins == 2:
                log('  %+.4f' % occ.magmom, end='')

            log(flush=True)

        
oops = """
Did not converge!

Here are some tips:

1) Make sure the geometry and spin-state is physically sound.
2) Use less aggressive density mixing.
3) Solve the eigenvalue problem more accurately at each scf-step.
4) Use a smoother distribution function for the occupation numbers.
5) Try adding more empty states.
6) Use enough k-points.
7) Don't let your structure optimization algorithm take too large steps.
8) Solve the Poisson equation more accurately.
9) Better initial guess for the wave functions.

See details here:

    https://wiki.fysik.dtu.dk/gpaw/documentation/convergence.html

"""
