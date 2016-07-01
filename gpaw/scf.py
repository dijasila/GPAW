import time
from math import log as ln

import numpy as np
from ase.units import Hartree

from gpaw import KohnShamConvergenceError
from gpaw.forces import calculate_forces


class SCFLoop:
    """Self-consistent field loop."""
    def __init__(self, eigenstates=0.1, energy=0.1, density=0.1, force=np.inf,
                 maxiter=100, fixdensity=False, niter_fixdensity=None):
        self.max_errors = {'eigenstates': eigenstates,
                           'energy': energy,
                           'force': force,
                           'density': density}

        self.maxiter = maxiter
        
        self.fixdensity = fixdensity

        self.old_energies = []
        self.old_F_av = None
        self.converged = False
        
        if niter_fixdensity is None:
            niter_fixdensity = 2
        self.niter_fixdensity = niter_fixdensity

        if fixdensity:
            self.fix_density()
            
        self.iter = None
        
        self.reset()

    def write(self, writer):
        writer.write(converged=self.converged)
        
    def read(self, reader):
        self.converged = reader.scf.converged
    
    def fix_density(self):
        self.fixdensity = True
        self.niter_fixdensity = 10000000
        self.max_errors['density'] = np.inf
        
    def reset(self):
        self.old_energies = []
        self.old_F_av = None
        self.converged = False

    def run(self, wfs, ham, dens, occ, log, callback):
        for self.iter in range(1, self.maxiter + 1):
            wfs.eigensolver.iterate(ham, wfs)
            occ.calculate(wfs)

            energy = ham.get_energy(occ)
            self.old_energies.append(energy)
            errors = self.collect_errors(dens, ham, wfs)

            # Converged?
            for kind, error in errors.items():
                if error > self.max_errors[kind]:
                    self.converged = False
                    break
            else:
                self.converged = True
                
            callback(self.iter)
            self.log(log, self.iter, wfs, ham, dens, occ, errors)
            
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
        
    def collect_errors(self, dens, ham, wfs):
        """Check convergence of eigenstates, energy and density."""

        errors = {'eigenstates': wfs.eigensolver.error,
                  'density': dens.mixer.get_charge_sloshing(),
                  'force': np.inf,
                  'energy': np.inf}

        if len(self.old_energies) >= 3:
            errors['energy'] = np.ptp(self.old_energies[-3:])

        if self.max_errors['force'] < np.inf:
            F_av = calculate_forces(wfs, dens, ham)
            if self.old_F_av is not None:
                errors['force'] = ((F_av - self.old_F_av)**2).sum(1).max()**0.5
            self.old_F_av = F_av
                
        return errors

    def log(self, log, iter, wfs, ham, dens, occ, errors):
        """Output from each iteration."""

        nvalence = wfs.nvalence
        if nvalence > 0:
            eigerr = errors['eigenstates'] * Hartree**2 / nvalence
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
                if self.max_errors['force'] < np.inf:
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

            if self.max_errors['force'] < np.inf:
                if errors['force'] is not None:
                    log('  %+.2f' %
                        (ln(errors['force']) / ln(10)), end='')
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
