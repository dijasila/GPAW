import time
from collections import deque

import numpy as np
from ase.units import Ha, Bohr

from gpaw import KohnShamConvergenceError
from gpaw.forces import calculate_forces
from gpaw.mpi import broadcast_float


class SCFLoop:
    """Self-consistent field loop."""
    def __init__(self, eigenstates=0.1, energy=0.1, density=0.1, force=np.inf,
                 maxiter=100, niter_fixdensity=None, nvalence=None,
                 criteria=None):
        self.max_errors = {'eigenstates': eigenstates,
                           'energy': energy,
                           'density': density,
                           'force': force}
        self.maxiter = maxiter
        self.niter_fixdensity = niter_fixdensity
        self.nvalence = nvalence

        self.niter = None
        self.reset()
        self.criteria = criteria
        if criteria is None:
            self.criteria = []

    def __str__(self):
        cc = self.max_errors
        s = 'Convergence criteria:\n'
        for name, val in [
            ('Maximum [total energy] change: {0:g} eV / electron',
             cc['energy'] * Ha / self.nvalence),
            ('                         (or): {0:g} eV',
             cc['energy'] * Ha),
            ('Maximum integral of absolute [dens]ity change: {0:g} electrons',
             cc['density'] / self.nvalence),
            ('Maximum integral of absolute eigenstate [wfs] change:'
             ' {0:g} eV^2',
             cc['eigenstates'] * Ha**2 / self.nvalence),
            ('Maximum change in atomic [forces]: {0:g} eV / Ang',
             cc['force'] * Ha / Bohr),
            ('Maximum number of [iter]ations: {0}', self.maxiter)]:
            if val < np.inf:
                s += ' {0}\n'.format(name.format(val))
        s += ("\n (Square brackets indicate name in SCF output, whereas a 'c'"
              " in\n the SCF output indicates the quantity has converged.)\n")
        s += str(self.criteria)
        return s

    def write(self, writer):
        writer.write(converged=self.converged)

    def read(self, reader):
        self.converged = reader.scf.converged

    def reset(self):
        self.old_energies = deque(maxlen=3)
        self.old_F_av = None
        self.converged_items = {key: False for key in self.max_errors}
        self.converged = False

    def irun(self, wfs, ham, dens, log, callback):
        self.niter = 1
        while self.niter <= self.maxiter:
            wfs.eigensolver.iterate(ham, wfs)
            e_entropy = wfs.calculate_occupation_numbers(dens.fixed)
            energy = ham.get_energy(e_entropy, wfs)
            self.old_energies.append(energy)  # Pops off > 3!
            errors = self.collect_errors(dens, ham, wfs)

            # Converged?
            for kind, error in errors.items():
                if error > self.max_errors[kind]:
                    self.converged_items[kind] = False
                else:
                    self.converged_items[kind] = True
            if all(self.converged_items.values()):
                self.converged = True
            else:
                self.converged = False

            # Check any user-custom convergence criteria.
            event = SCFEvent(dens=dens, ham=ham, wfs=wfs, log=log)
            for criterion in self.criteria:
                tol = criterion[1]
                error = criterion[0](event)
                if error > tol:
                    self.converged = False
                    log('{:s}: {:.5f}/{:.5f} x'
                        .format(criterion[0].__name__, error, tol))
                else:
                    log('{:s}: {:.5f}/{:.5f} c'
                        .format(criterion[0].__name__, error, tol))

            callback(self.niter)
            self.log(log, self.niter, wfs, ham, dens, errors)
            yield

            if self.converged and self.niter >= self.niter_fixdensity:
                break

            if self.niter > self.niter_fixdensity and not dens.fixed:
                dens.update(wfs)
                ham.update(dens)
            else:
                ham.npoisson = 0
            self.niter += 1

        # Don't fix the density in the next step.
        self.niter_fixdensity = 0

        if not self.converged:
            if not np.isfinite(errors['eigenstates']):
                msg = 'Not enough bands for ' + wfs.eigensolver.nbands_converge
                log(msg)
                raise KohnShamConvergenceError(msg)
            log(oops)
            raise KohnShamConvergenceError(
                'Did not converge!  See text output for help.')

    def collect_errors(self, dens, ham, wfs):
        """Check convergence of eigenstates, energy and density."""

        # XXX Make sure all agree on the density error:
        denserror = broadcast_float(dens.error, wfs.world)

        errors = {'eigenstates': wfs.eigensolver.error,
                  'density': denserror,
                  'energy': np.inf,
                  'force': np.inf}

        if dens.fixed:
            errors['density'] = 0.0

        if len(self.old_energies) == self.old_energies.maxlen:
            if np.isfinite(self.old_energies).all():
                errors['energy'] = np.ptp(self.old_energies)

        # We only want to calculate the (expensive) forces if we have to:
        check_forces = (self.max_errors['force'] < np.inf and
                        all(error <= self.max_errors[kind]
                            for kind, error in errors.items()))

        # XXX Note this checks just the difference in the last 2
        # iterations, whereas other quantities (energy, workfunction) do
        # a peak-to-peak on 3. Ok?
        if check_forces:
            with wfs.timer('Forces'):
                F_av = calculate_forces(wfs, dens, ham)
            if self.old_F_av is not None:
                errors['force'] = ((F_av - self.old_F_av)**2).sum(1).max()**0.5
            self.old_F_av = F_av

        return errors

    def log(self, log, niter, wfs, ham, dens, errors):
        """Output from each iteration."""

        if niter == 1:
            header = """\
                 log10-change:          total poisson
             time   wfs   dens         energy   iters"""
            if wfs.nspins == 2:
                header += '  magmom'
            if self.max_errors['force'] < np.inf:
                l1 = header.find('total') - 8
                header = header[:l1] + '       ' + header[l1:]
                l2 = header.find('energy') - 8
                header = header[:l2] + 'forces ' + header[l2:]
            log(header)

        c = {k: 'c' if v else ' ' for k, v in self.converged_items.items()}

        nvalence = wfs.nvalence
        eigerr = errors['eigenstates'] * Ha**2
        if (np.isinf(eigerr) or eigerr == 0 or nvalence == 0):
            eigerr = '-'
        else:
            eigerr = '{:+.2f}'.format(np.log10(eigerr / nvalence))

        denserr = errors['density']
        if (denserr is None or np.isinf(denserr) or denserr == 0 or
            nvalence == 0):
            denserr = '-'
        else:
            denserr = '{:+.2f}'.format(np.log10(denserr / nvalence))

        if ham.npoisson == 0:
            niterpoisson = ''
        else:
            niterpoisson = '{:d}'.format(ham.npoisson)

        T = time.localtime()
        log('iter:{:3d} {:02d}:{:02d}:{:02d} {:>5s}{:s} {:>5s}{:s} '
            .format(niter, T[3], T[4], T[5],
                    eigerr, c['eigenstates'],
                    denserr, c['density']), end='')

        if self.max_errors['force'] < np.inf:
            if errors['force'] == 0:
                forceerr = '-oo'
            elif errors['force'] < np.inf:
                forceerr = '{:+.2f}'.format(
                    np.log10(errors['force'] * Ha / Bohr))
            else:
                forceerr = '-'
            log('{:>5s}{:s} '.format(forceerr, c['force']), end='')

        if np.isfinite(ham.e_total_extrapolated):
            energy = '%.6f' % (Ha * ham.e_total_extrapolated)
        else:
            energy = ''

        log(' {:>12s}{:s}   {:>4s}'.format(
            energy, c['energy'], niterpoisson), end='')

        if wfs.nspins == 2 or not wfs.collinear:
            totmom_v, _ = dens.estimate_magnetic_moments()
            if wfs.collinear:
                log(f'  {totmom_v[2]:+.4f}', end='')
            else:
                log(' {:+.1f},{:+.1f},{:+.1f}'.format(*totmom_v), end='')

        log(flush=True)


class SCFEvent:
    """Object to pass the state of the SCF cycle to a convergence-checking
    function."""
    # XXX Note that the SCF cycle does not have a reference to the
    # calculator object. For now I am instead giving this event access
    # to the ham, wfs, etc., that SCF already has. But we could consider
    # changing how SCF is initialized to instead just give it a calc ref
    # rather than all these individual pieces. I'll leave that decision to
    # JJ and Ask.

    def __init__(self, dens, ham, wfs, log):
        self.dens = dens
        self.ham = ham
        self.wfs = wfs
        self.log = log


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
