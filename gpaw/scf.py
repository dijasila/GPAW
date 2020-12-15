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
                 workfunction=np.inf, maxiter=100, niter_fixdensity=None,
                 nvalence=None):
        self.max_errors = {'eigenstates': eigenstates,
                           'energy': energy,
                           'density': density,
                           'force': force,
                           'workfunction': workfunction}
        self.maxiter = maxiter
        self.niter_fixdensity = niter_fixdensity
        self.nvalence = nvalence

        self.niter = None
        self.reset()

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
            ('Maximum work function [wkfxn] change: {0:g} eV',
             cc['workfunction'] * Ha),
            ('Maximum number of [iter]ations: {0}', self.maxiter)]:
            if val < np.inf:
                s += ' {0}\n'.format(name.format(val))
        s += ("\n (Square brackets indicate name in SCF output, whereas a 'c'"
              " in\n the SCF output indicates the quantity has converged.)\n")
        return s

    def write(self, writer):
        writer.write(converged=self.converged)

    def read(self, reader):
        self.converged = reader.scf.converged

    def reset(self):
        self.old_energies = deque(maxlen=3)
        self.old_workfunctions = deque(maxlen=3)
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
        """Check convergence of eigenstates, energy, density,
        and work function."""

        # XXX Make sure all agree on the density error:
        denserror = broadcast_float(dens.error, wfs.world)

        errors = {'eigenstates': wfs.eigensolver.error,
                  'density': denserror,
                  'energy': np.inf}

        if dens.fixed:
            errors['density'] = 0.0

        if len(self.old_energies) == self.old_energies.maxlen:
            if np.isfinite(self.old_energies).all():
                errors['energy'] = np.ptp(self.old_energies)

        if self.max_errors['workfunction'] < np.inf:
            try:
                workfunctions = ham.get_workfunctions(wfs.fermi_level)
            except NotImplementedError:
                msg = ('System has no well-defined work function, '
                       'so please do not specify this convergence keyword. '
                       'Consider adding a dipole correction.')
                raise Exception(msg)
            else:
                old = self.old_workfunctions
                old.append(workfunctions)  # Pops off > 3!
                if len(old) == old.maxlen:
                    errors['workfunction'] = max(np.ptp(old, axis=0))

        # We only want to calculate the (expensive) forces if we have to:
        check_forces = (self.max_errors['force'] < np.inf and
                        all(error <= self.max_errors[kind]
                            for kind, error in errors.items()))

        errors['force'] = np.inf
        # XXX Note this checks just the difference in the last 2
        # iterations, whereas other quantities (at least energy) does
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
            if self.max_errors['workfunction'] < np.inf:
                l1 = header.find('total') - 7
                header = header[:l1] + '       ' + header[l1:]
                l2 = header.find('energy') - 7
                header = header[:l2] + 'wkfxn  ' + header[l2:]
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

        if self.max_errors['workfunction'] < np.inf:
            if len(self.old_workfunctions) == 3:
                wkferr = ('{:+.2f}'
                          .format(np.log10(Ha * errors['workfunction'])))
            else:
                wkferr = '-'
            log('{:>5s}{:s} '.format(wkferr, c['workfunction']), end='')

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
