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
        self.criteria = {c.name: c for c in criteria}

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
        for criterion in self.criteria.values():
            s += ' ' + criterion.description
        s += ("\n (Square brackets indicate name in SCF output, whereas a 'c'"
              " in\n the SCF output indicates the quantity has converged.)\n")
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
            errors = self.collect_errors(dens, ham, wfs, log)

            # Converged?
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
            log(oops, flush=True)
            raise KohnShamConvergenceError(
                'Did not converge!  See text output for help.')

    def collect_errors(self, dens, ham, wfs, log):
        # FIXME/ap: Remove log argument above!
        """Check convergence of eigenstates, energy and density."""

        # XXX Make sure all agree on the density error.
        denserror = broadcast_float(dens.error, wfs.world)

        errors = {'eigenstates': wfs.eigensolver.error,
                  'density': denserror,
                  'energy': np.inf}

        if dens.fixed:
            errors['density'] = 0.0

        if len(self.old_energies) == self.old_energies.maxlen:
            if np.isfinite(self.old_energies).all():
                errors['energy'] = np.ptp(self.old_energies)

        # Converged?
        self.converged_items = {kind: (error < self.max_errors[kind]) for
                                kind, error in errors.items()}

        # Check any user-custom convergence criteria.
        event = SCFEvent(dens=dens, ham=ham, wfs=wfs, log=log)
        for name, criterion in self.criteria.items():
            converged, entry = criterion(event)
            errors[name] = entry
            self.converged_items[name] = converged

        # We only want to calculate the (expensive) forces if we have to.
        check_forces = (self.max_errors['force'] < np.inf and
                        all(self.converged_items.values()))

        # XXX Note this checks just the difference in the last 2
        # iterations, whereas other quantities (energy, workfunction) do
        # a peak-to-peak on 3. Ok?
        errors['force'] = np.inf
        if check_forces:
            with wfs.timer('Forces'):
                F_av = calculate_forces(wfs, dens, ham)
            if self.old_F_av is not None:
                errors['force'] = ((F_av - self.old_F_av)**2).sum(1).max()**0.5
            self.old_F_av = F_av
            self.converged_items['force'] = (errors['force'] <
                                             self.max_errors['force'])
        elif self.max_errors['force'] < np.inf:
            self.converged_items['force'] = False

        return errors

    def log(self, log, niter, wfs, ham, dens, errors):
        """Output from each iteration."""
        if niter == 1:
            header1 = ('{:<12s} {:>8s} {:>12s}  '
                       .format('iterations', 'time', 'total'))
            header2 = ('{:>4s} {:>7s} {:>8s} {:>12s}  '
                       .format('scf', 'poisson', '', 'energy'))
            header1 += 'log10-change:'
            for title in ('wfs', 'dens'):
                header2 += '{:>5s}  '.format(title)
            if self.max_errors['force'] < np.inf:
                header1 += ' ' * 7
                header2 += '{:>5s}  '.format('force')
            for criterion in self.criteria.values():
                header1 += ' ' * 7
                header2 += '{:>5s}  '.format(criterion.tablename)
            if wfs.nspins == 2:
                header1 += '{:>8s} '.format('magmom')
                header2 += '{:>8s} '.format('')
            log(header1)
            log(header2)

        c = {k: 'c' if v else ' ' for k, v in self.converged_items.items()}

        # Iterations and time.
        now = time.localtime()
        line = ('{:4d} {:7d} {:02d}:{:02d}:{:02d} '
                .format(niter, ham.npoisson, now[3], now[4], now[5]))
        # Energy.
        if np.isfinite(ham.e_total_extrapolated):
            energy = '%.6f' % (Ha * ham.e_total_extrapolated)
        else:
            energy = ''
        line += '{:>12s}{:1s} '.format(energy, c['energy'])

        # Eigenstates.
        nvalence = wfs.nvalence
        error = errors['eigenstates'] * Ha**2
        if (np.isinf(error) or error == 0 or nvalence == 0):
            error = '-'
        else:
            error = '{:+5.2f}'.format(np.log10(error / nvalence))
        line += '{:>5s}{:1s} '.format(error, c['eigenstates'])

        # Density.
        error = errors['density']
        if (error is None or np.isinf(error) or error == 0 or
            nvalence == 0):
            error = '-'
        else:
            error = '{:+5.2f}'.format(np.log10(error / nvalence))
        line += '{:>5s}{:1s} '.format(error, c['density'])

        # Force.
        if self.max_errors['force'] < np.inf:
            if errors['force'] == 0:
                error = '-oo'
            elif errors['force'] < np.inf:
                error = '{:+.2f}'.format(
                    np.log10(errors['force'] * Ha / Bohr))
            else:
                error = '-'
            line += '{:>5s}{:s} '.format(error, c['force'])

        # Custom criteria.
        for name in self.criteria:
            line += '{:>5s}{:s}'.format(errors[name], c[name])

        # Magnetic moment.
        if wfs.nspins == 2 or not wfs.collinear:
            totmom_v, _ = dens.estimate_magnetic_moments()
            if wfs.collinear:
                line += f'  {totmom_v[2]:+.4f}'
            else:
                line += ' {:+.1f},{:+.1f},{:+.1f}'.format(*totmom_v)

        log(line, flush=True)


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


class WorkFunction:
    """A convergence criterion for the work function.

    Parameters:

    tol : float
        Tolerance for conversion; that is the maximum variation among the
        last n_old values of either work function.
    n_old : int
        Number of work functions to compare. I.e., if n_old is 3, then this
        compares the peak-to-peak difference among the current work
        function and the two previous.
    """

    def __init__(self, tol, n_old=3):
        self.tol = tol
        self.n_old = n_old
        self.name = 'work function'
        self.tablename = 'wkfxn'
        self.description = ('Maximum change in the last {:d} '
                            'work functions [wkfxn]: {:g} eV'
                            .format(n_old, tol))
        self._old = deque(maxlen=n_old)

    def to_dict(self):
        return {'name': self.name,
                'tol': self.tol,
                'n_old': self.n_old}

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        workfunctions = context.ham.get_workfunctions(context.wfs.fermi_level)
        workfunctions = Ha * np.array(workfunctions)
        self._old.append(workfunctions)  # Pops off >3!
        if len(self._old) == self._old.maxlen:
            error = max(np.ptp(self._old, axis=0))
        else:
            error = np.inf
        converged = (error < self.tol)
        if error < np.inf:
            entry = '{:+5.2f}'.format(np.log10(error))
        else:
            entry = '-'
        return converged, entry


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
