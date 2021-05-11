import time
from collections import deque
from inspect import signature

import numpy as np
from ase.units import Ha, Bohr
from ase.calculators.calculator import InputError

from gpaw import KohnShamConvergenceError
from gpaw.forces import calculate_forces
from gpaw.mpi import broadcast_float


class SCFLoop:
    """Self-consistent field loop."""
    def __init__(self, criteria, maxiter=100, niter_fixdensity=None):
        self.criteria = criteria
        self.maxiter = maxiter
        self.niter_fixdensity = niter_fixdensity

        self.niter = None
        self.reset()

    def __str__(self):
        s = 'Convergence criteria:\n'
        for criterion in self.criteria.values():
            if criterion.description is not None:
                s += ' ' + criterion.description + '\n'
        s += ' Maximum number of [scf] iterations: {:d}'.format(self.maxiter)
        s += ("\n (Square brackets indicate name in SCF output, whereas a 'c'"
              " in\n the SCF output indicates the quantity has converged.)\n")
        return s

    def write(self, writer):
        writer.write(converged=self.converged)

    def read(self, reader):
        self.converged = reader.scf.converged

    def reset(self):
        for criterion in self.criteria.values():
            criterion.reset()
        self.converged = False

    def irun(self, wfs, ham, dens, log, callback):
        self.niter = 1
        while self.niter <= self.maxiter:
            wfs.eigensolver.iterate(ham, wfs)
            e_entropy = wfs.calculate_occupation_numbers(dens.fixed)
            ham.get_energy(e_entropy, wfs)

            entries = {}  # for log file, per criteria.
            converged_items = {}  # True/False, per criteria.
            context = SCFEvent(dens=dens, ham=ham, wfs=wfs, log=log)

            # Cheap items (criterion.calc_last == False).
            for name, criterion in self.criteria.items():
                if not criterion.calc_last:
                    converged, entry = criterion(context)
                    converged_items[name] = converged
                    entries[name] = entry
            cheap_are_done = all(converged_items.values())

            # Expensive items (criterion.calc_last == True).
            missing = set(self.criteria) - set(converged_items)
            for name in missing:
                criterion = self.criteria[name]
                if cheap_are_done:
                    converged, entry = criterion(context)
                else:
                    converged, entry = False, ''
                converged_items[name] = converged
                entries[name] = entry

            # Converged?
            self.converged = all(converged_items.values())

            callback(self.niter)
            self.log(log, converged_items, entries, context)
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
            if not np.isfinite(self.criteria['eigenstates'].get_error()):
                msg = 'Not enough bands for ' + wfs.eigensolver.nbands_converge
                log(msg, flush=True)
                raise KohnShamConvergenceError(msg)
            log(oops, flush=True)
            raise KohnShamConvergenceError(
                'Did not converge!  See text output for help.')

    def log(self, log, converged_items, entries, context):
        """Output from each iteration."""
        custom = (set(self.criteria) -
                  {'energy', 'eigenstates', 'density', 'forces'})
        if self.niter == 1:
            header1 = ('{:<12s} {:>8s} {:>12s}  '
                       .format('iterations', 'time', 'total'))
            header2 = ('{:>4s} {:>7s} {:>8s} {:>12s}  '
                       .format('scf', 'poisson', '', 'energy'))
            header1 += 'log10-change:'
            for title in ('wfs', 'dens'):
                header2 += '{:>5s}  '.format(title)
            if np.isfinite(self.criteria['forces'].tol):
                header1 += ' ' * 7
                header2 += '{:>5s}  '.format('force')
            for name in custom:
                criterion = self.criteria[name]
                header1 += ' ' * 7
                header2 += '{:>5s}  '.format(criterion.tablename)
            if context.wfs.nspins == 2:
                header1 += '{:>8s} '.format('magmom')
                header2 += '{:>8s} '.format('')
            log(header1)
            log(header2)

        c = {k: 'c' if v else ' ' for k, v in converged_items.items()}

        # Iterations and time.
        now = time.localtime()
        line = ('{:4d} {:7d} {:02d}:{:02d}:{:02d} '
                .format(self.niter, context.ham.npoisson, *now[3:6]))

        # Energy.
        line += '{:>12s}{:1s} '.format(entries['energy'], c['energy'])

        # Eigenstates.
        line += '{:>5s}{:1s} '.format(entries['eigenstates'], c['eigenstates'])

        # Density.
        line += '{:>5s}{:1s} '.format(entries['density'], c['density'])

        # Force (optional).
        if np.isfinite(self.criteria['forces'].tol):
            line += '{:>5s}{:s} '.format(entries['forces'], c['forces'])

        # Custom criteria (optional).
        for name in custom:
            line += '{:>5s}{:s}'.format(entries[name], c[name])

        # Magnetic moment (optional).
        if context.wfs.nspins == 2 or not context.wfs.collinear:
            totmom_v, _ = context.dens.estimate_magnetic_moments()
            if context.wfs.collinear:
                line += f'  {totmom_v[2]:+.4f}'
            else:
                line += ' {:+.1f},{:+.1f},{:+.1f}'.format(*totmom_v)

        log(line, flush=True)


class SCFEvent:
    """Object to pass the state of the SCF cycle to a convergence-checking
    function."""

    # FIXME/ap: Note that the SCF cycle does not have a reference to the
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


def get_criterion(name):
    """Returns one of the pre-specified criteria by it's .name attribute,
    and raises sensible error if missing."""
    # Add new criteria to this list.
    criteria = [Energy, Density, Eigenstates, Forces, WorkFunction]
    criteria = {c.name: c for c in criteria}
    try:
        return criteria[name]
    except KeyError:
        msg = ('The convergence keyword "{:s}" was supplied, which we do '
               'not know how to handle. If this is a typo, please '
               'correct. If this is a custom convergence criterion, you '
               'may need to re-import it manually.'.format(name))
        raise InputError(msg)


def dict2criterion(dictionary):
    """Converts a dictionary to a convergence criterion.

    The dictionary can either be that generated from 'todict'; that is like
    {'name': 'energy', 'tol': 0.005, 'n_old': 3}. Or from user-specified
    shortcut like {'energy': 0.005} or {'energy': (0.005, 3)}.
    """
    d = dictionary.copy()
    if 'name' in d:  # from 'todict'
        name = d.pop('name')
        Criterion = get_criterion(name)
        return Criterion(**d)
    else:
        assert len(d) == 1
        name = list(d.keys())[0]
        Criterion = get_criterion(name)
        return Criterion(*[d[name]])


class CriteriaMixin:
    """Automates the creation of the __repr__ and todict methods for generic
    classes. This will work for classes that save all arguments directly,
    like __init__(self, a, b):  --> self.a = a, self.b = b. The todict
    method requires the class have a self.name attribute.
    """
    calc_last = False
    # If calc_last is True, will only be checked after all other (non-last)
    # criteria have been met.

    def __repr__(self):
        parameters = signature(self.__class__).parameters
        s = ','.join([str(getattr(self, p)) for p in parameters])
        return self.__class__.__name__ + '(' + s + ')'

    def todict(self):
        d = {'name': self.name}
        parameters = signature(self.__class__).parameters
        for parameter in parameters:
            d[parameter] = getattr(self, parameter)
        return d

    def reset(self):
        pass


class Energy(CriteriaMixin):
    """A convergence criterion for the total energy.

    Parameters:

    tol : float
        Tolerance for conversion; that is the maximum variation among the
        last n_old values of the total energy, normalized per valence
        electron. [eV/(valence electron)]
    n_old : int
        Number of energy values to compare. I.e., if n_old is 3, then this
        compares the peak-to-peak difference among the current total energy
        and the two previous.
    """
    name = 'energy'
    tablename = 'energy'

    def __init__(self, tol, n_old=3):
        self.tol = tol
        self.n_old = n_old
        self.description = ('Maximum [total energy] change in last {:d} cyles:'
                            ' {:g} eV / electron'
                            .format(self.n_old, self.tol))

    def reset(self):
        self._old = deque(maxlen=self.n_old)

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        # FIXME/ap: I believe the current code was calculating the peak-to-
        # peak energy difference on e_total_free, while reporting
        # e_total_extrapolated in the SCF table (logfile). I changed it to
        # use e_total_extrapolated for both. (Should be a miniscule
        # difference, but at more consistent.)
        energy = context.ham.e_total_extrapolated * Ha
        energy_per_el = energy / context.wfs.nvalence
        self._old.append(energy_per_el)  # Pops off >3!
        error = np.inf
        if len(self._old) == self._old.maxlen:
            error = np.ptp(self._old)
        converged = error < self.tol
        entry = ''
        if np.isfinite(energy):
            entry = '{:11.6f}'.format(energy)
        return converged, entry


class Density(CriteriaMixin):
    """A convergence criterion for the electron density.

    Parameters:

    tol : float
        Tolerance for conversion; that is the maximum variation among the
        last n_old values of the electron density, normalized per valence
        electron. [electrons/(valence electron)]
    """
    name = 'density'
    tablename = 'dens'

    def __init__(self, tol):
        self.tol = tol
        self.description = ('Maximum integral of absolute [dens]ity change: '
                            '{:g} electrons / valence electron'
                            .format(self.tol))

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        nv = context.wfs.nvalence
        # Make sure all agree on the density error.
        error = broadcast_float(context.dens.error, context.wfs.world) / nv
        converged = (error < self.tol)
        if (error is None or np.isinf(error) or error == 0 or nv == 0):
            entry = ''
        else:
            entry = '{:+5.2f}'.format(np.log10(error))
        return converged, entry


class Eigenstates(CriteriaMixin):
    """A convergence criterion for the eigenstates.

    Parameters:

    tol : float
        Tolerance for conversion; that is the maximum variation among the
        last n_old values of the residuals of the Kohn--Sham equations,
        normalized per valence electron. [eV^2/(valence electron)]
    """
    name = 'eigenstates'
    tablename = 'wfs'  # FIXME/ap: should we make this 'eig'?

    def __init__(self, tol):
        self.tol = tol
        self.description = ('Maximum integral of absolute eigenstate [wfs] '
                            'change: {:g} eV^2 / valence electron'
                            .format(self.tol))

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        error = self.get_error(context)
        converged = (error < self.tol)
        if (context.wfs.nvalence == 0 or error == 0 or np.isinf(error)):
            entry = ''
        else:
            entry = '{:+5.2f}'.format(np.log10(error))
        return converged, entry

    def get_error(self, context):
        """Returns the raw error."""
        return context.wfs.eigensolver.error * Ha**2 / context.wfs.nvalence


class Forces(CriteriaMixin):
    """A convergence criterion for the forces.

    Parameters:

    tol : float
        Tolerance for conversion; that is the maximum variation among the
        last n_old values of the maximum force on any atom. [eV/Angstrom]
    """
    name = 'forces'
    tablename = 'force'
    calc_last = True

    def __init__(self, tol):
        self.tol = tol
        self.description = None
        if np.isfinite(self.tol):
            self.description = ('Maximum change in the atomic [forces] across '
                                'last 2 cycles: {:g} eV/Ang'.format(self.tol))
        self.reset()

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        if np.isinf(self.tol):  # criterion is off
            return True, ''
        with context.wfs.timer('Forces'):
            F_av = calculate_forces(context.wfs, context.dens, context.ham)
            F_av *= Ha / Bohr
        error = np.inf
        if self.old_F_av is not None:
            error = ((F_av - self.old_F_av)**2).sum(1).max()**0.5
        self.old_F_av = F_av
        converged = (error < self.tol)
        entry = ''
        if np.isfinite(error):
            entry = '{:+5.2f}'.format(np.log10(error))
        return converged, entry

    def reset(self):
        self.old_F_av = None


class WorkFunction(CriteriaMixin):
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
    name = 'work function'
    tablename = 'wkfxn'

    def __init__(self, tol=0.005, n_old=3):
        self.tol = tol
        self.n_old = n_old
        self.description = ('Maximum change in the last {:d} '
                            'work functions [wkfxn]: {:g} eV'
                            .format(n_old, tol))

    def reset(self):
        self._old = deque(maxlen=self.n_old)

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
            entry = ''
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
