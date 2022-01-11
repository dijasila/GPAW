import time
import warnings
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
        self.converged = False
        self.eigensolver_used = None

    def __str__(self):
        s = 'Convergence criteria:\n'
        for criterion in self.criteria.values():
            if criterion.description is not None:
                s += ' ' + criterion.description + '\n'
        s += ' Maximum number of scf [iter]ations: {:d}'.format(self.maxiter)
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
        self.eigensolver_used = None

    def irun(self, wfs, ham, dens, log, callback):

        self.eigensolver_used = getattr(wfs.eigensolver, "name", None)
        self.check_eigensolver_state(wfs, ham, dens)
        self.niter = 1
        cheap, expensive = self.prepare_convergence_criteria()

        while self.niter <= self.maxiter:
            self.iterate_eigensolver(wfs, ham, dens)

            self.check_convergence(
                dens, ham, wfs, log, cheap, expensive, callback)
            yield

            if self.converged and self.niter >= self.niter_fixdensity:
                self.do_if_converged(wfs, ham, dens, log)
                break

            self.update_ham_and_dens(wfs, ham, dens)
            self.niter += 1

        # Don't fix the density in the next step.
        self.niter_fixdensity = 0

        if not self.converged:
            self.not_converged(dens, ham, wfs, log)

    def log(self, log, converged_items, entries, context):
        """Output from each iteration."""
        write_iteration(self.criteria, converged_items, entries, context, log)

    def prepare_convergence_criteria(self):
        cheap = {k: c for k, c in self.criteria.items() if not c.calc_last}
        expensive = {k: c for k, c in self.criteria.items() if c.calc_last}
        return cheap, expensive

    def check_convergence(self, dens, ham, wfs, log, cheap, expensive,
                          callback):

        entries = {}  # for log file, per criteria
        converged_items = {}  # True/False, per criteria
        context = SCFEvent(dens=dens, ham=ham, wfs=wfs, niter=self.niter,
                           log=log)

        for name, criterion in cheap.items():
            converged, entry = criterion(context)
            converged_items[name] = converged
            entries[name] = entry

        calculate_expensive = False
        if all(converged_items.values()):
            # Stays on rest of cycle even if a cheap one slips back out.
            calculate_expensive = True

        for name, criterion in expensive.items():
            converged, entry = False, ''
            if calculate_expensive:
                converged, entry = criterion(context)
            converged_items[name] = converged
            entries[name] = entry

        # Converged?
        self.converged = all(converged_items.values())

        callback(self.niter)
        self.log(log, converged_items, entries, context)

    def not_converged(self, dens, ham, wfs, log):

        context = SCFEvent(dens=dens, ham=ham, wfs=wfs, niter=self.niter,
                           log=log)
        eigerr = self.criteria['eigenstates'].get_error(context)
        if not np.isfinite(eigerr):
            msg = 'Not enough bands for ' + wfs.eigensolver.nbands_converge
            log(msg, flush=True)
            raise KohnShamConvergenceError(msg)
        log(oops, flush=True)
        raise KohnShamConvergenceError(
            'Did not converge!  See text output for help.')

    def check_eigensolver_state(self, wfs, ham, dens):

        if self.eigensolver_used == 'etdm':
            wfs.eigensolver.eg_count = 0
            wfs.eigensolver.globaliters = 0
            wfs.eigensolver.check_assertions(wfs, dens)
            if wfs.eigensolver.dm_helper is None:
                wfs.eigensolver.initialize_dm_helper(wfs, ham, dens)

    def iterate_eigensolver(self, wfs, ham, dens):

        if self.eigensolver_used == 'etdm':
            wfs.eigensolver.iterate(ham, wfs, dens)
            wfs.eigensolver.check_mom(wfs, dens)
            e_entropy = 0.0
            kin_en_using_band = False
        else:
            wfs.eigensolver.iterate(ham, wfs)
            e_entropy = wfs.calculate_occupation_numbers(dens.fixed)
            kin_en_using_band = True

        ham.get_energy(e_entropy, wfs, kin_en_using_band=kin_en_using_band)

    def do_if_converged(self, wfs, ham, dens, log):

        if self.eigensolver_used == 'etdm':
            energy = ham.get_energy(0.0, wfs, kin_en_using_band=False)
            wfs.calculate_occupation_numbers(dens.fixed)
            wfs.eigensolver.get_canonical_representation(
                ham, wfs, dens, sort_eigenvalues=True)
            energy_converged = wfs.eigensolver.update_ks_energy(ham, wfs, dens)
            energy_diff_after_scf = abs(energy - energy_converged) * Ha
            if energy_diff_after_scf > 1.0e-6:
                warnings.warn('Jump in energy of %f eV detected at the end of '
                              'SCF after getting canonical orbitals, SCF '
                              'might have converged to the wrong solution '
                              'or achieved energy convergence to the correct '
                              'solution above 1.0e-6 eV'
                              % (energy_diff_after_scf))

            log('\nOccupied states converged after'
                ' {:d} e/g evaluations'.format(wfs.eigensolver.eg_count))

    def update_ham_and_dens(self, wfs, ham, dens):

        to_update = self.niter > self.niter_fixdensity and not dens.fixed
        if self.eigensolver_used == 'etdm' or not to_update:
            ham.npoisson = 0
        else:
            dens.update(wfs)
            ham.update(dens)


def write_iteration(criteria, converged_items, entries, ctx, log):
    custom = (set(criteria) -
              {'energy', 'eigenstates', 'density'})

    if ctx.niter == 1:
        header1 = ('     {:<4s} {:>8s} {:>12s}  '
                   .format('iter', 'time', 'total'))
        header2 = ('     {:>4s} {:>8s} {:>12s}  '
                   .format('', '', 'energy'))
        header1 += 'log10-change:'
        for title in ('eigst', 'dens'):
            header2 += '{:>5s}  '.format(title)
        for name in custom:
            criterion = criteria[name]
            header1 += ' ' * 7
            header2 += '{:>5s}  '.format(criterion.tablename)
        if ctx.wfs.nspins == 2:
            header1 += '{:>8s} '.format('magmom')
            header2 += '{:>8s} '.format('')
        log(header1.rstrip())
        log(header2.rstrip())

    c = {k: 'c' if v else ' ' for k, v in converged_items.items()}

    # Iterations and time.
    now = time.localtime()
    line = ('iter:{:4d} {:02d}:{:02d}:{:02d} '
            .format(ctx.niter, *now[3:6]))

    # Energy.
    line += '{:>12s}{:1s} '.format(entries['energy'], c['energy'])

    # Eigenstates.
    line += '{:>5s}{:1s} '.format(entries['eigenstates'], c['eigenstates'])

    # Density.
    line += '{:>5s}{:1s} '.format(entries['density'], c['density'])

    # Custom criteria (optional).
    for name in custom:
        line += '{:>5s}{:s} '.format(entries[name], c[name])

    # Magnetic moment (optional).
    if ctx.wfs.nspins == 2 or not ctx.wfs.collinear:
        totmom_v, _ = ctx.dens.calculate_magnetic_moments()
        if ctx.wfs.collinear:
            line += f'  {totmom_v[2]:+.4f}'
        else:
            line += ' {:+.1f},{:+.1f},{:+.1f}'.format(*totmom_v)

    log(line.rstrip(), flush=True)


class SCFEvent:
    """Object to pass the state of the SCF cycle to a convergence-checking
    function."""

    def __init__(self, dens, ham, wfs, niter, log):
        self.dens = dens
        self.ham = ham
        self.wfs = wfs
        self.niter = niter
        self.log = log


def get_criterion(name):
    """Returns one of the pre-specified criteria by it's .name attribute,
    and raises sensible error if missing."""
    # All built-in criteria should be in this list.
    criteria = [Energy, Density, Eigenstates, Forces, WorkFunction, MinIter]
    criteria = {c.name: c for c in criteria}
    try:
        return criteria[name]
    except KeyError:
        msg = ('The convergence keyword "{:s}" was supplied, which we do not '
               'know how to handle. If this is a typo, please correct. If this'
               ' is a user-written convergence criterion, it cannot be '
               'imported with this function; please see the GPAW manual for '
               'details.'.format(name))
        raise InputError(msg)


def dict2criterion(dictionary):
    """Converts a dictionary to a convergence criterion.

    The dictionary can either be that generated from 'todict'; that is like
    {'name': 'energy', 'tol': 0.005, 'n_old': 3}. Or from user-specified
    shortcut like {'energy': 0.005} or {'energy': (0.005, 3)}, or a
    combination like {'energy': {'name': 'energy', 'tol': 0.005, 'n_old': 3}.
    """
    d = dictionary.copy()
    if 'name' in d:  # from 'todict'
        name = d.pop('name')
        ThisCriterion = get_criterion(name)
        return ThisCriterion(**d)
    assert len(d) == 1
    name = list(d.keys())[0]
    if isinstance(d[name], dict) and 'name' in d[name]:
        return dict2criterion(d[name])
    ThisCriterion = get_criterion(name)
    return ThisCriterion(*[d[name]])


class Criterion:
    """Base class for convergence criteria.

    Automates the creation of the __repr__ and todict methods for generic
    classes. This will work for classes that save all arguments directly,
    like __init__(self, a, b):  --> self.a = a, self.b = b. The todict
    method requires the class have a self.name attribute. All criteria
    (subclasses of Criterion) must define self.name, self.tablename,
    self.description, self.__init__, and self.__call___. See the online
    documentation for details.
    """
    # If calc_last is True, will only be checked after all other (non-last)
    # criteria have been met.
    calc_last = False

    def __repr__(self):
        parameters = signature(self.__class__).parameters
        s = ', '.join([str(getattr(self, p)) for p in parameters])
        return self.__class__.__name__ + '(' + s + ')'

    def todict(self):
        d = {'name': self.name}
        parameters = signature(self.__class__).parameters
        for parameter in parameters:
            d[parameter] = getattr(self, parameter)
        return d

    def reset(self):
        pass


# Built-in criteria follow. Make sure that any new criteria added below
# are also added to to the list in get_criterion() so that it can import
# them correctly by name.


class Energy(Criterion):
    """A convergence criterion for the total energy.

    Parameters:

    tol : float
        Tolerance for conversion; that is the maximum variation among the
        last n_old values of the (extrapolated) total energy, normalized per
        valence electron. [eV/(valence electron)]
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
        # Note the previous code was calculating the peak-to-
        # peak energy difference on e_total_free, while reporting
        # e_total_extrapolated in the SCF table (logfile). I changed it to
        # use e_total_extrapolated for both. (Should be a miniscule
        # difference, but more consistent.)
        total_energy = context.ham.e_total_extrapolated * Ha
        if context.wfs.nvalence == 0:
            energy = total_energy
        else:
            energy = total_energy / context.wfs.nvalence
        self._old.append(energy)  # Pops off >3!
        error = np.inf
        if len(self._old) == self._old.maxlen:
            error = np.ptp(self._old)
        converged = error < self.tol
        entry = ''
        if np.isfinite(energy):
            entry = '{:11.6f}'.format(total_energy)
        return converged, entry


class Density(Criterion):
    """A convergence criterion for the electron density.

    Parameters:

    tol : float
        Tolerance for conversion; that is the maximum change in the electron
        density, calculated as the integrated absolute value of the density
        change, normalized per valence electron. [electrons/(valence electron)]
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
        if context.dens.fixed:
            return True, ''
        nv = context.wfs.nvalence
        if nv == 0:
            return True, ''
        # Make sure all agree on the density error.
        error = broadcast_float(context.dens.error, context.wfs.world) / nv
        converged = (error < self.tol)
        if (error is None or np.isinf(error) or error == 0):
            entry = ''
        else:
            entry = '{:+5.2f}'.format(np.log10(error))
        return converged, entry


class Eigenstates(Criterion):
    """A convergence criterion for the eigenstates.

    Parameters:

    tol : float
        Tolerance for conversion; that is the maximum change in the
        eigenstates, calculated as the integration of the square of the
        residuals of the Kohn--Sham equations, normalized per valence
        electron. [eV^2/(valence electron)]
    """
    name = 'eigenstates'
    tablename = 'eigst'

    def __init__(self, tol):
        self.tol = tol
        self.description = ('Maximum integral of absolute [eigenst]ate '
                            'change: {:g} eV^2 / valence electron'
                            .format(self.tol))

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        if context.wfs.nvalence == 0:
            return True, ''
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


class Forces(Criterion):
    """A convergence criterion for the forces.

    Parameters:

    tol : float
        Tolerance for conversion; that is, the force on each atom is compared
        with its force from the previous iteration, and the change in each
        atom's force is calculated as an l2-norm (Euclidean distance). The
        atom with the largest norm must be less than tol. [eV/Angstrom]
    calc_last : bool
        If True, calculates forces last; that is, it waits until all other
        convergence criteria are satisfied before checking to see if the
        forces have converged. (This is more computationally efficient.)
        If False, checks forces at each SCF step.
    """
    name = 'forces'
    tablename = 'force'

    def __init__(self, tol, calc_last=True):
        self.tol = tol
        self.description = ('Maximum change in the atomic [forces] across '
                            'last 2 cycles: {:g} eV/Ang'.format(self.tol))
        self.calc_last = calc_last
        self.reset()

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        if np.isinf(self.tol):  # criterion is off; backwards compatibility
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


class WorkFunction(Criterion):
    """A convergence criterion for the work function.

    Parameters:

    tol : float
        Tolerance for conversion; that is the maximum variation among the
        last n_old values of either work function. [eV]
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
        workfunctions = context.ham.get_workfunctions(context.wfs)
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


class MinIter(Criterion):
    """A convergence criterion that enforces a minimum number of iterations.

    Parameters:

    n : int
        Minimum number of iterations that must be complete before
        the SCF cycle exits.
    """
    calc_last = False
    name = 'minimum iterations'
    tablename = 'minit'

    def __init__(self, n):
        self.n = n
        self.description = f'Minimum number of iterations [minit]: {n:d}'

    def __call__(self, context):
        converged = context.niter >= self.n
        entry = '{:d}'.format(context.niter)
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
