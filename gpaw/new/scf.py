from __future__ import annotations

import itertools
import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable

from gpaw.convergence_criteria import (Criterion, check_convergence,
                                       dict2criterion)
from gpaw.scf import write_iteration
from gpaw.typing import Array2D
from gpaw.yml import indent

if TYPE_CHECKING:
    from gpaw.new.calculation import DFTState


class SCFConvergenceError(Exception):
    ...


class SCFLoop:
    def __init__(self,
                 hamiltonian,
                 occ_calc,
                 eigensolver,
                 mixer,
                 world,
                 convergence,
                 maxiter):
        self.hamiltonian = hamiltonian
        self.eigensolver = eigensolver
        self.mixer = mixer
        self.occ_calc = occ_calc
        self.world = world
        self.convergence = convergence
        self.maxiter = maxiter
        self.niter = 0
        self.update_density_and_potential = True
        self.observers = []

    def __repr__(self):
        return 'SCFLoop(...)'

    def __str__(self):
        return (f'eigensolver:\n{indent(self.eigensolver)}\n'
                f'{self.mixer}\n'
                f'occupation numbers:\n{indent(self.occ_calc)}\n')

    def iterate(self,
                state: DFTState,
                pot_calc,
                convergence=None,
                maxiter=None,
                calculate_forces=None,
                log=None):

        cc = create_convergence_criteria(convergence or self.convergence)
        maxiter = maxiter or self.maxiter

        if log:
            log('convergence criteria:')
            for criterion in cc.values():
                if criterion.description is not None:
                    log('- ' + criterion.description)
            log(f'maximum number of iterations: {self.maxiter}\n')

        self.mixer.reset()

        if self.update_density_and_potential:
            dens_error, magn_error = self.mixer.mix(state.density)
        else:
            dens_error, magn_error = 0.0

        for self.niter in itertools.count(start=1):
            wfs_error = self.eigensolver.iterate(state, self.hamiltonian)
            state.ibzwfs.calculate_occs(
                self.occ_calc,
                fixed_fermi_level=not self.update_density_and_potential)

            ctx = SCFContext(
                state, self.niter,
                wfs_error, dens_error, magn_error,
                self.world, calculate_forces)

            yield ctx

            converged, converged_items, entries = check_convergence(cc, ctx)
            if log:
                with log.comment():
                    write_iteration(cc, converged_items, entries, ctx, log)
            self.call_observers(self.niter)

            if converged:
                break
            if self.niter == maxiter:
                raise SCFConvergenceError

            if self.update_density_and_potential:
                state.density.update(pot_calc.nct_R, state.ibzwfs)
                dens_error, magn_error = self.mixer.mix(state.density)
                state.potential, state.vHt_x, _ = pot_calc.calculate(
                    state.density, state.vHt_x)

            self.call_observers(self.niter, final=True)

    def call_observers(self, iter, final=False):
        """Call all registered callback functions."""
        for function, n, args, kwargs in self.observers:
            call = False
            # Call every n iterations, including the last
            if n > 0:
                if ((iter % n) == 0) != final:
                    call = True
            # Call only on iteration n
            elif n < 0 and not final:
                if iter == abs(n):
                    call = True
            # Call only on convergence
            elif n == 0 and final:
                call = True
            if call:
                if isinstance(function, str):
                    function = getattr(self, function)
                # Replace self reference with self
                self_ = self.self_ref
                args = tuple([self if arg is self_ else arg for arg in args])
                function(*args, **kwargs)

    def attach(self, function, n=1, *args, **kwargs):
        """Register observer function to run during the SCF cycle.

        Call *function* using *args* and
        *kwargs* as arguments.

        If *n* is positive, then
        *function* will be called every *n* SCF iterations + the
        final iteration if it would not be otherwise

        If *n* is negative, then *function* will only be
        called on iteration *abs(n)*.

        If *n* is 0, then *function* will only be called
        on convergence"""

        try:
            slf = function.__self__
        except AttributeError:
            pass
        else:
            if slf is self:
                # function is a bound method of self.  Store the name
                # of the method and avoid circular reference:
                function = function.__func__.__name__

        # Replace self in args with another unique reference
        # to avoid circular reference
        if not hasattr(self, 'self_ref'):
            self.self_ref = object()
        self_ = self.self_ref
        args = tuple([self_ if arg is self else arg for arg in args])

        self.observers.append((function, n, args, kwargs))


class SCFContext:
    def __init__(self,
                 state: DFTState,
                 niter: int,
                 wfs_error: float,
                 dens_error: float,
                 magn_error: float,
                 world,
                 calculate_forces: Callable[[], Array2D]):
        self.state = state
        self.niter = niter
        energy = (sum(state.potential.energies.values()) +
                  sum(state.ibzwfs.energies.values()))
        self.ham = SimpleNamespace(e_total_extrapolated=energy)
        self.wfs = SimpleNamespace(nvalence=state.ibzwfs.nelectrons,
                                   world=world,
                                   eigensolver=SimpleNamespace(
                                       error=wfs_error),
                                   nspins=state.density.ndensities,
                                   collinear=state.density.collinear)
        self.dens = SimpleNamespace(
            calculate_magnetic_moments=state.density
            .calculate_magnetic_moments,
            fixed=False,
            error=dens_error, merror=magn_error)
        self.calculate_forces = calculate_forces


def create_convergence_criteria(criteria: dict[str, Any]
                                ) -> dict[str, Criterion]:
    for k, v in [('energy', 0.0005),        # eV / electron
                 ('density', 1.0e-4),       # electrons / electron
                 ('eigenstates', 4.0e-8)]:  # eV^2 / electron
        if k not in criteria:
            criteria[k] = v
    # Gather convergence criteria for SCF loop.
    custom = criteria.pop('custom', [])
    for name, criterion in criteria.items():
        if hasattr(criterion, 'todict'):
            # 'Copy' so no two calculators share an instance.
            criteria[name] = dict2criterion(criterion.todict())
        else:
            criteria[name] = dict2criterion({name: criterion})

    if not isinstance(custom, (list, tuple)):
        custom = [custom]
    for criterion in custom:
        if isinstance(criterion, dict):  # from .gpw file
            msg = ('Custom convergence criterion "{:s}" encountered, '
                   'which GPAW does not know how to load. This '
                   'criterion is NOT enabled; you may want to manually'
                   ' set it.'.format(criterion['name']))
            warnings.warn(msg)
            continue

        criteria[criterion.name] = criterion
        msg = ('Custom convergence criterion {:s} encountered. '
               'Please be sure that each calculator is fed a '
               'unique instance of this criterion. '
               'Note that if you save the calculator instance to '
               'a .gpw file you may not be able to re-open it. '
               .format(criterion.name))
        warnings.warn(msg)

    for criterion in criteria.values():
        criterion.reset()

    return criteria
