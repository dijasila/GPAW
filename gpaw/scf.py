import datetime
from math import log10, nan, inf

import numpy as np
from ase.units import Bohr, Ha

from gpaw import KohnShamConvergenceError
from gpaw.forces import calculate_forces
from gpaw.mpi import broadcast_float


class SCFLoop:
    """Self-consistent field loop."""
    def __init__(self, eigenstates=0.1, energy=0.1, density=0.1, force=np.inf,
                 maxiter=100, niter_fixdensity=None, nvalence=None):
        self.max_errors = {'eigenstates': eigenstates,
                           'energy': energy,
                           'force': force,
                           'density': density}
        self.maxiter = maxiter
        self.niter_fixdensity = niter_fixdensity
        self.nvalence = nvalence

        self.old_energies = []
        self.old_F_av = None
        self.converged = False

        self.niter = None

        self.reset()

    def __str__(self):
        cc = self.max_errors
        s = 'Convergence criteria:\n'
        for name, val in [
            ('total energy change: {0:g} eV / electron',
             cc['energy'] * Ha / self.nvalence),
            ('integral of absolute density change: {0:g} electrons',
             cc['density'] / self.nvalence),
            ('integral of absolute eigenstate change: {0:g} eV^2',
             cc['eigenstates'] * Ha**2 / self.nvalence),
            ('change in atomic force: {0:g} eV / Ang',
             cc['force'] * Ha / Bohr),
            ('number of iterations: {0}', self.maxiter)]:
            if val < np.inf:
                s += '  Maximum {0}\n'.format(name.format(val))
        return s

    def write(self, writer):
        writer.write(converged=self.converged)

    def read(self, reader):
        self.converged = reader.scf.converged

    def reset(self):
        self.old_energies = []
        self.old_F_av = None
        self.converged = False

    def irun(self, wfs, ham, dens, log, callback):
        self.niter = 1

        log("""\
            ---- log10-errors -----      total
            time      wfs    density    force      energy    magmom""")

        with log.table('iterations'):
            while self.niter <= self.maxiter:
                wfs.eigensolver.iterate(ham, wfs)
                e_entropy = wfs.calculate_occupation_numbers(dens.fixed)
                energy = ham.get_energy(e_entropy, wfs)
                self.old_energies.append(energy)
                errors = self.collect_errors(dens, ham, wfs)

                # Converged?
                for kind, error in errors.items():
                    if error > self.max_errors[kind]:
                        self.converged = False
                        break
                else:
                    self.converged = True

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

        # Don't fix the density in the next step:
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
                  'energy': np.inf}

        if dens.fixed:
            errors['density'] = 0.0

        if len(self.old_energies) >= 3:
            energies = self.old_energies[-3:]
            if np.isfinite(energies).all():
                errors['energy'] = np.ptp(energies)

        # We only want to calculate the (expensive) forces if we have to:
        check_forces = (self.max_errors['force'] < np.inf and
                        all(error <= self.max_errors[kind]
                            for kind, error in errors.items()))

        errors['force'] = inf
        if check_forces:
            with wfs.timer('Forces'):
                F_av = calculate_forces(wfs, dens, ham)
            if self.old_F_av is not None:
                errors['force'] = ((F_av - self.old_F_av)**2).sum(1).max()**0.5
            self.old_F_av = F_av

        return errors

    def log(self, log, niter, wfs, ham, dens, errors):
        """Output from each iteration."""
        nvalence = wfs.nvalence

        log10errors = [
            log10(x) if x != 0 else nan
            for x in [
                errors['eigenstates'] * Ha**2 / nvalence if nvalence else nan,
                errors['density'] / nvalence if nvalence else nan,
                errors['force'] * Ha / Bohr if errors['force'] < inf else nan]]

        if wfs.nspins == 2 or not wfs.collinear:
            totmom_v, _ = dens.estimate_magnetic_moments()
            if wfs.collinear:
                mom = (totmom_v[2], '+.4f')
            else:
                mom = (totmom_v, '+.1f')
        else:
            mom = 0

        log.row([niter,
                 datetime.datetime.now().time(),
                 *log10errors,
                 Ha * ham.e_total_extrapolated,
                 mom],
                widths=[3, 8, 5, 5, 5, 11, 18])
        log.flush()


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
