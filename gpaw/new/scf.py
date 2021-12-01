import itertools
import warnings
from functools import partial
from math import inf
from types import SimpleNamespace

from gpaw.new.density import Density
from gpaw.new.potential import Potential
from gpaw.new.wave_functions import IBZWaveFunctions
from gpaw.scf import dict2criterion, write_iteration


class SCFConvergenceError(Exception):
    ...


class SCFLoop:
    def __init__(self,
                 hamiltonian,
                 pot_calc,
                 occ_calc,
                 eigensolver,
                 mixer,
                 world,
                 convergence,
                 maxiter):
        self.hamiltonian = hamiltonian
        self.pot_calc = pot_calc
        self.eigensolver = eigensolver
        self.mixer = mixer
        self.occ_calc = occ_calc
        self.world = world
        self.convergence = convergence
        self.maxiter = maxiter

    def __str__(self):
        return str(self.pot_calc)

    def iterate(self,
                ibzwfs: IBZWaveFunctions,
                density: Density,
                potential: Potential,
                convergence=None,
                maxiter=None,
                log=None):
        cc = create_convergence_criteria(convergence or self.convergence)
        maxiter = maxiter or self.maxiter

        dS = density.overlap_correction

        self.mixer.reset()

        dens_error = inf
        # dens_error = self.mixer.mix(density)
        for niter in itertools.count(1):
            dH = potential.dH
            Ht = partial(self.hamiltonian.apply, potential.vt_sR)
            wfs_error = self.eigensolver.iterate(ibzwfs, Ht, dH, dS)
            ibzwfs.calculate_occs(self.occ_calc)

            ctx = SCFContext(
                ibzwfs, density, potential, niter,
                wfs_error, dens_error,
                self.world)

            yield ctx

            entries, converged = check_convergence(ctx, cc)
            if log:
                write_iteration(cc, converged, entries, ctx, log)
            if all(converged.values()):
                break
            if niter == maxiter:
                raise SCFConvergenceError

            ibzwfs.calculate_density(out=density)
            dens_error = self.mixer.mix(density)
            potential = self.pot_calc.calculate(density)


class SCFContext:
    def __init__(self, ibzwfs, density, potential,
                 niter: int,
                 wfs_error: float,
                 dens_error: float,
                 world):
        self.density = density
        self.potential = potential
        self.niter = niter
        energy = (sum(potential.energies.values()) +
                  sum(ibzwfs.energies.values()))
        self.ham = SimpleNamespace(e_total_extrapolated=energy)
        self.wfs = SimpleNamespace(nvalence=ibzwfs.nelectrons,
                                   world=world,
                                   eigensolver=SimpleNamespace(
                                       error=wfs_error),
                                   nspins=density.ndensities,
                                   collinear=density.collinear)
        self.dens = SimpleNamespace(fixed=False, error=dens_error)


def check_convergence(ctx, criteria):
    entries = {}  # for log file, per criteria
    converged_items = {}  # True/False, per criteria

    for name, criterion in criteria.items():
        if not criterion.calc_last:
            converged_items[name], entries[name] = criterion(ctx)

    converged = all(converged_items.values())

    for name, criterion in criteria.items():
        if criterion.calc_last:
            if converged:
                converged_items[name], entries[name] = criterion(ctx)
            else:
                converged_items[name], entries[name] = False, ''

    # Converged?
    return entries, converged_items


def create_convergence_criteria(criteria):
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
