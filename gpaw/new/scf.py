from functools import partial
from types import SimpleNamespace
from gpaw.scf import write_iteration
import warnings
from gpaw.scf import dict2criterion
from gpaw.new.wave_functions import IBZWaveFunctions
from gpaw.new.potential import Potential
from gpaw.new.density import Density


class SCFLoop:
    def __init__(self,
                 hamiltonian,
                 pot_calc,
                 occ_calc,
                 eigensolver,
                 mixer,
                 world):
        self.hamiltonian = hamiltonian
        self.pot_calc = pot_calc
        self.eigensolver = eigensolver
        self.mixer = mixer
        self.occ_calc = occ_calc
        self.world = world

    def iterate(self,
                ibz_wfs: IBZWaveFunctions,
                density: Density,
                potential: Potential):
        dS = density.setups.overlap_correction
        dH = potential.dH
        Ht = partial(self.hamiltonian.apply, potential.vt)

        niter = 0
        while True:
            wfs_error = self.eigensolver.iterate(ibz_wfs, Ht, dH, dS)
            ibz_wfs.calculate_occs(self.occ_calc)
            ibz_wfs.calculate_density(out=density)
            dens_error = 0#self.mix(density)
            potential = self.pot_calc.calculate(density)

            ctx = SCFContext(ibz_wfs, density, potential, niter,
                             wfs_error, dens_error, self.world)
            yield ctx
            niter += 1

    def converge(self,
                 ibz_wfs,
                 density,
                 potential,
                 convergence=None,
                 maxiter=99,
                 log=None):
        cc = create_convergence_criteria(convergence)
        for ctx in self.iterate(ibz_wfs, density, potential):
            entries, converged = check_convergence(ctx, cc)
            if log:
                write_iteration(cc, converged, entries, ctx, log)
            if all(converged.values()):
                break
            if ctx.niter == maxiter:
                raise ...
        return ctx.density, ctx.potential


def calculate_energy(ibz_wfs: IBZWaveFunctions,
                     potential: Potential) -> float:
    return (sum(potential.energies.values()) +
            ibz_wfs.e_band +
            ibz_wfs.e_entropy)


class SCFContext:
    def __init__(self, ibz_wfs, density, potential, niter,
                 wfs_error, dens_error, world):
        self.density = density
        self.potential = potential
        self.niter = niter
        energy = calculate_energy(ibz_wfs, potential)
        self.ham = SimpleNamespace(e_total_extrapolated=energy)
        self.wfs = SimpleNamespace(nvalence=ibz_wfs.nelectrons,
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
