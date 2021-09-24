from functools import partial
from types import SimpleNamespace


class SCFLoop:
    def __init__(self,
                 ibz_wfs,
                 density,
                 potential,
                 hamiltonian,
                 pot_calc,
                 eigensolver,
                 mixer):
        self.ibz_wfs = ibz_wfs
        self.density = density
        self.potential = potential
        self.hamiltonian = hamiltonian
        self.eigensolver = eigensolver
        self.occs_calc = ...
    def converge(self, conv_criteria):
        for _ in self.iconverge(conv_criteria):
            pass

    def iconverge(self, conv_criteria):
        dS = self.density.setups.overlap_correction
        dH = self.potential.dH
        Ht = partial(self.hamiltonian.apply, self.potential.vt)
        for i in range(999_999_999_999_999):
            self.eigensolver.iterate(self.ibz_wfs, Ht, dH, dS)
            self.ibz_wfs.calculate_occs(self.occs_calc)
            energy = calculate_energy(self.occs_calc,
                                      self.ibz_wfs,
                                      self.potential)
            ctx = SCFEvent(energy, self.ibz_wfs)
            entries, converged = check_convergence(ctx, conv_criteria)
            yield entries
            if all(converged.values()):
                break


def calculate_energy(occs_calc, ibz_wfs, potential):
    return sum(potential.energies.values())


class SCFEvent:
    def __init__(self, energy, ibz_wfs):
        self.ham = SimpleNamespace(e_total_extrapolated=energy)
        self.wfs = SimpleNamespace(nvalence=ibz_wfs.nvalence)


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
