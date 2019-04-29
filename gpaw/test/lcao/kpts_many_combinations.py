from ase.build import bulk
from gpaw import GPAW
from gpaw.wavefunctions.lcao import LCAO
from gpaw.mpi import world

def getcalc(**kwargs):
    return calc

kwargs = []
energies = []
eerrs = []
forces = []
ferrs = []


def ikwargs():
    for spinpol in [False, True]:
        for augment_grids in [False, True]:
            for atomic_correction in ['dense', 'scipy']:
                mode = LCAO(atomic_correction=atomic_correction)
                for sl_auto in [False, True]:
                    for kpt in [1, 2, 4]:
                        if kpt == 4:
                            continue
                        for band in [1, 2, 4]:
                            for domain in [1, 2, 4]:
                                if kpt * band * domain != world.size:
                                    continue
                                parallel = dict(kpt=kpt,
                                                band=band,
                                                domain=domain,
                                                sl_auto=sl_auto,
                                                augment_grids=augment_grids)
                                yield dict(parallel=parallel,
                                           mode=mode,
                                           spinpol=spinpol)


# We want a non-trivial cell:
atoms = bulk('Ti') * (2, 2, 1)
atoms.cell[0] *= 1.04
# We want most arrays to be different so we can detect ordering/shape trouble:
atoms.symbols = 'Ti2HFeCAuPbO'
atoms.rattle(stdev=0.1)


for i, kwargs in enumerate(ikwargs()):
    if world.rank == 0:
        print(i, kwargs)
    calc = GPAW(basis='szp(dzp)', xc='oldPBE', h=0.3,
                txt='gpaw.{:02d}.txt'.format(i),
                kpts=(4, 1, 1), **kwargs)
    def stopcalc():
        calc.scf.converged = True
    calc.attach(stopcalc, 3)
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    energies.append(e)
    forces.append(f)

    if energies:
        eerr = abs(e - energies[0])
        ferr = abs(f - forces[0]).max()
        eerrs.append(eerr)
        ferrs.append(ferr)
        if world.rank == 0:
            print('Eerr {} Ferr {}'.format(eerr, ferr))
            print()

if world.rank == 0:
    print('eerrs', eerrs)
    print('ferrs', ferrs)

    print('maxeerr', max(eerrs))
    print('maxferr', max(ferrs))
