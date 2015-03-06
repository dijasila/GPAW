# Test that the atomic corrections of LCAO work correctly,
# by verifying that the different implementations yield the same numbers.
#
# For example the corrections P^* dH P to the Hamiltonian.
#
# This is done by invoking GPAW once for each type of calculation.

from ase.structure import molecule
from gpaw import GPAW, LCAO, PoissonSolver
from gpaw.lcao.atomic_correction import DenseAtomicCorrection, \
    DistributedAtomicCorrection, ScipyAtomicCorrection

# Use a cell large enough that some overlaps are zero.
# Thus the matrices will have at least some sparsity.
system = molecule('H2O')
system.center(vacuum=3.0)
system.pbc = (0, 0, 1)
system = system.repeat((1, 1, 2))

corrections = [DenseAtomicCorrection(), DistributedAtomicCorrection()]

try:
    import scipy
except ImportError:
    pass
else:
    corrections.append(ScipyAtomicCorrection(tolerance=0.0))

energies = []
for correction in corrections:
    parallel = dict()
    if correction.name != 'dense':
        parallel=dict(sl_auto=True)
    calc = GPAW(mode=LCAO(atomic_correction=correction),
                basis='sz(dzp)',
                poissonsolver=PoissonSolver(relax='GS', eps=1e100, nn=1),
                parallel=parallel,
                h=0.35)
    def stopcalc():
        calc.scf.converged = True
    calc.attach(stopcalc, 1)
    system.set_calculator(calc)
    energy = system.get_potential_energy()
    energies.append(energy)

master = calc.wfs.world.rank == 0
if master:
    print 'energies', energies

eref = energies[0]
errs = []
for energy, c in zip(energies, corrections):
    err = abs(energy - eref)
    errs.append(err)
    nops = calc.wfs.world.sum(c.nops)
    if master:
        print 'err=%e :: name=%s :: nops=%d' % (err, c.name, nops)

assert max(errs) < 1e-12
