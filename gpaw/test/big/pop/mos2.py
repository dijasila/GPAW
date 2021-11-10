"""MoS2 slab.

Parallelization is done over:

* k-points (embarrassingly parallel)
* electronic bands
* domain

and we must have K*B*D=world.size.

Problem size can be tweaked by adjusting N and ecut.

The calculation has several sections:

* importing modules
* first iteration (contains initialization stuff)
* more iterations (should all be similar)
* forces
* stress tensor

"""
from time import time
from ase.build import mx2
from gpaw import GPAW
from gpaw.mpi import world

K = 8
B = 1
D = world.size // (K * B)
assert K * D * B == world.size

N = 6
atoms = mx2('MoS2')
atoms.center(vacuum=4.0, axis=2)
atoms *= (N, N, 1)
atoms.rattle(stdev=0.01)

# from ase.visualize import view
# view(atoms)

ecut = 600
atoms.calc = GPAW(
    kpts=(4, 4, 1),
    xc='PBE',
    txt=f'MoS2-{N}x{N}-k{K}-d{D}-b{B}.txt',
    parallel={
        'domain': D,
        'band': B,
        'kpt': K,
        'sl_auto': True},
    mode={'name': 'pw', 'ecut': ecut})

t = time()
for iter, _ in enumerate(atoms.calc.icalculate(atoms)):
    if world.rank == 0:
        print(iter, time() - t)
        t = time()
    if iter == 4:
        atoms.calc.scf.converged = True
        break

atoms.get_forces()
atoms.get_stress()
