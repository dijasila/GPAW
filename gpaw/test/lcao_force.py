# This tests calculates the force on the atoms of a small molecule.
#
# If the test fails, set the fd boolean below to enable a (costly) finite
# difference check.

import numpy as np
from ase.structure import molecule
from gpaw import GPAW
from gpaw.atom.generator2 import generate


hbasis = generate(['H']).create_basis_set(tailnorm=0.005)
obasis = generate(['O']).create_basis_set(tailnorm=0.005)
basis = {'O' : obasis, 'H' : hbasis}

system = molecule('H2O')
system.center(vacuum=1.5)
system.rattle(stdev=.2, seed=42)
system.set_pbc(1)

calc = GPAW(h=0.2,
            mode='lcao',
            basis=basis,
            kpts=[(0., 0., 0.), (.3, .1, .4)],
            convergence={'density':1e-5, 'energy': 1e-6}
            )

system.set_calculator(calc)

F_ac = system.get_forces()


# Previous FD result, generated by disabled code below
F_ac_ref = np.array([[ 1.49759077,  1.73841543, -6.70403376],
                     [-0.91202797, -1.55756761,  3.8283681 ],
                     [-0.5866586 , -0.1465678 ,  2.72455111]])

err_ac = np.abs(F_ac - F_ac_ref)
err = err_ac.max()

print 'Force'
print F_ac
print
print 'Reference result'
print F_ac_ref
print
print 'Error'
print err_ac
print
print 'Max error'
print err

# ASE uses dx = [+|-] 0.001 by default,
# error should be around 2e-3.  In fact 4e-3 would probably be acceptable
assert err < 3e-2

# Set boolean to run new FD check
fd = False

if fd:
    from ase.calculators.test import numeric_forces
    F_ac_fd = numeric_forces(system)
    print 'Self-consistent forces'
    print F_ac
    print 'FD'
    print F_ac_fd
    print repr(F_ac_fd)
    print F_ac - F_ac_fd, np.abs(F_ac - F_ac_fd).max()

    assert np.abs(F_ac - F_ac_fd).max() < 4e-3
