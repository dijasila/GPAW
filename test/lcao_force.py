# This tests calculates the force on the atoms of a small molecule.
#
# If the test fails, set the fd boolean below to enable a (costly) finite
# difference check.

import numpy as np
from ase.data.molecules import molecule
from gpaw import GPAW
from gpaw.atom.basis import BasisMaker

obasis = BasisMaker('O').generate(2, 1)
hbasis = BasisMaker('H').generate(2, 1)
basis = {'O' : obasis, 'H' : hbasis}

system = molecule('H2O')
system.center(vacuum=1.5)
system.rattle(stdev=.2, seed=42)
system.set_pbc(1)

calc = GPAW(h=0.2,
            mode='lcao',
            basis=basis,
            kpts=[(0., 0., 0.), (.3, .1, .4)],
            convergence={'density':1e-5}
            )

system.set_calculator(calc)

F_ac = system.get_forces()


# Previous FD result, generated by disabled code below
#F_ac_ref = np.array([[ 1.02663971,  1.77315595, -4.37292514],
#                     [-0.72530244, -0.91067585,  2.95066214],
#                     [-0.29580753, -0.83924051,  1.39731849]])
F_ac_ref = np.array([[ 1.03186879,  1.64447806, -4.82345575],
                     [-0.70540087, -0.89132284,  3.0395603 ],
                     [-0.32250709, -0.7297437,   1.75381613]])



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
assert err < 2e-3

# Set boolean to run new FD check
fd = 0

if fd:
    from ase.calculators import numeric_force

    F_ac_fd = np.zeros(F_ac.shape)
    
    for b in range(3):
        for v in range(3):
            F_ac_fd[b, v] = numeric_force(system, b, v)

    print 'Self-consistent forces'
    print F_ac
    print 'FD'
    print F_ac_fd

    assert np.abs(F_ac - F_ac_fd).max() < 4e-3
