from ase import *
from gpaw import Calculator
from build import fcc100


a = 4.05
fcc = fcc100('Al', a, 2, 10.0)

# Add the adsorbate:
fcc.append(Atom('H', (0, 0, 1.55)))

calc = Calculator(nbands=2 * 5,
                  kpts=(4, 4, 1),
                  h = 0.25)
fcc.set_calculator(calc)

# Make a trajectory file:
traj = PickleTrajectory('ontop.traj', 'w', fcc)

# Only the height (z-coordinate) of the H atom is relaxed:
fcc.set_constraint(FixAtoms(range(4))

# The energy minimum is found using the "Molecular Dynamics
# Minimization" algorithm.  The stopping criteria is: the force on the
# H atom should be less than 0.05 eV/Ang
dyn = QuasiNewton(fcc)

dyn.attach(traj)
dyn.run(fmax=0.05)

calc.write('relax.gpw') # Write gpw output after the minimization

print 'ontop:', fcc.get_potential_energy()
print 'height:', fcc.position[-1, 2]
