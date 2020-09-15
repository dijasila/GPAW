# creates: h2o-symmetries.txt
from gpaw import GPAW
from ase.build import molecule

# Ground state calculation:
atoms = molecule('H2O')
atoms.center(vacuum=2.5)
atoms.calc = GPAW(mode='lcao', txt='h2o.txt')
e = atoms.get_potential_energy()
# Analyze symmetry:
from gpaw.point_groups import SymmetryChecker
checker = SymmetryChecker('C2v', atoms.positions[0], radius=2.0)
for n in range(4):
    result = checker.check_band(atoms.calc, n)
    print(n, result['symmetry'])
# Write wave functions for later analysis:
atoms.calc.write('h2o.gpw', mode='all')
# Create file for docs:
checker.check_calculation(atoms.calc, n1=0, n2=4, output='h2o-symmetries.txt')
