from gpaw import Calculator
from ase import *

a = 4.23
atoms = Crystal([Atom('Na', (0, 0, 0)),
                 Atom('Na', (0.5, 0.5, 0.5))])
atoms.set_cell((a, a, a), fix=False)

# Make self-consistent calculation and save results
h = 0.25
calc = Calculator(h=.25, kpts=(8, 8, 8), width=0.05,
                  nbands=3, txt='Na_sc.txt')
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Na_sc.gpw')

# Calculate band structure along Gamma-X i.e. from 0 to 0.5
nkpt = 50
kpts = [(k / float(2 * nkpt), 0, 0) for k in range(nkpt)]
print kpts
calc = Calculator('Na_sc.gpw', txt='Na_harris.txt',
                  kpts=kpts, fixdensity=True, nbands=5,
                  eigensolver='cg')
calc.get_potential_energy()

# Write the results to a file e.g. for plotting with gnuplot
f = open('Na_bands.dat', 'w')
for k, kpt_c in enumerate(calc.get_i_b_z_k_points()):
    for eig in calc.get_eigenvalues(kpt=k):
        print >> f, kpt_c[0], eig - calc.get_fermi_level()
