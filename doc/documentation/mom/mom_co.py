from ase.build import molecule
from ase.optimize import LBFGS
from gpaw import GPAW
from gpaw.mom import mom_calculation


atoms = molecule('CO')
atoms.center(vacuum=5)

calc = GPAW(mode='lcao',
            basis='dzp',
            nbands=8,
            h=0.2,
            xc='PBE',
            symmetry='off',
            convergence={'density': 1e-7},
            txt='co.txt')
atoms.calc = calc

# Ground-state calculation
E_gs = atoms.get_potential_energy()

# Ground-state occupation numbers
f = [calc.get_occupation_numbers(spin=0) / 2.]

# Singlet sigma->pi* occupation numbers
f[0][4] -= 0.5
f[0][5] += 0.5

# Excited-state MOM calculation with Gaussian
# smearing of the occupation numbers
mom_calculation(calc, atoms, f, width=0.01)

opt = LBFGS(atoms)
opt.run(fmax=0.05)

d = atoms.get_distance(0, 1)

print('Optimized C-O bond length of sigma->pi* state: %.2f Å' % (d))
# https://doi.org/10.1007/978-1-4757-0961-2
print('Experimental C-O bond length of sigma->pi* state: 1.24 Å')
