from ase.build import molecule
from ase.optimize import LBFGS
from gpaw import GPAW, mom

atoms = molecule('CO')
atoms.center(vacuum=5)

calc = GPAW(mode='lcao',
            basis='dzp',
            nbands=8,
            h=0.2,
            xc='PBE',
            symmetry='off',
            convergence={'energy': 100,
                         'density': 1e-7})

atoms.calc = calc
# Ground-state calculation
E_gs = atoms.get_potential_energy()

f = [calc.get_occupation_numbers(spin=0) / 2.]
f[0][4] -= 0.5
f[0][5] += 0.5

# Excited-state MOM calculation with Gaussian
# smearing of the occupation numbers
mom.mom_calculation(calc, atoms, f, width=0.01)

opt = LBFGS(atoms)
opt.run(fmax=0.05)

d = atoms.get_distance(0, 1)

print('Optimized C-O bondlength: %.2f Å' % (d))
# Experimental value from
print('Experimental C-O bondlength: 1.24 Å')

