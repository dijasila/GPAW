from ase.build import molecule
from ase.optimize import LBFGS
from ase.parallel import paropen
from gpaw import GPAW
from gpaw.mom import prepare_mom_calculation


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
f[0][4] -= 0.5  # Remove one electron from homo (sigma)
f[0][5] += 0.5  # Add one electron to lumo (pi*)

# Excited-state MOM calculation with Gaussian
# smearing of the occupation numbers
prepare_mom_calculation(calc, atoms, f, width=0.01)

opt = LBFGS(atoms, logfile='co.log')
opt.run(fmax=0.05)

d = atoms.get_distance(0, 1)

with paropen('co.log', 'a') as fd:
    print(f'Optimized C-O bond length of sigma->pi* state: {d:.2f} Å', file=fd)
    # https://doi.org/10.1007/978-1-4757-0961-2
    print('Experimental C-O bond length of sigma->pi* state: 1.24 Å', file=fd)
