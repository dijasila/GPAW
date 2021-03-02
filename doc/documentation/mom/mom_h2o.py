import copy
from ase.build import molecule
from gpaw import GPAW, restart
from gpaw import mom

atoms = molecule('H2O')
atoms.center(vacuum=7)

calc = GPAW(mode='fd',
            basis='dzp',
            nbands=6,
            h=0.2,
            xc='PBE',
            spinpol=True,
            symmetry='off',
            convergence={'bands': -1})
atoms.calc = calc

# Ground-state calculation
E_gs = atoms.get_potential_energy()

# Save the ground-state wave functions for initialization
# of the spin-mixed excited-state calculation later
calc.write('h2o_fd_gs.gpw', 'all')

# Ground-state occupation numbers
f_n_gs = []
for s in range(2):
    f_n_gs.append(calc.get_occupation_numbers(spin=s))

# Triplet homo-1->3s occupation numbers
f_n_t = copy.deepcopy(f_n_gs)
f_n_t[0][2] -= 1.  # Remove one electron from homo-1 spin up
f_n_t[1][4] += 1.  # Add one electron to lumo (3s) spin down

# MOM calculation for triplet n->3s state
mom.mom_calculation(calc, atoms, f_n_t)
E_t = atoms.get_potential_energy()

atoms, calc = restart('h2o_fd_gs.gpw', txt='-')

# Mixed-spin homo-1->3s occupation numbers
f_n_m = copy.deepcopy(f_n_gs)
f_n_m[0][2] -= 1.  # Remove one electron from homo-1 spin up
f_n_m[0][4] += 1.  # Add one electron to lumo (3s) spin up

# MOM calculation for spin-mixed homo-1->3s state
mom.mom_calculation(calc, atoms, f_n_m)
E_m = atoms.get_potential_energy()

print('Excitation energy triplet homo-1->3s state: %.2f eV' %(E_t - E_gs))
print('Excitation energy singlet homo-1->3s state: %.2f eV' %(2 * E_m - E_gs))
print('Experimental excitation energy triplet homo-1->3s state: 9.46 eV')
print('Experimental excitation energy singlet homo-1->3s state: 9.67 eV')
