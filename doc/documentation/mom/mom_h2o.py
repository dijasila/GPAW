import copy
from ase.build import molecule
from gpaw import GPAW, restart
from gpaw.mom import mom_calculation

atoms = molecule('H2O')
atoms.center(vacuum=7)

calc = GPAW(mode='fd',
            basis='dzp',
            nbands=6,
            h=0.2,
            xc='PBE',
            spinpol=True,
            symmetry='off',
            convergence={'bands': -1},
            txt='h2o.txt')
atoms.calc = calc

# Ground-state calculation
E_gs = atoms.get_potential_energy()

# Save the ground-state wave functions for initialization
# of the mixed-spin excited-state calculation later
calc.write('h2o_fd_gs.gpw', 'all')

# Ground-state occupation numbers
f_gs = []
for s in range(2):
    f_gs.append(calc.get_occupation_numbers(spin=s))

# Triplet n->3s occupation numbers
f_t = copy.deepcopy(f_gs)
f_t[0][2] -= 1.  # Remove one electron from homo-1 (n) spin up
f_t[1][4] += 1.  # Add one electron to lumo (3s) spin down

# MOM calculation for triplet n->3s state
mom_calculation(calc, atoms, f_t)
E_t = atoms.get_potential_energy()

atoms, calc = restart('h2o_fd_gs.gpw', txt='-')

# Mixed-spin n->3s occupation numbers
f_m = copy.deepcopy(f_gs)
f_m[0][2] -= 1.  # Remove one electron from homo-1 (n) spin up
f_m[0][4] += 1.  # Add one electron to lumo (3s) spin up

# MOM calculation for mixed-spin n->3s state
mom_calculation(calc, atoms, f_m)
E_m = atoms.get_potential_energy()
E_s = 2 * E_m - E_t  # Spin purified energy

print('Excitation energy triplet n->3s state: %.2f eV' % (E_t - E_gs))
print('Excitation energy singlet n->3s state: %.2f eV' % (E_s - E_gs))
# https://doi.org/10.1021/acs.jctc.8b00406
print('Experimental excitation energy triplet n->3s state: 9.46 eV')
print('Experimental excitation energy singlet n->3s state: 9.67 eV')
