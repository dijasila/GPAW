import copy

from ase.build import molecule

from gpaw import GPAW, restart
from gpaw import mom

atoms = molecule('H2O')
atoms.center(vacuum=5)

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

# Save the ground-state wave functions to initialize
# the excited-state calculations
calc.write('h2o_fd_gs.gpw', 'all')

f_n_gs = []
for s in range(2):
    f_n_gs.append(calc.get_occupation_numbers(spin=s))

f_n_t = copy.deepcopy(f_n_gs)
f_n_t[0][2] -= 1. # Remove one electron from
f_n_t[1][4] += 1.

mom.mom_calculation(calc, atoms, f_n_t)
E_t = atoms.get_potential_energy()

atoms, calc = restart('h2o_fd_gs.gpw', txt='-')

f_n_m = copy.deepcopy(f_n_gs)
f_n_m[0][2] -= 1.
f_n_m[0][4] += 1.

mom.mom_calculation(calc, atoms, f_n_m)
E_m = atoms.get_potential_energy()
print(E_m)



