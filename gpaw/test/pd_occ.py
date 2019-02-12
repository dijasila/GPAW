from gpaw.setup import create_setup
magmom = 1.0
hund = False
charge = 0.0
nspins = 2
s = create_setup('Pd')
f_si = s.calculate_initial_occupation_numbers(magmom, hund, charge, nspins)
print(f_si)
