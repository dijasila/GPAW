from gpaw.setup import create_setup

s = create_setup('Pd')
f_si = s.calculate_initial_occupation_numbers(magmom=1,
                                              hund=False,
                                              charge=0,
                                              nspins=2)
print(f_si)
magmom = (f_si[0] - f_si[1]).sum()
assert abs(magmom - 1.0) < 1e-10, f_si
