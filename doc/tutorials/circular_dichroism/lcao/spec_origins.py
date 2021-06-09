from gpaw.tddft.spectrum import rotatory_strength_spectrum

for i in range(1, 6):
    rotatory_strength_spectrum([f'mm-o{i}-{k}.dat' for k in 'xyz'],
                               f'rot_spec-o{i}.dat',
                               folding='Gauss', width=0.2,
                               e_min=0.0, e_max=10.0, delta_e=0.01)
