from gpaw.tddft.spectrum import rotatory_strength_spectrum

for tag in ['COM', 'COM+x', 'COM+y', 'COM+z', '123']:
    rotatory_strength_spectrum([f'mm-{tag}-{k}.dat' for k in 'xyz'],
                               f'rot_spec-{tag}.dat',
                               folding='Gauss', width=0.2,
                               e_min=0.0, e_max=10.0, delta_e=0.01)
