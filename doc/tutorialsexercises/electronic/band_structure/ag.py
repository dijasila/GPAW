# web-page: Ag.png
from ase.build import bulk
from gpaw import GPAW

# Perform standard ground state calculation (with plane wave basis)
ag = bulk('Ag')
calc = GPAW(mode='pw',
            xc='LDA',
            kpts=(10, 10, 10),
            txt='Ag_LDA.txt')
ag.calc = calc
ag.get_potential_energy()
calc.write('Ag_LDA.gpw')

# Restart from ground state and fix potential:
calc = GPAW('Ag_LDA.gpw').fixed_density(
    nbands=16,
    basis='dzp',
    symmetry='off',
    convergence={'bands': 12},
    kpts={'path': 'WLGXWK', 'npoints': 100})

# Plot the band structure
band_structure = calc.band_structure()
band_structure.plot(filename='Ag.png', emax=20.0, show=True)
