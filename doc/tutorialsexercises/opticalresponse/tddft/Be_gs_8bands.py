from ase import Atoms
from gpaw import GPAW

# Beryllium atom
atoms = Atoms(symbols='Be',
              positions=[(0, 0, 0)],
              pbc=False)

# Add 4.0 ang vacuum around the atom
atoms.center(vacuum=4.0)

# Create GPAW calculator
calc = GPAW(nbands=10, h=0.3)
# Attach calculator to atoms
atoms.calc = calc

# Calculate the ground state
energy = atoms.get_potential_energy()

# converge also the empty states (the density is converged already)
calc = calc.fixed_density(
    convergence={'bands': 8})

# Save the ground state
calc.write('Be_gs_8bands.gpw', 'all')
