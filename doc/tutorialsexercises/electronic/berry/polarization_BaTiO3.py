import numpy as np
from ase.units import _e
from gpaw import GPAW
from gpaw.berryphase import get_polarization_phase
from gpaw.mpi import world

# Create gpw-file with wave functions for all k-points in the BZ:
calc = GPAW('BaTiO3.gpw').fixed_density(symmetry='off')
calc.write('BaTiO3+wfs.gpw', mode='all')

phi_c = get_polarization_phase('BaTiO3+wfs.gpw')
cell_cv = calc.atoms.cell * 1e-10
V = calc.atoms.get_volume() * 1e-30
px, py, pz = (phi_c / (2 * np.pi) % 1) @ cell_cv / V * _e
if world.rank == 0:
    with open('polarization_BaTiO3.out', 'w') as fd:
        print(f'P: {pz} C/m^2', file=fd)
