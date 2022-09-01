import numpy as np
from ase.io import read
from ase.units import _e
from gpaw.berryphase import get_polarization_phase
from gpaw.mpi import world

phi_c = get_polarization_phase('BaTiO3.gpw')
atoms = read('BaTiO3.gpw')
cell_cv = atoms.cell * 1e-10
V = atoms.get_volume() * 1e-30
P = (phi_c / (2 * np.pi) % 1) @ cell_cv / V * _e
if world.rank == 0:
    with open('polarization_BaTiO3.out', 'w') as fd:
        print(f'P: {P} C/m^2', file=fd)
